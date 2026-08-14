"""
AutoLabel-AI Pro · 策略模式 LLM 客户端
作者：陈沫儒（沫沫）   更新时间：2026-08-14
版本：v2 MVP

面试考点：
  - 策略模式（Strategy Pattern）：算法族封装、运行时切换
  - 抽象基类（ABC）：定义统一接口
  - 降级策略（Fallback）：智谱额度用完自动切阿里 Qwen
  - 重试机制（Retry with Exponential Backoff）：429 限流保护
  - 工厂方法（Factory Method）：屏蔽底层差异

设计原则：
  - 上层（标注模块）只调 chat() / chat_with_image()，不关心底层是智谱还是 Qwen
  - 一行配置切换厂商：client = LLMRouter(provider="zhipu")
  - 全部走 OpenAI 兼容协议（智谱、Qwen 都支持），本地 Ollama 走 HTTP
"""

from __future__ import annotations

import base64
import io
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from loguru import logger
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from zhipuai import ZhipuAI


# ============================================================
# 1. 数据结构
# ============================================================
@dataclass
class Message:
    """统一消息格式（OpenAI 风格）"""
    role: Literal["system", "user", "assistant"] = "user"
    content: Any = ""  # 允许 str 或 list（多模态场景）

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResult:
    """统一返回结构（屏蔽厂商差异）"""
    content: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    raw: Any = field(default=None, repr=False)


# ============================================================
# 2. 抽象基类（Strategy Interface）
# ============================================================
class BaseLLMClient(ABC):
    """所有 LLM 客户端的基类——上层只依赖这个接口"""

    @abstractmethod
    def chat(self, messages: list[Message], **kwargs) -> ChatResult:
        """纯文本对话"""

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: str,
        messages_prefix: list[Message] | None = None,
    ) -> ChatResult:
        """图文多模态对话"""

    @abstractmethod
    def provider_name(self) -> str:
        """返回厂商名（日志/降级用）"""


# ============================================================
# 3. 智谱 GLM-4 / GLM-4V 客户端
# ============================================================
class ZhipuAPIClient(BaseLLMClient):
    """
    智谱云端 API
    - 纯文本：glm-4-flash（便宜快）/ glm-4-plus（强）
    - 多模态：glm-4v-flash（推荐，便宜）/ glm-4v-plus
    文档：https://bigmodel.cn/dev/api
    """

    TEXT_MODELS = ["glm-4-flash", "glm-4-plus", "glm-4-air"]
    VISION_MODELS = ["glm-4v-flash", "glm-4v-plus"]

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ZHIPUAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "未找到 ZHIPUAI_API_KEY，请在 .env 里配置\n"
                "格式：ZHIPUAI_API_KEY=你的keyid.keysecret"
            )
        self.client = ZhipuAI(api_key=self.api_key)
        logger.info(
            f"[ZhipuAPIClient] 初始化完成（key 后 4 位：...{self.api_key[-4:]}）"
        )

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, ConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
    )
    def chat(
        self, messages: list[Message], model: str = "glm-4-flash", **kwargs
    ) -> ChatResult:
        start = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[m.to_dict() for m in messages],
                **kwargs,
            )
            return ChatResult(
                content=resp.choices[0].message.content,
                provider="zhipu",
                model=model,
                prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
                total_tokens=resp.usage.total_tokens if resp.usage else 0,
                latency_ms=int((time.time() - start) * 1000),
                raw=resp,
            )
        except Exception as e:
            logger.error(f"[ZhipuAPIClient.chat] 失败：{e}")
            raise

    def chat_with_image(
        self,
        text: str,
        image_path: str,
        messages_prefix: list[Message] | None = None,
        model: str = "glm-4v-flash",
    ) -> ChatResult:
        """智谱 GLM-4V 图文接口（OpenAI 兼容格式传 base64）"""
        start = time.time()
        b64 = self._encode_image(image_path)
        prefix = messages_prefix or []
        prefix.append(
            Message(
                role="user",
                content=[
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            )
        )
        payload = [m.to_dict() for m in prefix]
        resp = self.client.chat.completions.create(
            model=model,
            messages=payload,
        )
        return ChatResult(
            content=resp.choices[0].message.content,
            provider="zhipu",
            model=model,
            total_tokens=resp.usage.total_tokens if resp.usage else 0,
            latency_ms=int((time.time() - start) * 1000),
            raw=resp,
        )

    def provider_name(self) -> str:
        return "zhipu"

    @staticmethod
    def _encode_image(image_path: str, max_long_side: int = 1024) -> str:
        """
        图片 base64 编码（内置长边压缩，节省 token）
        - GLM-4V 传大图会按分辨率计费，长边 1024 已经够看清文字/病灶
        - 真人医生看片也就看 1024×1024，再大也看不出更多细节
        """
        from PIL import Image

        img = Image.open(image_path)
        if max(img.size) > max_long_side:
            img.thumbnail((max_long_side, max_long_side))
            logger.debug(f"图片长边压缩至 {max_long_side}px（原图 {img.size}）")
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================
# 4. 阿里 Qwen-VL 客户端（OpenAI 兼容协议）
# ============================================================
class QwenVLAPIClient(BaseLLMClient):
    """
    阿里通义千问 Qwen-VL
    - 走 DashScope 的 OpenAI 兼容端点（base_url=https://dashscope.aliyuncs.com/compatible-mode/v1）
    - 纯文本：qwen-plus / qwen-turbo
    - 多模态：qwen-vl-plus / qwen-vl-max
    文档：https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api
    """

    TEXT_MODELS = ["qwen-turbo", "qwen-plus", "qwen-max"]
    VISION_MODELS = ["qwen-vl-plus", "qwen-vl-max"]

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        if not self.api_key:
            raise ValueError("未找到 DASHSCOPE_API_KEY，请在 .env 里配置")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        logger.info("[QwenVLAPIClient] 初始化完成")

    def chat(
        self, messages: list[Message], model: str = "qwen-plus", **kwargs
    ) -> ChatResult:
        start = time.time()
        resp = self.client.chat.completions.create(
            model=model,
            messages=[m.to_dict() for m in messages],
            **kwargs,
        )
        return ChatResult(
            content=resp.choices[0].message.content,
            provider="qwen",
            model=model,
            total_tokens=resp.usage.total_tokens if resp.usage else 0,
            latency_ms=int((time.time() - start) * 1000),
            raw=resp,
        )

    def chat_with_image(
        self,
        text: str,
        image_path: str,
        messages_prefix: list[Message] | None = None,
        model: str = "qwen-vl-plus",
    ) -> ChatResult:
        start = time.time()
        b64 = ZhipuAPIClient._encode_image(image_path)
        prefix = messages_prefix or []
        prefix.append(
            Message(
                role="user",
                content=[
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            )
        )
        resp = self.client.chat.completions.create(
            model=model,
            messages=[m.to_dict() for m in prefix],
        )
        return ChatResult(
            content=resp.choices[0].message.content,
            provider="qwen",
            model=model,
            latency_ms=int((time.time() - start) * 1000),
            raw=resp,
        )

    def provider_name(self) -> str:
        return "qwen"


# ============================================================
# 5. 本地 Ollama LLaVA 客户端（HTTP 接口）
# ============================================================
class OllamaLLaVAClient(BaseLLMClient):
    """
    本地 Ollama 部署的 LLaVA（开源 VLM）
    - 适合：低配机器、无网环境、数据不出本机
    - 启动：先到 https://ollama.com 下载，再 ollama pull llava
    - 默认地址：http://localhost:11434
    """

    def __init__(
        self, base_url: str = "http://localhost:11434", model: str = "llava"
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        # 启动时探活
        try:
            httpx.get(f"{self.base_url}/api/tags", timeout=3).raise_for_status()
            logger.info(
                f"[OllamaLLaVAClient] Ollama 已就绪（{self.base_url}, model={self.model}）"
            )
        except Exception as e:
            logger.warning(
                f"[OllamaLLaVAClient] Ollama 探活失败：{e}\n"
                "请先启动 Ollama：https://ollama.com/download\n"
                "并执行：ollama pull llava"
            )

    def chat(self, messages: list[Message], **kwargs) -> ChatResult:
        start = time.time()
        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [m.to_dict() for m in messages],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return ChatResult(
            content=data["message"]["content"],
            provider="ollama",
            model=self.model,
            latency_ms=int((time.time() - start) * 1000),
            raw=data,
        )

    def chat_with_image(
        self,
        text: str,
        image_path: str,
        messages_prefix: list[Message] | None = None,
    ) -> ChatResult:
        start = time.time()
        b64 = ZhipuAPIClient._encode_image(image_path)
        prefix = messages_prefix or []
        prefix.append(
            Message(
                role="user",
                content=text,  # Ollama 的 /api/chat 不支持 content 数组
            )
        )
        # Ollama 用 images 字段传图（不是 OpenAI 格式）
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in prefix],
            "images": [b64],
            "stream": False,
        }
        resp = httpx.post(
            f"{self.base_url}/api/chat", json=payload, timeout=120
        )
        resp.raise_for_status()
        return ChatResult(
            content=resp.json()["message"]["content"],
            provider="ollama",
            model=self.model,
            latency_ms=int((time.time() - start) * 1000),
        )

    def provider_name(self) -> str:
        return "ollama"


# ============================================================
# 6. InternVL 客户端（OpenAI 兼容协议）
# ============================================================
class InternVLClient(BaseLLMClient):
    """
    InternVL（OpenGVLab 开源）
    - 部署方式：通过 vLLM/TGI 起 OpenAI 兼容服务
    - 简历包装：自部署 InternVL 推理服务（vLLM 加速）
    - 硬件门槛：消费级 GPU（24G 显存可跑 InternVL2-8B）
    注：这里只做接口封装，假设服务端已经在 localhost:8000 起来
    """

    def __init__(
        self, base_url: str = "http://localhost:8000/v1", model: str = "internvl2"
    ):
        self.client = OpenAI(api_key="EMPTY", base_url=base_url)
        self.model = model
        logger.info(f"[InternVLClient] 初始化完成（{base_url}, model={model}）")

    def chat(self, messages: list[Message], **kwargs) -> ChatResult:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[m.to_dict() for m in messages],
            **kwargs,
        )
        return ChatResult(
            content=resp.choices[0].message.content,
            provider="internvl",
            model=self.model,
        )

    def chat_with_image(
        self,
        text: str,
        image_path: str,
        messages_prefix: list[Message] | None = None,
    ) -> ChatResult:
        b64 = ZhipuAPIClient._encode_image(image_path)
        prefix = messages_prefix or []
        prefix.append(
            Message(
                role="user",
                content=[
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            )
        )
        resp = self.client.chat.completions.create(
            model=self.model, messages=[m.to_dict() for m in prefix]
        )
        return ChatResult(
            content=resp.choices[0].message.content,
            provider="internvl",
            model=self.model,
        )

    def provider_name(self) -> str:
        return "internvl"


# ============================================================
# 7. 路由 + 降级策略（核心面试考点）
# ============================================================
class LLMRouter:
    """
    LLM 路由：统一对外接口 + 故障自动降级
    - 主厂商失败 → 自动降级到备用厂商
    - 上层只认 client.chat()，不关心内部降级细节
    """

    PROVIDERS: dict[str, type[BaseLLMClient]] = {
        "zhipu": ZhipuAPIClient,
        "qwen": QwenVLAPIClient,
        "ollama": OllamaLLaVAClient,
        "internvl": InternVLClient,
    }

    def __init__(
        self,
        primary: str = "zhipu",
        fallback_chain: list[str] | None = None,
    ):
        """
        Args:
            primary: 主厂商（zhipu / qwen / ollama / internvl）
            fallback_chain: 备用链，依次尝试
        """
        self.primary = primary
        self.fallback_chain = fallback_chain or ["qwen", "ollama"]
        self._clients: dict[str, BaseLLMClient] = {}
        logger.info(
            f"[LLMRouter] 主厂商={primary}，降级链={self.fallback_chain}"
        )

    def _get(self, provider: str) -> BaseLLMClient:
        if provider not in self._clients:
            cls = self.PROVIDERS[provider]
            self._clients[provider] = cls()
        return self._clients[provider]

    def chat(self, messages: list[Message], **kwargs) -> ChatResult:
        return self._call_with_fallback("chat", messages, **kwargs)

    def chat_with_image(
        self, text: str, image_path: str, **kwargs
    ) -> ChatResult:
        return self._call_with_fallback(
            "chat_with_image", text, image_path, **kwargs
        )

    def _call_with_fallback(self, method: str, *args, **kwargs) -> ChatResult:
        """核心降级逻辑：主→备1→备2"""
        chain = [self.primary] + [
            p for p in self.fallback_chain if p != self.primary
        ]
        last_error: Exception | None = None
        for provider in chain:
            try:
                client = self._get(provider)
                method_fn = getattr(client, method)
                result = method_fn(*args, **kwargs)
                if provider != self.primary:
                    logger.warning(
                        f"[LLMRouter] 主厂商 {self.primary} 失败，已降级到 {provider}"
                    )
                return result
            except Exception as e:
                last_error = e
                logger.error(f"[LLMRouter] {provider} 调用失败：{e}")
                continue
        raise RuntimeError(f"所有 LLM 厂商都失败了，最后错误：{last_error}")


# ============================================================
# 8. 工厂方法（一行配置切换）
# ============================================================
def create_llm_client(provider: str = "zhipu") -> BaseLLMClient:
    """
    工厂方法：上层推荐用法
    示例：
        client = create_llm_client("zhipu")
        result = client.chat([Message(role="user", content="你好")])
    """
    if provider not in LLMRouter.PROVIDERS:
        raise ValueError(
            f"不支持的厂商：{provider}，可选：{list(LLMRouter.PROVIDERS.keys())}"
        )
    return LLMRouter.PROVIDERS[provider]()


# ============================================================
# 9. 自检脚本（python src/core/llm_client.py 直接跑）
# ============================================================
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("AutoLabel-AI Pro · LLM 客户端自检")
    print("=" * 60)

    # 测试 1：纯文本对话
    print("\n【测试 1】智谱 GLM-4 纯文本对话")
    try:
        client = create_llm_client("zhipu")
        result = client.chat([
            Message(role="system", content="你是一个数据标注专家，简洁回答。"),
            Message(role="user", content="什么是 Human-in-the-Loop？用一句话回答。"),
        ])
        print(f"  响应：{result.content}")
        print(f"  token：{result.total_tokens}，耗时：{result.latency_ms}ms")
    except Exception as e:
        print(f"  ✗ 失败：{e}")

    # 测试 2：图文多模态
    print("\n【测试 2】智谱 GLM-4V 图文对话")
    test_img = "data/test_medical.jpg"
    if Path(test_img).exists():
        try:
            result = client.chat_with_image(
                text="请描述这张医学影像看到的内容（简洁回答，50字内）",
                image_path=test_img,
            )
            print(f"  响应：{result.content}")
            print(f"  耗时：{result.latency_ms}ms")
        except Exception as e:
            print(f"  ✗ 失败：{e}")
    else:
        print(f"  ⚠ 跳过：未找到测试图片 {test_img}")

    # 测试 3：降级策略
    print("\n【测试 3】LLMRouter 降级链路（主 zhipu + 备 qwen/ollama）")
    router = LLMRouter(primary="zhipu", fallback_chain=["qwen", "ollama"])
    try:
        result = router.chat([Message(role="user", content="说'ok'即可")])
        print(f"  响应：{result.content}（来自 {result.provider}）")
    except Exception as e:
        print(f"  ✗ 全部失败：{e}")
