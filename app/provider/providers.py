# providers/openai.py
import asyncio
import json
from typing import AsyncIterator, Dict, Tuple, Optional
from client_pool import pool
import httpx

OPENAI_BASE = "https://api.openai.com"
CHAT_URL = "/v1/chat/completions"  # 非流式/流式统一端点

def _auth_headers(key: str) -> dict:
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

async def chat_stream(
    api_key: str,
    model: str,
    messages: list,
    extra: Optional[dict] = None,
) -> AsyncIterator[Tuple[str, Optional[Dict]]]:
    """
    以流式方式调用 OpenAI Chat Completions：
    产出 ("chunk", {"text": "..."}), ... 最后产出 ("usage", {...}) 和 ("completed", {"full": "..."}).
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if extra:
        payload.update(extra)

    # 通过连接池复用
    resp = await pool.request_with_retry(
        OPENAI_BASE, "POST", CHAT_URL,
        headers=_auth_headers(api_key), json=payload, stream=True
    )
    if resp.is_error:
        text = await resp.aread()
        raise httpx.HTTPStatusError(
            f"OpenAI error {resp.status_code}: {text.decode('utf-8','ignore')}",
            request=resp.request, response=resp
        )

    full_text = []
    usage = None

    async with resp.aiter_lines() as lines:
        async for line in lines:
            if not line:
                continue
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                # 增量内容
                delta = obj.get("choices", [{}])[0].get("delta", {})
                if "content" in delta and delta["content"]:
                    t = delta["content"]
                    full_text.append(t)
                    yield ("chunk", {"text": t})

                # 部分 SDK 会把 usage 放在最后一帧的 "usage" 字段
                if "usage" in obj:
                    usage = obj["usage"]

    # 完成与 usage
    final_text = "".join(full_text)
    if usage:
        yield ("usage", usage)
    yield ("completed", {"full": final_text})


async def chat_once(
    api_key: str,
    model: str,
    messages: list,
    extra: Optional[dict] = None,
) -> Tuple[str, Dict]:
    """
    非流式一次性拿完整结果 + usage（供标题生成等）。
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if extra:
        payload.update(extra)

    resp = await pool.request_with_retry(
        OPENAI_BASE, "POST", CHAT_URL,
        headers=_auth_headers(api_key), json=payload, stream=False
    )
    if resp.is_error:
        raise httpx.HTTPStatusError(
            f"OpenAI error {resp.status_code}: {await resp.aread()}",
            request=resp.request, response=resp
        )
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text, usage
