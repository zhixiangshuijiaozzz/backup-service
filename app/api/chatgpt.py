# app/main.py
import asyncio
import base64
import json
import logging
import os
import re
import time
from typing import Optional, AsyncGenerator, List, Dict, Any, Union

import httpx
from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from httpx import ProxyError, ConnectError, ReadTimeout, PoolTimeout

from pydantic import BaseModel
from typing import Optional

# ===== OpenAI / Anthropic SDK =====
try:
    from openai import AsyncOpenAI, OpenAI
except Exception as _:
    from openai import AsyncOpenAI, OpenAI  # noqa

try:
    from anthropic import AsyncAnthropic
except Exception:
    AsyncAnthropic = None  # 如果未安装，相关接口会抛异常

# =========================
# 日志
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("py-proxy")

# =========================
# FastAPI
# =========================
app = FastAPI(title="LLM Proxy", version="1.1.0")
router = APIRouter(tags=["ChatGateway"])

# =========================
# Pydantic 模型
# =========================
class ChatContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None
    file_url: Optional[Dict[str, Any]] = None

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ChatContentItem]]

class ChatRequest(BaseModel):
    model: str
    key: str
    messages: List[ChatMessage]


# 已有 ChatRequest 不动，新增一个专门做摘要的请求体
class SummaryRequest(BaseModel):
    model: str              # 走哪个 OpenAI 模型，比如 gpt-4.1-mini 等
    key: str                # 前端传来的 API Key
    text: str               # 需要被总结的原文
    max_words: Optional[int] = 200  # 可选：限制摘要长度（字/词，自己文案里说明）
    language: Optional[str] = "zh"  # 可选：摘要语言，默认中文


# =========================
# 公共工具
# =========================
def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}****{key[-4:]}"

def transform_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """
    将你的前端结构（字符串或多段 content）转换为：
    - OpenAI chat.completions 可接受的 messages
    - Anthropic messages (tool-free 的简单 user/assistant 文本/图像)
    """
    transformed = []
    for m in messages:
        if isinstance(m.content, str):
            transformed.append({"role": m.role, "content": m.content})
            continue

        msg_dict = {"role": m.role, "content": []}
        for c in m.content:
            if c.type == "text" and c.text is not None:
                msg_dict["content"].append({"type": "text", "text": c.text})
            elif c.type == "image_url" and c.image_url and "url" in c.image_url:
                msg_dict["content"].append({"type": "image_url", "image_url": {"url": c.image_url["url"]}})
            elif c.type == "file_url" and c.file_url and "url" in c.file_url:
                msg_dict["content"].append({"type": "text", "text": f"[FILE] {c.file_url['url']}"})
        transformed.append(msg_dict)
    return transformed

def safe_dump_req(req: ChatRequest) -> str:
    try:
        d = req.model_dump() if hasattr(req, "model_dump") else jsonable_encoder(req)
        d["key"] = mask_key(d.get("key", ""))
        return json.dumps(d, ensure_ascii=False)
    except Exception as e:
        return f"<dump req failed: {e}>"

def sse_frame(event: Optional[str], data: str) -> str:
    """
    SSE 规范：data 里不能直接带换行；多行需要逐行加 "data: "。
    注意：不要在这里做 JSON 转义，直接原样逐行输出。
    """
    lines = data.split("\n")
    buf: List[str] = []
    if event:
        buf.append(f"event: {event}")
    for ln in lines:
        buf.append(f"data: {ln}")

    return "\n".join(buf) + "\n\n"

# =========================
# 运行参数 & 代理
# =========================


# 优先从 HTTPS/HTTP 大小写环境变量读取
PROXY_URL = (
    os.getenv("HTTPS_PROXY")
    or os.getenv("https_proxy")
    or os.getenv("HTTP_PROXY")
    or os.getenv("http_proxy")
    or None
)

TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=120.0)

# 统一的连接池（HTTP/2 + KeepAlive）
shared_async_httpx = httpx.AsyncClient(
    verify=False,
    timeout=TIMEOUT,
    http2=True,  # ✅ 开启 HTTP/2
    limits=httpx.Limits(
        max_connections=200,
        max_keepalive_connections=50,
        keepalive_expiry=30.0,
    ),
    # proxy=PROXY_URL,  # ✅ 启用代理（如未设置则为 None）
)

logger.info(
    "[startup] http2=%s, proxy=%s",
    getattr(shared_async_httpx, "_transport", None).__class__.__name__ if getattr(shared_async_httpx, "_transport", None) else "unknown",
    PROXY_URL or "<direct>"
)

openai_async = AsyncOpenAI(api_key="__placeholder__", http_client=shared_async_httpx)
anthropic_async = AsyncAnthropic(api_key="__placeholder__", http_client=shared_async_httpx) if AsyncAnthropic else None

# =========================
# Token 估算兜底（当 SDK 无 usage）
# =========================
def estimate_tokens_by_chars(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)

def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += estimate_tokens_by_chars(c)
        elif isinstance(c, list):
            for item in c:
                if item.get("type") == "text":
                    total += estimate_tokens_by_chars(item.get("text") or "")
    return total

# =========================
# 图片识别（OpenAI）
# =========================
@router.post("/chatgpt/image")
async def process_image(
    model: str = Form(...),
    key: str = Form(...),
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    logger.info("[/chatgpt/image] 收到图片请求: model=%s, key=%s, filename=%s, content_type=%s",
                model, mask_key(key), image.filename, image.content_type)
    try:
        raw = await image.read()
        await image.seek(0)
        base64_image = base64.b64encode(raw).decode("utf-8")

        client = OpenAI(
            api_key=key,
            http_client=httpx.Client(
                verify=False,
                timeout=TIMEOUT,
                http2=True,
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                    keepalive_expiry=30.0,
                ),
                proxy=PROXY_URL,
            )
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{image.content_type or 'image/jpeg'};base64,{base64_image}"}}
                ]
            }]
        )
        content = resp.choices[0].message.content if resp.choices else ""
        return {"content": content}
    except Exception as e:
        logger.error("[/chatgpt/image] 处理出错: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# =========================
# OpenAI: 流式（SSE 规范版）
# =========================
@router.post("/chatgpt/stream")
async def chatgpt_stream(req: ChatRequest):
    logger.info("[/chatgpt/stream] 收到请求: %s", safe_dump_req(req))
    start = time.time()
    try:
        messages = transform_messages(req.messages)
        derived = openai_async.with_options(api_key=req.key)

        async def sse() -> AsyncGenerator[str, None]:
            usage_final = None
            acc_text_len = 0
            first_chunk_at = None
            chunks = 0

            try:
                started_at = time.time()
                completion = await derived.chat.completions.create(
                    model=req.model,
                    messages=messages,
                    stream=True,
                    # 你也可以按需暴露 temperature/top_p
                    # temperature=0.5,
                    # top_p=0.9,
                )
                async for chunk in completion:
                    # 增量文本
                    if getattr(chunk, "choices", None):
                        delta = chunk.choices[0].delta
                        if delta and getattr(delta, "content", None):
                            txt = delta.content
                            chunks += 1
                            acc_text_len += len(txt)
                            now = time.time()
                            if first_chunk_at is None:
                                first_chunk_at = now
                                logger.info("[/chatgpt/stream] 首个chunk到达 TTFB=%.0fms",
                                            (now - started_at) * 1000)
                            if chunks % 50 == 0:
                                dur = max(1e-6, now - first_chunk_at)
                                cps = acc_text_len / dur
                                logger.info("[/chatgpt/stream] chunk=%d, acc_len=%d, rate=%.1f chars/s",
                                            chunks, acc_text_len, cps)
                            yield sse_frame("chunk", txt)

                    # 某些 SDK/模型在流中会带 usage
                    if getattr(chunk, "usage", None) and usage_final is None:
                        u = chunk.usage
                        usage_final = {
                            "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                            "cached_input_tokens": getattr(u, "cached_prompt_tokens", 0) or 0,
                            "output_tokens": getattr(u, "completion_tokens", 0) or 0,
                        }

            except Exception as e:
                logger.error("[/chatgpt/stream] 异常: %s", str(e), exc_info=True)
                yield sse_frame("error", f"{type(e).__name__}: {str(e)}")
            finally:
                cost = int((time.time() - start) * 1000)
                # 总体速率汇总
                if first_chunk_at:
                    total_dur = max(1e-6, time.time() - first_chunk_at)
                    cps = acc_text_len / total_dur
                    logger.info("[/chatgpt/stream] 总结：chunks=%d, 输出字符=%d, 平均速率=%.1f chars/s",
                                chunks, acc_text_len, cps)

                logger.info("[/chatgpt/stream] 完成，耗时=%dms", cost)
                # 兜底 usage
                if not usage_final:
                    usage_final = {
                        "input_tokens": 0,
                        "cached_input_tokens": 0,
                        "output_tokens": 0,
                    }
                # usage 事件（单行 JSON）
                yield "event: usage\ndata: " + json.dumps(usage_final, ensure_ascii=False) + "\n\n"
                # completed 仅发哨兵，避免重放全文
                yield sse_frame("completed", "[DONE]")

        # 规范 SSE 响应
        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # Nginx 等反代避免缓冲
                # "Connection": "keep-alive",  # 可选
            }
        )
    except Exception as e:
        logger.error("[/chatgpt/stream] 初始化出错: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error initializing chat: {str(e)}")

# =========================
# OpenAI: 非流式
# =========================
@router.post("/chatgpt")
async def chatgpt_completion(req: ChatRequest):
    logger.info("[/chatgpt] 收到请求: %s", safe_dump_req(req))
    start = time.time()
    try:
        messages = transform_messages(req.messages)
        derived = openai_async.with_options(api_key=req.key)

        completion = await derived.chat.completions.create(
            model=req.model,
            messages=messages,
            stream=False,
            # temperature=0.5,
            # top_p=0.9,
        )
        content = completion.choices[0].message.content if completion.choices else "No content generated."
        usage = getattr(completion, "usage", None)
        cost = int((time.time() - start) * 1000)
        logger.info("[/chatgpt] 成功，length=%d, cost=%dms", len(content or ""), cost)
        return {"content": content, "usage": jsonable_encoder(usage) if usage else None}
    except Exception as e:
        logger.error("[/chatgpt] 出错: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during chat completion: {str(e)}")

@router.post("/chatgpt/summary")
async def chatgpt_summary(req: SummaryRequest):
    """
    使用 OpenAI 模型对文本做摘要的简单接口
    """
    logger.info("[/chatgpt/summary] 收到请求: model=%s, key=%s, text_len=%d",
                req.model, mask_key(req.key), len(req.text or ""))

    start = time.time()
    try:
        derived = openai_async.with_options(api_key=req.key)

        # 根据语言简单设定 system 提示词
        if req.language == "en":
            system_prompt = (
                "You are a professional summarization assistant. "
                "Please generate a concise English summary of the user's content."
            )
            length_hint = f"The summary should be within about {req.max_words} words."
        else:
            # 默认中文
            system_prompt = (
                "你是一个专业的中文文本摘要助手。"
                "请在保留关键信息的前提下，生成结构清晰、可读性高的摘要。"
            )
            length_hint = f"摘要长度控制在大约 {req.max_words} 字以内。"

        user_content = (
            f"{length_hint}\n\n"
            f"以下是需要总结的原文：\n\n{req.text}"
        )

        completion = await derived.chat.completions.create(
            model=req.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            stream=False,
            # 可按需开放 temperature / top_p
            # temperature=0.3,
            # top_p=0.9,
        )

        summary_text = completion.choices[0].message.content if completion.choices else ""
        usage = getattr(completion, "usage", None)
        cost = int((time.time() - start) * 1000)

        logger.info("[/chatgpt/summary] 成功，summary_len=%d, cost=%dms",
                    len(summary_text or ""), cost)

        return {
            "summary": summary_text,
            "usage": jsonable_encoder(usage) if usage else None,
        }

    except Exception as e:
        logger.error("[/chatgpt/summary] 出错: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during summarization: {str(e)}")


# =========================
# Anthropic: 流式（SSE 规范版）
# =========================
@router.post("/claude/stream")
async def claude_stream(req: ChatRequest):
    if anthropic_async is None:
        raise HTTPException(status_code=500, detail="Anthropic SDK 未安装")
    logger.info("[/claude/stream] 收到请求: %s", safe_dump_req(req))
    start = time.time()
    try:
        messages = transform_messages(req.messages)
        derived = anthropic_async.with_options(api_key=req.key)

        async def sse() -> AsyncGenerator[str, None]:
            est_input = estimate_messages_tokens(messages)
            est_output_acc_text_len = 0
            usage_final = None
            first_chunk_at = None
            chunks = 0

            try:
                started_at = time.time()
                stream = await derived.messages.create(
                    model=req.model,
                    messages=messages,
                    stream=True,
                )
                async for ev in stream:
                    t = getattr(ev, "type", None)
                    if t == "content_block_delta" and hasattr(ev, "delta"):
                        txt = getattr(ev.delta, "text", None)
                        if txt:
                            chunks += 1
                            est_output_acc_text_len += len(txt)
                            now = time.time()
                            if first_chunk_at is None:
                                first_chunk_at = now
                                logger.info("[/claude/stream] 首个chunk到达 TTFB=%.0fms",
                                            (now - started_at) * 1000)
                            if chunks % 50 == 0:
                                dur = max(1e-6, now - first_chunk_at)
                                cps = est_output_acc_text_len / dur
                                logger.info("[/claude/stream] chunk=%d, acc_len=%d, rate=%.1f chars/s",
                                            chunks, est_output_acc_text_len, cps)
                            yield sse_frame("chunk", txt)
                    elif t == "content_block_start" and hasattr(ev, "content_block"):
                        txt = getattr(ev.content_block, "text", None)
                        if txt:
                            chunks += 1
                            est_output_acc_text_len += len(txt)
                            now = time.time()
                            if first_chunk_at is None:
                                first_chunk_at = now
                                logger.info("[/claude/stream] 首个chunk到达 TTFB=%.0fms",
                                            (now - started_at) * 1000)
                            yield sse_frame("chunk", txt)

            except Exception as e:
                logger.error("[/claude/stream] 异常: %s", str(e), exc_info=True)
                yield sse_frame("error", f"{type(e).__name__}: {str(e)}")
            finally:
                cost = int((time.time() - start) * 1000)
                if first_chunk_at:
                    total_dur = max(1e-6, time.time() - first_chunk_at)
                    cps = est_output_acc_text_len / total_dur
                    logger.info("[/claude/stream] 总结：chunks=%d, 输出字符=%d, 平均速率=%.1f chars/s",
                                chunks, est_output_acc_text_len, cps)

                logger.info("[/claude/stream] 完成，耗时=%dms", cost)
                if not usage_final:
                    out_tokens = estimate_tokens_by_chars("x" * est_output_acc_text_len)
                    usage_final = {"input_tokens": est_input, "cached_input_tokens": 0, "output_tokens": out_tokens}
                # usage 事件（单行 JSON）
                yield "event: usage\ndata: " + json.dumps(usage_final, ensure_ascii=False) + "\n\n"
                # completed 仅发哨兵
                yield sse_frame("completed", "[DONE]")

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
    except Exception as e:
        logger.error("[/claude/stream] 初始化出错: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error initializing chat: {str(e)}")

# =========================
# Anthropic: 非流式
# =========================
@router.post("/claude")
async def claude_completion(req: ChatRequest):
    if anthropic_async is None:
        raise HTTPException(status_code=500, detail="Anthropic SDK 未安装")
    logger.info("[/claude] 收到请求: %s", safe_dump_req(req))
    start = time.time()
    try:
        messages = transform_messages(req.messages)
        derived = anthropic_async.with_options(api_key=req.key)

        completion = await derived.messages.create(
            model=req.model,
            messages=messages,
            max_tokens=DEFAULT_MAX_TOKENS,
            stream=False,
        )
        if not completion.content or not completion.content[0].text:
            raise HTTPException(status_code=500, detail="Claude API 返回的内容为空")

        content = completion.content[0].text
        cost = int((time.time() - start) * 1000)
        logger.info("[/claude] 成功，length=%d, cost=%dms", len(content or ""), cost)
        return Response(content=content, media_type="text/markdown")
    except Exception as e:
        logger.error("[/claude] 出错: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during chat completion: {str(e)}")

# =========================
# 诊断
# =========================
@router.get("/diag/openai")
async def diag_openai(key: str):
    try:
        r = await shared_async_httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        return {"status": r.status_code, "ok": (r.status_code == 200), "proxy": PROXY_URL or "<direct>"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": type(e).__name__, "message": str(e)})

# =========================
# 优雅关闭
# =========================
@app.on_event("shutdown")
async def shutdown_event():
    try:
        await shared_async_httpx.aclose()
    except Exception:
        pass

# 注册路由
app.include_router(router)
