"""
AI Studio API Client - Direct calls to Google AI Studio with static API key.
AI Studio API 客户端 - 使用静态 API Key 直接调用 Google AI Studio
"""

import asyncio
import json
from typing import Any, Dict, Optional

from fastapi import Response
from config import get_aistudio_api_key, get_aistudio_base_url, get_retry_429_enabled, get_retry_429_max_retries, get_retry_429_interval
from log import log
from src.httpx_client import stream_post_async, post_async


async def _get_config_or_error() -> tuple[str, str]:
    """
    获取 AI Studio 配置，未配置时抛出异常。

    Returns:
        (base_url, api_key)

    Raises:
        ValueError: 未配置 API Key
    """
    api_key = await get_aistudio_api_key()
    if not api_key:
        raise ValueError("AI Studio API Key 未配置，请在控制面板中设置 aistudio_api_key")
    base_url = await get_aistudio_base_url()
    return base_url.rstrip("/"), api_key


def _build_url(base_url: str, model: str, api_key: str, stream: bool = False) -> str:
    """构建 AI Studio API 请求 URL"""
    if stream:
        return f"{base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse&key={api_key}"
    return f"{base_url}/v1beta/models/{model}:generateContent?key={api_key}"


def _is_retryable(status_code: int) -> bool:
    return status_code in (429, 500, 503)


async def stream_request(
    body: Dict[str, Any],
    native: bool = False,
    headers: Optional[Dict[str, str]] = None,
):
    """
    流式请求 AI Studio API

    Args:
        body: Gemini 格式请求体（包含 model 和 contents 等）
        native: 是否返回原生 bytes 流
        headers: 额外请求头

    Yields:
        Response 对象（错误时）或流数据（成功时）
    """
    model_name = body.get("model", "")

    try:
        base_url, api_key = await _get_config_or_error()
    except ValueError as e:
        yield Response(
            content=json.dumps({"error": str(e)}),
            status_code=400,
            media_type="application/json",
        )
        return

    target_url = _build_url(base_url, model_name, api_key, stream=True)

    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    # 从 body 中提取 Gemini 请求体（去掉 model 字段）
    request_body = {k: v for k, v in body.items() if k != "model"}

    retry_enabled = await get_retry_429_enabled()
    max_retries = await get_retry_429_max_retries()
    retry_interval = await get_retry_429_interval()

    for attempt in range(max_retries + 1):
        success_recorded = False

        try:
            async for chunk in stream_post_async(
                url=target_url,
                body=request_body,
                native=native,
                headers=req_headers,
            ):
                if isinstance(chunk, Response):
                    status_code = chunk.status_code
                    error_body = ""
                    try:
                        error_body = chunk.body.decode("utf-8") if isinstance(chunk.body, bytes) else str(chunk.body)
                    except Exception:
                        pass

                    if _is_retryable(status_code) and retry_enabled and attempt < max_retries:
                        log.warning(
                            f"[AISTUDIO STREAM] 请求失败 (status={status_code}), "
                            f"重试 {attempt + 1}/{max_retries}"
                        )
                        await asyncio.sleep(retry_interval)
                        break
                    else:
                        log.error(
                            f"[AISTUDIO STREAM] 请求失败 (status={status_code}): "
                            f"{error_body[:300]}"
                        )
                        yield chunk
                        return
                else:
                    if not success_recorded:
                        success_recorded = True
                        log.debug(f"[AISTUDIO STREAM] 开始接收流式响应，模型: {model_name}")
                    yield chunk

            if success_recorded:
                log.debug(f"[AISTUDIO STREAM] 流式响应完成，模型: {model_name}")
                return

            # 空回复重试
            if retry_enabled and attempt < max_retries:
                continue

        except Exception as e:
            log.error(f"[AISTUDIO STREAM] 异常: {e}")
            if retry_enabled and attempt < max_retries:
                await asyncio.sleep(retry_interval)
                continue

    # 所有重试耗尽
    yield Response(
        content=json.dumps({"error": "AI Studio 请求失败，所有重试均已耗尽"}),
        status_code=429,
        media_type="application/json",
    )


async def non_stream_request(
    body: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
) -> Response:
    """
    非流式请求 AI Studio API

    Args:
        body: Gemini 格式请求体（包含 model 和 contents 等）
        headers: 额外请求头

    Returns:
        Response 对象
    """
    model_name = body.get("model", "")

    try:
        base_url, api_key = await _get_config_or_error()
    except ValueError as e:
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=400,
            media_type="application/json",
        )

    target_url = _build_url(base_url, model_name, api_key, stream=False)

    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    request_body = {k: v for k, v in body.items() if k != "model"}

    retry_enabled = await get_retry_429_enabled()
    max_retries = await get_retry_429_max_retries()
    retry_interval = await get_retry_429_interval()
    last_error_response = None

    for attempt in range(max_retries + 1):
        try:
            response = await post_async(
                url=target_url,
                json=request_body,
                headers=req_headers,
                timeout=300.0,
            )

            if response.status_code == 200:
                log.debug(f"[AISTUDIO] 非流式响应成功，模型: {model_name}")
                return Response(
                    content=response.content,
                    status_code=200,
                    headers=dict(response.headers),
                )

            error_text = ""
            try:
                error_text = response.text
            except Exception:
                pass

            last_error_response = Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
            )

            if _is_retryable(response.status_code) and retry_enabled and attempt < max_retries:
                log.warning(
                    f"[AISTUDIO] 非流式请求失败 (status={response.status_code}), "
                    f"重试 {attempt + 1}/{max_retries}"
                )
                await asyncio.sleep(retry_interval)
                continue

            log.error(
                f"[AISTUDIO] 非流式请求失败 (status={response.status_code}): "
                f"{error_text[:300]}"
            )
            return last_error_response

        except Exception as e:
            log.error(f"[AISTUDIO] 非流式请求异常: {e}")
            if retry_enabled and attempt < max_retries:
                await asyncio.sleep(retry_interval)
                continue

    if last_error_response:
        return last_error_response
    return Response(
        content=json.dumps({"error": "AI Studio 请求失败，所有重试均已耗尽"}),
        status_code=500,
        media_type="application/json",
    )
