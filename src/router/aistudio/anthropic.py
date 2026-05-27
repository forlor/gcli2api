"""
Anthropic Router - Handles Anthropic/Claude format API requests via AI Studio
通过 AI Studio 处理 Anthropic/Claude 格式请求的路由模块
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 标准库
import json

# 第三方库
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# 本地模块 - 配置和日志
from config import get_anti_truncation_max_attempts
from log import log

# 本地模块 - 工具和认证
from src.utils import (
    get_base_model_from_feature_model,
    is_anti_truncation_model,
    is_fake_streaming_model,
    authenticate_bearer,
)

# 本地模块 - 转换器（假流式需要）
from src.converter.fake_stream import (
    parse_response_for_fake_stream,
    build_anthropic_fake_stream_chunks,
)

# 本地模块 - 基础路由工具
from src.router.hi_check import is_health_check_request, create_health_check_response
from src.router.stream_passthrough import (
    build_streaming_response_or_error,
    prepend_async_item,
    read_first_async_item,
)

# 本地模块 - 数据模型
from src.models import ClaudeRequest, model_to_dict

# 本地模块 - Token估算
from src.token_estimator import estimate_input_tokens


# ==================== 路由器初始化 ====================

router = APIRouter()


# ==================== API 路由 ====================

@router.post("/aistudio/v1/messages")
async def messages(
    claude_request: ClaudeRequest,
    token: str = Depends(authenticate_bearer)
):
    """
    处理Anthropic/Claude格式的消息请求（流式和非流式）via AI Studio
    """
    log.debug(f"[AISTUDIO-ANTHROPIC] Request for model: {claude_request.model}")

    # 转换为字典
    normalized_dict = model_to_dict(claude_request)

    # 健康检查
    if is_health_check_request(normalized_dict, format="anthropic"):
        response = create_health_check_response(format="anthropic")
        return JSONResponse(content=response)

    # 处理模型名称和功能检测
    use_fake_streaming = is_fake_streaming_model(claude_request.model)
    use_anti_truncation = is_anti_truncation_model(claude_request.model)
    real_model = get_base_model_from_feature_model(claude_request.model)

    # 获取流式标志
    is_streaming = claude_request.stream

    # 对于抗截断模型的非流式请求，给出警告
    if use_anti_truncation and not is_streaming:
        log.warning("抗截断功能仅在流式传输时有效，非流式请求将忽略此设置")

    # 更新模型名为真实模型名
    normalized_dict["model"] = real_model

    # 转换为 Gemini 格式
    from src.converter.anthropic2gemini import anthropic_to_gemini_request
    gemini_dict = await anthropic_to_gemini_request(normalized_dict)

    # 添加 model 字段
    gemini_dict["model"] = real_model

    # 规范化 Gemini 请求（使用 geminicli 模式，标准 Gemini 格式）
    from src.converter.gemini_fix import normalize_gemini_request
    gemini_dict = await normalize_gemini_request(gemini_dict, mode="geminicli")

    # ========== 非流式请求 ==========
    if not is_streaming:
        from src.api.aistudio import non_stream_request
        response = await non_stream_request(body=gemini_dict)

        status_code = getattr(response, "status_code", 200)

        if hasattr(response, "body"):
            response_body = response.body.decode() if isinstance(response.body, bytes) else response.body
        elif hasattr(response, "content"):
            response_body = response.content.decode() if isinstance(response.content, bytes) else response.content
        else:
            response_body = str(response)

        try:
            gemini_response = json.loads(response_body)
        except Exception as e:
            log.error(f"Failed to parse Gemini response: {e}")
            raise HTTPException(status_code=500, detail="Response parsing failed")

        from src.converter.anthropic2gemini import gemini_to_anthropic_response
        anthropic_response = gemini_to_anthropic_response(
            gemini_response,
            real_model,
            status_code
        )

        return JSONResponse(content=anthropic_response, status_code=status_code)

    # ========== 流式请求 ==========

    # ========== 假流式生成器 ==========
    async def fake_stream_generator():
        from src.api.aistudio import non_stream_request

        response = await non_stream_request(body=gemini_dict)

        if hasattr(response, "status_code") and response.status_code != 200:
            log.error(f"Fake streaming got error response: status={response.status_code}")
            yield response
            return

        if hasattr(response, "body"):
            response_body = response.body.decode() if isinstance(response.body, bytes) else response.body
        elif hasattr(response, "content"):
            response_body = response.content.decode() if isinstance(response.content, bytes) else response.content
        else:
            response_body = str(response)

        try:
            gemini_response = json.loads(response_body)

            if "error" in gemini_response:
                log.error(f"Fake streaming got error in response body: {gemini_response['error']}")
                from src.converter.anthropic2gemini import gemini_to_anthropic_response
                anthropic_error = gemini_to_anthropic_response(
                    gemini_response,
                    real_model,
                    200
                )
                yield f"data: {json.dumps(anthropic_error)}\n\n".encode()
                yield "data: [DONE]\n\n".encode()
                return

            content, reasoning_content, finish_reason, images = parse_response_for_fake_stream(gemini_response)

            chunks = build_anthropic_fake_stream_chunks(content, reasoning_content, finish_reason, real_model, images)
            for chunk in chunks:
                yield f"data: {json.dumps(chunk)}\n\n".encode()

        except Exception as e:
            log.error(f"Response parsing failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)}})}\n\n".encode()

        yield "data: [DONE]\n\n".encode()

    # ========== 流式抗截断生成器 ==========
    async def anti_truncation_generator():
        from src.converter.anti_truncation import AntiTruncationStreamProcessor, apply_anti_truncation
        from src.api.aistudio import stream_request
        from src.converter.anthropic2gemini import gemini_stream_to_anthropic_stream
        from fastapi import Response

        max_attempts = await get_anti_truncation_max_attempts()

        # apply_anti_truncation 期望 {"model": ..., "request": {...}} 格式
        wrapped_payload = {"model": gemini_dict.get("model"), "request": {k: v for k, v in gemini_dict.items() if k != "model"}}
        anti_truncation_payload = apply_anti_truncation(wrapped_payload)

        # 恢复为 aistudio 的扁平结构
        at_body = dict(anti_truncation_payload.get("request", {}))
        at_body["model"] = anti_truncation_payload.get("model", gemini_dict.get("model"))

        first_attempt_stream = stream_request(body=at_body, native=False)
        try:
            first_chunk = await read_first_async_item(first_attempt_stream)
        except StopAsyncIteration:
            return

        if isinstance(first_chunk, Response):
            yield first_chunk
            return

        first_attempt_pending = True

        async def stream_request_wrapper(payload):
            nonlocal first_attempt_pending
            # payload 是 {"model": ..., "request": {...}} 格式，需解包
            body = dict(payload.get("request", {}))
            body["model"] = payload.get("model")

            if first_attempt_pending:
                first_attempt_pending = False
                stream_gen = prepend_async_item(first_chunk, first_attempt_stream)
            else:
                stream_gen = stream_request(body=body, native=False)
            return StreamingResponse(stream_gen, media_type="text/event-stream")

        processor = AntiTruncationStreamProcessor(
            stream_request_wrapper,
            anti_truncation_payload,
            max_attempts,
            enable_prefill_mode=True,
        )

        async def bytes_wrapper():
            async for chunk in processor.process_stream():
                if isinstance(chunk, str):
                    yield chunk.encode('utf-8')
                else:
                    yield chunk

        async for anthropic_chunk in gemini_stream_to_anthropic_stream(
            bytes_wrapper(),
            real_model,
            200
        ):
            if anthropic_chunk:
                yield anthropic_chunk

    # ========== 普通流式生成器 ==========
    async def normal_stream_generator():
        from src.api.aistudio import stream_request
        from fastapi import Response
        from src.converter.anthropic2gemini import gemini_stream_to_anthropic_stream

        stream_gen = stream_request(body=gemini_dict, native=False)
        try:
            first_chunk = await read_first_async_item(stream_gen)
        except StopAsyncIteration:
            return

        if isinstance(first_chunk, Response):
            yield first_chunk
            return

        async def gemini_chunk_wrapper():
            async for chunk in prepend_async_item(first_chunk, stream_gen):
                if isinstance(chunk, Response):
                    try:
                        error_content = chunk.body if isinstance(chunk.body, bytes) else (chunk.body or b'').encode('utf-8')
                        gemini_error = json.loads(error_content.decode('utf-8'))
                        from src.converter.anthropic2gemini import gemini_to_anthropic_response
                        anthropic_error = gemini_to_anthropic_response(
                            gemini_error,
                            real_model,
                            chunk.status_code
                        )
                        yield f"data: {json.dumps(anthropic_error)}\n\n".encode('utf-8')
                    except Exception:
                        yield f"data: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': 'Stream error'}})}\n\n".encode('utf-8')
                    yield b"data: [DONE]\n\n"
                    return
                else:
                    if isinstance(chunk, str):
                        yield chunk.encode('utf-8')
                    else:
                        yield chunk

        async for anthropic_chunk in gemini_stream_to_anthropic_stream(
            gemini_chunk_wrapper(),
            real_model,
            200
        ):
            if anthropic_chunk:
                yield anthropic_chunk

    # ========== 根据模式选择生成器 ==========
    if use_fake_streaming:
        return await build_streaming_response_or_error(fake_stream_generator())
    elif use_anti_truncation:
        log.info("启用流式抗截断功能")
        return await build_streaming_response_or_error(anti_truncation_generator())
    else:
        return await build_streaming_response_or_error(normal_stream_generator())


@router.post("/aistudio/v1/messages/count_tokens")
async def count_tokens(
    request: Request,
    _token: str = Depends(authenticate_bearer)
):
    """处理Anthropic格式的token计数请求"""
    try:
        payload = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": f"JSON 解析失败: {str(e)}"}}
        )

    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": "请求体必须为 JSON object"}}
        )

    if not payload.get("model") or not isinstance(payload.get("messages"), list):
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": "缺少必填字段：model / messages"}}
        )

    input_tokens = 0
    try:
        input_tokens = estimate_input_tokens(payload)
    except Exception as e:
        log.error(f"[AISTUDIO-ANTHROPIC] token 估算失败: {e}")

    return JSONResponse(content={"input_tokens": input_tokens})
