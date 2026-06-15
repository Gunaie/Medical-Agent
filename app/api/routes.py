# app/api/routes.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import uuid
import logging

from app.services.chat_service import process_chat, process_chat_stream

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户输入的问题")
    session_id: Optional[str] = Field(None, description="会话ID，不传则自动生成")


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    source: Optional[str] = None


@router.get("/", tags=["Root"])
async def root():
    return {
        "status": "online",
        "service": "MedAgent Enterprise",
        "message": "Welcome to Medical Agent API. Please use /docs for documentation."
    }


@router.post("/v1/chat", tags=["Chat"], response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """原有非流式接口，保持不变"""
    sid = request.session_id or str(uuid.uuid4())
    try:
        result = await process_chat(session_id=sid, query=request.query)
        return ChatResponse(
            answer=result["answer"],
            session_id=sid,
            source=result.get("source")
        )
    except Exception as e:
        logger.error(f"Agent 系统异常: {str(e)}", exc_info=True)
        return ChatResponse(
            answer=f"⚠️ 系统处理异常，请稍后重试。(Error: {str(e)})",
            session_id=sid,
            source=None
        )


@router.post("/v1/chat/stream", tags=["Chat Stream"])
async def chat_stream_endpoint(request: ChatRequest):
    """新增：SSE 流式对话接口"""
    sid = request.session_id or str(uuid.uuid4())
    return StreamingResponse(
        process_chat_stream(session_id=sid, query=request.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )