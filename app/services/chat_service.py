# app/services/chat_service.py
import asyncio
import json
from functools import lru_cache

from langchain_core.messages import HumanMessage, AIMessage

from app.core.agent_builder import build_medical_agent
from app.memory.session_store import SessionManager
from app.utils.logger import logger
from app.config.settings import settings

session_manager = SessionManager(redis_url=settings.REDIS_URL)


@lru_cache(maxsize=1)
def _get_agent_template():
    """缓存 Agent 实例，避免重复构建"""
    logger.info("Initializing Medical ReAct Agent Template...")
    return build_medical_agent()


def _prepare_agent_input(session_id: str, query: str):
    """提取公共的历史消息处理逻辑"""
    chat_history_obj = session_manager.get_session(session_id)
    history_messages = list(chat_history_obj.messages)
    max_history = 10
    if len(history_messages) > max_history * 2:
        history_messages = history_messages[-(max_history * 2):]

    return {
        "messages": history_messages + [HumanMessage(content=query)]
    }, chat_history_obj


def _save_session_async(session_id: str, chat_history_obj):
    """
    ✅ 【修复】异步保存会话到 Redis，不阻塞主流程

    使用 asyncio.create_task 实现真正的异步保存，
    即使 Redis 写入失败也不会影响用户响应。
    """

    async def _do_save():
        try:
            session_manager.save_session(session_id)
            logger.info("Session saved to Redis", session_id=session_id)
        except Exception as e:
            logger.warning(f"Redis session save failed for {session_id}: {e}")

    # 在已有事件循环中创建后台任务
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_do_save())
    except RuntimeError:
        # 如果没有运行中的事件循环，同步执行（fallback）
        try:
            session_manager.save_session(session_id)
        except Exception as e:
            logger.warning(f"Redis session save failed for {session_id}: {e}")


async def process_chat(session_id: str, query: str) -> dict:
    """原有非流式处理逻辑"""
    try:
        agent = _get_agent_template()
        agent_input, chat_history_obj = _prepare_agent_input(session_id, query)

        loop = asyncio.get_event_loop()
        response_data = await loop.run_in_executor(None, agent.invoke, agent_input)

        messages = response_data.get("messages", [])
        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                answer = msg.content
                break

        if not answer:
            answer = "抱歉，我暂时无法回答这个问题，请换个方式提问。"

        source_files = []
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == "tool":
                try:
                    content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if isinstance(content, dict) and "sources" in content:
                        source_files.extend(content["sources"])
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue

        # ✅ 【修复】保存用户消息和 AI 回复到内存
        try:
            chat_history_obj.add_messages([
                HumanMessage(content=query),
                AIMessage(content=answer)
            ])
        except Exception as mem_err:
            logger.warning(f"Memory save failed for {session_id}: {mem_err}")

        # ✅ 【修复】异步持久化到 Redis
        _save_session_async(session_id, chat_history_obj)

        return {
            "answer": answer,
            "source": ", ".join(list(set(source_files))) if source_files else None
        }

    except Exception as e:
        logger.error("ReAct_Agent_Failed", error=str(e), session_id=session_id)
        return {"answer": f"系统处理异常，请稍后重试。(Error: {str(e)})", "source": None}


async def process_chat_stream(session_id: str, query: str):
    """SSE 流式处理生成器"""
    agent = _get_agent_template()
    agent_input, chat_history_obj = _prepare_agent_input(session_id, query)

    full_answer = ""
    source_files = []

    try:
        yield f"data: {json.dumps({'event': 'status', 'content': '正在分析您的问题...'})}\n\n"

        async for event in agent.astream_events(agent_input, version="v2"):
            kind = event.get("event")

            # 1. 安全提取 AI 增量回复
            if kind == "on_chat_model_stream":
                try:
                    chunk = event.get("data", {}).get("chunk")

                    if chunk is None:
                        continue

                    delta = ""
                    if hasattr(chunk, "content"):
                        delta = chunk.content or ""
                    elif isinstance(chunk, dict):
                        delta = chunk.get("content", "") or ""
                    elif isinstance(chunk, str):
                        delta = chunk

                    if delta and isinstance(delta, str) and delta.strip():
                        full_answer += delta
                        yield f"data: {json.dumps({'event': 'message', 'content': delta})}\n\n"

                except (IndexError, TypeError, AttributeError) as e:
                    logger.warning(f"Non-fatal stream chunk parse error: {e}")
                    continue

            # 2. 安全提取工具返回的来源
            elif kind == "on_tool_end":
                output = event.get("data", {}).get("output")
                if output and isinstance(output, str):
                    try:
                        parsed = json.loads(output)
                        if isinstance(parsed, dict) and "sources" in parsed:
                            source_files.extend(parsed["sources"])
                    except (json.JSONDecodeError, TypeError):
                        pass

    except Exception as e:
        logger.error("Stream_Agent_Failed", error=str(e), session_id=session_id, exc_info=True)
        yield f"data: {json.dumps({'event': 'error', 'content': str(e)})}\n\n"
        return

    # 发送最终回答（如果为空）
    if not full_answer:
        full_answer = "抱歉，我暂时无法回答这个问题，请换个方式提问。"
        yield f"data: {json.dumps({'event': 'message', 'content': full_answer})}\n\n"

    # 发送来源信息
    if source_files:
        unique_sources = ", ".join(list(set(source_files)))
        yield f"data: {json.dumps({'event': 'source', 'content': unique_sources})}\n\n"

    yield f"data: {json.dumps({'event': 'done'})}\n\n"

    # ✅ 【修复】保存对话历史到内存
    try:
        chat_history_obj.add_messages([
            HumanMessage(content=query),
            AIMessage(content=full_answer)
        ])
    except Exception as mem_err:
        logger.warning(f"Stream Memory save failed for {session_id}: {mem_err}")

    # ✅ 【修复】异步持久化到 Redis（不阻塞 SSE 流）
    _save_session_async(session_id, chat_history_obj)