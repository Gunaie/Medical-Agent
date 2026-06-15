# app/services/chat_service.py
import asyncio
import json
from functools import lru_cache

from langchain_core.messages import HumanMessage, AIMessage

from app.core.agent_builder import build_medical_agent
from app.memory.session_store import SessionManager
from app.utils.logger import logger

session_manager = SessionManager()


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


async def process_chat(session_id: str, query: str) -> dict:
    """原有非流式处理逻辑，完全保持不变"""
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

        try:
            chat_history_obj.add_messages([
                HumanMessage(content=query),
                AIMessage(content=answer)
            ])
        except Exception as mem_err:
            logger.warning(f"Memory save failed for {session_id}: {mem_err}")

        return {
            "answer": answer,
            "source": ", ".join(list(set(source_files))) if source_files else None
        }

    except Exception as e:
        logger.error("ReAct_Agent_Failed", error=str(e), session_id=session_id)
        return {"answer": f"系统处理异常，请稍后重试。(Error: {str(e)})", "source": None}


async def process_chat_stream(session_id: str, query: str):
    """SSE 流式处理生成器（已修复 IndexError 与流式中断问题）"""
    agent = _get_agent_template()
    agent_input, chat_history_obj = _prepare_agent_input(session_id, query)

    full_answer = ""
    source_files = []

    try:
        yield f"data: {json.dumps({'event': 'status', 'content': '正在分析您的问题...'})}\n\n"

        # ✅ 【核心修复】使用 astream_events 替代 astream，并增加严格的类型与边界检查
        async for event in agent.astream_events(agent_input, version="v2"):
            kind = event.get("event")

            # 1. 安全提取 AI 增量回复
            if kind == "on_chat_model_stream":
                try:
                    chunk = event.get("data", {}).get("chunk")

                    # ⚠️ 关键防御：多重空值与类型保护，兼容不同 LangChain 版本的 chunk 结构
                    if chunk is None:
                        continue

                    delta = ""
                    if hasattr(chunk, "content"):
                        delta = chunk.content or ""
                    elif isinstance(chunk, dict):
                        delta = chunk.get("content", "") or ""
                    elif isinstance(chunk, str):
                        delta = chunk

                    # 仅当有实际非空内容时才输出和累积，过滤空白字符
                    if delta and isinstance(delta, str) and delta.strip():
                        full_answer += delta
                        yield f"data: {json.dumps({'event': 'message', 'content': delta})}\n\n"

                except (IndexError, TypeError, AttributeError) as e:
                    # ✅ 异常容忍：跳过当前损坏的 chunk，继续处理后续事件，绝不中断 SSE 连接
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
        # 捕获流式循环外部的致命异常（如 Agent 初始化失败、数据库断连等）
        logger.error("Stream_Agent_Failed", error=str(e), session_id=session_id, exc_info=True)
        yield f"data: {json.dumps({'event': 'error', 'content': str(e)})}\n\n"
        return

    # 后续的正常结束逻辑保持不变...
    if not full_answer:
        full_answer = "抱歉，我暂时无法回答这个问题，请换个方式提问。"
        yield f"data: {json.dumps({'event': 'message', 'content': full_answer})}\n\n"

    if source_files:
        unique_sources = ", ".join(list(set(source_files)))
        yield f"data: {json.dumps({'event': 'source', 'content': unique_sources})}\n\n"

    yield f"data: {json.dumps({'event': 'done'})}\n\n"

    # 异步保存 Memory（不阻塞流式输出）
    try:
        chat_history_obj.add_messages([
            HumanMessage(content=query),
            AIMessage(content=full_answer)
        ])
    except Exception as mem_err:
        logger.warning(f"Stream Memory save failed for {session_id}: {mem_err}")