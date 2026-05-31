from langchain_core.messages import HumanMessage, AIMessage
from agent import build_agent
from prompt import REWRITE_PROMPT
from utils import get_llm
from langchain_core.prompts import ChatPromptTemplate


def _normalize_message(m):
    """将 dict 或 Message 对象统一转为 LangChain Message 对象"""
    if isinstance(m, dict):
        role = m.get("role", "user")
        content = m.get("content", "")
        return HumanMessage(content=content) if role == "user" else AIMessage(content=content)
    return m  # 已经是 Message 对象则直接返回


def rewrite_query(input_text: str, chat_history: list) -> str:
    """结合对话历史改写用户输入为独立完整的问题"""
    if not chat_history:
        return input_text

    llm = get_llm()

    # 兼容 dict 和 Message 两种格式
    history_str = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in map(_normalize_message, chat_history)
    )

    # 【核心修复】使用专用的 REWRITE_PROMPT，变量名与模板严格对应
    chain = REWRITE_PROMPT | llm
    result = chain.invoke({
        "chat_history": history_str,
        "input": input_text
    })

    return result.content.strip()


def chat(input_text: str, chat_history: list) -> str:
    """Gradio ChatInterface 回调函数"""
    # Step 1: 查询改写
    refined_query = rewrite_query(input_text, chat_history)

    # Step 2: 构建标准消息列表（兼容 dict 格式）
    messages = [_normalize_message(m) for m in chat_history]
    messages.append(HumanMessage(content=refined_query))

    # Step 3: 调用 Agent
    agent = build_agent()
    response = agent.invoke({"messages": messages})

    # 提取最后一条 AI 回复
    ai_messages = [m for m in response["messages"] if isinstance(m, AIMessage)]
    return ai_messages[-1].content if ai_messages else "抱歉，未能生成回复。"