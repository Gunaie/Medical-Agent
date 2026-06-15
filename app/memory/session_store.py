# app/memory/session_store.py
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from typing import Dict


class SessionManager:
    """会话管理器：基于内存管理多用户对话历史"""

    def __init__(self):
        # 使用字典存储多个用户的会话历史，key为session_id
        self._store: Dict[str, BaseChatMessageHistory] = {}

    def get_session(self, session_id: str) -> BaseChatMessageHistory:
        """获取或创建指定 session_id 的会话历史对象"""
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
        return self._store[session_id]

    def clear_session(self, session_id: str):
        """清除指定会话的历史记录"""
        if session_id in self._store:
            del self._store[session_id]