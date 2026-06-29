# app/memory/session_store.py
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, Optional
import json

# 尝试导入 Redis，未安装则降级到内存
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class SessionManager:
    """会话管理器：支持内存和 Redis 两种模式

    使用方式：
        # 内存模式（默认）
        session_manager = SessionManager()

        # Redis 模式
        session_manager = SessionManager(redis_url="redis://localhost:6379/0")
    """

    def __init__(self, redis_url: Optional[str] = None):
        # 内存存储（始终存在，作为一级缓存）
        self._store: Dict[str, BaseChatMessageHistory] = {}

        # Redis 持久化存储（可选）
        self._redis = None

        if REDIS_AVAILABLE and redis_url:
            try:
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                print("✅ Redis 会话存储已启用")
            except Exception as e:
                print(f"⚠️ Redis 连接失败，降级到内存存储: {e}")
                self._redis = None

    def get_session(self, session_id: str) -> BaseChatMessageHistory:
        """获取或创建指定 session_id 的会话历史对象

        优先从内存获取，内存未命中则尝试从 Redis 恢复。
        """
        # 1. 优先从内存获取（热数据）
        if session_id in self._store:
            return self._store[session_id]

        # 2. 尝试从 Redis 恢复（冷数据）
        if self._redis:
            try:
                data = self._redis.get(f"session:{session_id}")
                if data:
                    history = InMemoryChatMessageHistory()
                    # 反序列化消息
                    messages = json.loads(data)
                    for msg in messages:
                        if msg.get("type") == "human":
                            history.add_message(HumanMessage(content=msg.get("content", "")))
                        elif msg.get("type") == "ai":
                            history.add_message(AIMessage(content=msg.get("content", "")))
                    # 恢复到内存缓存
                    self._store[session_id] = history
                    return history
            except Exception as e:
                print(f"⚠️ Redis 会话恢复失败: {e}")

        # 3. 创建新会话
        self._store[session_id] = InMemoryChatMessageHistory()
        return self._store[session_id]

    def clear_session(self, session_id: str):
        """清除指定会话的历史记录（内存 + Redis）"""
        if session_id in self._store:
            del self._store[session_id]

        if self._redis:
            try:
                self._redis.delete(f"session:{session_id}")
            except Exception as e:
                print(f"⚠️ Redis 会话删除失败: {e}")

    def save_session(self, session_id: str):
        """将会话持久化到 Redis（建议在流式输出结束后调用）

        注意：此方法是异步友好的，不阻塞主流程。
        """
        if not self._redis or session_id not in self._store:
            return

        try:
            history = self._store[session_id]
            # 序列化消息为 JSON
            messages = []
            for msg in history.messages:
                msg_type = "human" if isinstance(msg, HumanMessage) else "ai"
                messages.append({
                    "type": msg_type,
                    "content": msg.content
                })

            # 写入 Redis，设置 24 小时过期时间
            self._redis.setex(
                f"session:{session_id}",
                86400,  # 24小时过期
                json.dumps(messages, ensure_ascii=False)
            )
        except Exception as e:
            print(f"⚠️ 会话保存到 Redis 失败: {e}")

    def get_stats(self) -> dict:
        """获取会话存储统计信息"""
        stats = {
            "memory_sessions": len(self._store),
            "redis_enabled": self._redis is not None
        }
        return stats