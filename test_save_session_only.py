# test_save_session_only.py
import asyncio
import redis
import json
from app.services.chat_service import _save_session_async, session_manager
from app.config.settings import settings
from langchain_core.messages import HumanMessage, AIMessage


async def test():
    print("=" * 60)
    print("🧪 直接测试 _save_session_async")
    print("=" * 60)

    session_id = "save-test-001"

    # 1. 创建会话并添加消息
    print(f"\n📝 创建会话: {session_id}")
    session = session_manager.get_session(session_id)
    session.add_messages([
        HumanMessage(content="测试问题"),
        AIMessage(content="测试回答")
    ])
    print(f"   ✅ 内存消息数: {len(session.messages)}")

    # 2. 调用 _save_session_async
    print("\n💾 调用 _save_session_async...")
    _save_session_async(session_id, session)

    # 3. 等待异步任务完成
    print("⏳ 等待 1 秒...")
    await asyncio.sleep(1)

    # 4. 检查 Redis
    print("\n🔍 检查 Redis...")
    r = redis.from_url(settings.REDIS_URL, decode_responses=True)
    key = f"session:{session_id}"
    data = r.get(key)

    if data:
        msgs = json.loads(data)
        print(f"   ✅ Redis 写入成功！共 {len(msgs)} 条消息")
        for m in msgs:
            print(f"      [{m['type']}] {m['content']}")
    else:
        print(f"   ❌ Redis 中没有数据！")

    # 5. 模拟重启恢复
    print("\n🔄 模拟重启（清空内存）...")
    session_manager._store.clear()

    print("📥 从 Redis 恢复...")
    restored = session_manager.get_session(session_id)
    print(f"   恢复后消息数: {len(restored.messages)}")

    # 清理
    r.delete(key)
    session_manager.clear_session(session_id)
    print("\n🧹 已清理")


if __name__ == "__main__":
    asyncio.run(test())