# test_chat_service_redis.py
import asyncio
import redis
import json
from app.services.chat_service import process_chat, session_manager
from app.config.settings import settings


async def test():
    print("=" * 60)
    print("🧪 测试 chat_service → Redis 持久化")
    print("=" * 60)

    session_id = "chat-test-001"
    query = "感冒了怎么办"

    print(f"\n📨 用户提问: '{query}'")
    print(f"🆔 会话ID: {session_id}")

    # 调用 chat_service
    result = await process_chat(session_id=session_id, query=query)

    print(f"\n🤖 AI回复: {result['answer'][:100]}...")

    # ✅ 等待异步保存完成
    print("\n⏳ 等待异步保存到 Redis...")
    await asyncio.sleep(1)

    # 检查 Redis
    print("🔍 检查 Redis 是否写入...")
    r = redis.from_url(settings.REDIS_URL, decode_responses=True)
    key = f"session:{session_id}"
    data = r.get(key)

    if data:
        msgs = json.loads(data)
        print(f"   ✅ Redis 写入成功！共 {len(msgs)} 条消息")
        for m in msgs:
            print(f"      [{m['type']}] {m['content'][:50]}...")
    else:
        print(f"   ❌ Redis 中没有数据！")

    # 清理
    r.delete(key)
    session_manager.clear_session(session_id)
    print("\n🧹 测试数据已清理")


if __name__ == "__main__":
    asyncio.run(test())