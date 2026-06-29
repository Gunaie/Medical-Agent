# 在 Python 交互式环境中测试
import redis
from app.memory.session_store import SessionManager

# 1. 启动服务，发送一条消息
# 2. 检查 Redis 中是否有数据
r = redis.from_url("redis://localhost:6379/0", decode_responses=True)
keys = r.keys("session:*")
print(f"Redis 中的会话数: {len(keys)}")
for key in keys:
    print(f"{key}: {r.get(key)[:200]}...")  # 打印前 200 字符