# test_agent_debug.py
import asyncio
from app.core.agent_builder import build_medical_agent
from langchain_core.messages import HumanMessage


async def test():
    agent = build_medical_agent()

    try:
        result = agent.invoke({
            "messages": [HumanMessage(content="感冒了怎么办")]
        })
        print("✅ Agent 调用成功")
        print(f"返回类型: {type(result)}")
        print(f"keys: {result.keys() if hasattr(result, 'keys') else 'N/A'}")
    except Exception as e:
        print(f"❌ Agent 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test())