import os
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv

load_dotenv()

# app/tools/llm.py (修改 get_llm)
def get_llm():
    """获取 Qwen-Plus 大模型实例"""
    return ChatTongyi(
        model=os.getenv("LLM_MODEL", "qwen-plus"),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.3,
        streaming=True   # ✅ 【核心修改】开启 Token 级流式
    )

def get_embeddings():
    """获取 text-embedding-v4 向量化模型实例"""
    return DashScopeEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )