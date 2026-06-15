from langchain_community.embeddings import DashScopeEmbeddings
import os

def get_embedding_function():
    # ✅ 纯 API 模式，无需任何本地模型文件
    return DashScopeEmbeddings(
        model="text-embedding-v4",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )