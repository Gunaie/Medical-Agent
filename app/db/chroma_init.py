# app/db/chroma_init.py
import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from pathlib import Path

# ✅ 加载环境变量
load_dotenv()

# ✅ 【核心修复】只保留这一处路径定义，删除后续所有重复的 client 初始化
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

# ✅ 确保目录存在（可选，防止首次运行报错）
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

ef = OpenAIEmbeddingFunction(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name="text-embedding-v4"
)

# ✅ 使用上方唯一初始化的 client，不再重复创建
medical_collection = client.get_or_create_collection(
    name="medical_docs",
    metadata={"description": "权威指南、药典、共识、说明书"},
    embedding_function=ef
)

support_collection = client.get_or_create_collection(
    name="platform_customer_service",
    metadata={"description": "服务流程、联系方式、投诉建议、账号问题"},
    embedding_function=ef
)