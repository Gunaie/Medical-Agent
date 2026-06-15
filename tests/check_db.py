# tests/check_db.py
import chromadb
from app.config.settings import settings

client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
col = client.get_collection("medical_docs")

# ✅ peek() 不接受 include 参数，直接调用即可（默认返回 documents + metadatas + ids）
sample = col.peek(limit=3)

print("=== ChromaDB Metadata 验证 ===")
print(f"集合总文档数: {col.count()}")
print(f"采样元数据: {sample['metadatas']}")

# 自动检查 source_file 是否存在
has_source = any(
    isinstance(md, dict) and ("source_file" in md or "source" in md or "file_name" in md)
    for md in (sample['metadatas'] or [])
)
print(f"\n✅ source 字段存在: {has_source}" if has_source else "\n❌ 未检测到 source 相关字段，请检查入库脚本是否写入了 metadata")