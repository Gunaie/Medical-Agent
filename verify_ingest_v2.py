import chromadb

# ⚠️ 请根据你项目实际配置修改路径和 collection 名称
client = chromadb.PersistentClient(path="D:/Projects/Medical-Agent/chroma_db")

print("📂 当前所有 Collections:")
for col in client.list_collections():
    print(f"   - {col.name} | 文档数: {col.count()}")

# 尝试获取你的目标 collection（替换为实际名称）
try:
    coll = client.get_collection("medical_docs")
    print(f"\n✅ medical_docs 文档数: {coll.count()}")

    if coll.count() > 0:
        sample = coll.peek(limit=2)
        print(f"📄 样本 Metadata: {sample['metadatas']}")
        print(f"📄 样本 Content 前100字: {sample['documents'][0][:100]}...")
    else:
        print("❌ Collection 存在但为空！需重新运行 ingest 脚本")
except Exception as e:
    print(f"❌ 获取 Collection 失败: {e}")