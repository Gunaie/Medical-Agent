# verify_ingest_v2.py
import chromadb

# ⚠️ 路径与原配置保持一致
CHROMA_PATH = "D:/Projects/Medical-Agent/chroma_db"
TARGET_COLLECTION = "platform_customer_service"

client = chromadb.PersistentClient(path=CHROMA_PATH)

print("📂 当前所有 Collections:")
all_collections = client.list_collections()
if not all_collections:
    print("   ❌ 数据库为空，请先运行 data_process_v2.py")
else:
    for col in all_collections:
        print(f"   - {col.name} | 文档数: {col.count()}")

print(f"\n{'='*50}")
print(f"🎯 目标验证: [{TARGET_COLLECTION}]")
print(f"{'='*50}")

try:
    coll = client.get_collection(TARGET_COLLECTION)
    count = coll.count()
    print(f"✅ 文档数: {count}")

    # 校验集合级元数据
    col_meta = coll.metadata
    print(f"🏷️  集合 Metadata: {col_meta}")

    if count > 0:
        sample = coll.peek(limit=2)

        # 校验 chunk 级业务描述是否注入成功
        has_desc = all(
            "collection_desc" in m for m in sample["metadatas"]
        )
        desc_status = "✅" if has_desc else "❌ 缺失 collection_desc"
        print(f"🔍 Chunk 业务描述注入: {desc_status}")

        print(f"📄 样本 Metadata: {sample['metadatas']}")
        print(f"📄 样本 Content 前100字: {sample['documents'][0][:100]}...")
    else:
        print("❌ Collection 存在但为空！需重新运行 data_process_v2.py")

except Exception as e:
    print(f"❌ 获取 Collection [{TARGET_COLLECTION}] 失败: {e}")
    print("💡 提示: 若刚删除过 chroma_db，请确认已重新运行入库脚本")