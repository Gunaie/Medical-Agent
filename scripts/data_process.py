# scripts/data_process_v2.py
import os
import sys
import hashlib
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader

from app.tools.llm_tool import get_embeddings
from app.config.settings import settings
import chromadb

DATA_DIR = os.path.join(BASE_DIR, "data", "input")


def load_documents():
    """遍历目录加载多格式文档"""
    docs = []
    if not os.path.exists(DATA_DIR):
        print(f"⚠️ 数据目录不存在: {DATA_DIR}")
        return docs

    print(f"📂 扫描数据目录: {DATA_DIR}")
    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        ext = os.path.splitext(fname)[1].lower()

        loader = None
        if ext == ".txt":
            loader = TextLoader(fpath, encoding="utf-8")
        elif ext == ".csv":
            loader = CSVLoader(fpath, encoding="utf-8")
        elif ext == ".pdf":
            loader = PyPDFLoader(fpath)
        else:
            continue

        try:
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source_file"] = fname
            docs.extend(loaded_docs)
            print(f"   ✅ 成功加载: {fname} ({len(loaded_docs)} 个原始片段)")
        except Exception as e:
            print(f"   ❌ 加载失败 {fname}: {e}")

    return docs


def build_vectorstore(collection_name: str, collection_metadata: dict = None):
    print(f"📄 加载文档到 [{collection_name}]...")
    docs = load_documents()
    if not docs:
        print("未找到任何有效文档，退出。")
        return

    print("✂️  分割文本 (医疗优化参数)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", ";", "；", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"   共生成 {len(chunks)} 个 Chunk")

    print(f"🔢 向量化入库 (text-embedding-v4) -> {collection_name}...")
    embeddings = get_embeddings()

    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata=collection_metadata
    )

    batch_size = 20
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding & Upserting"):
        batch = chunks[i: i + batch_size]

        texts = [doc.page_content for doc in batch]
        ids = [hashlib.md5(doc.page_content.encode()).hexdigest() for doc in batch]

        # 为每个 chunk 注入集合级描述元数据
        enriched_metadatas = []
        for doc in batch:
            meta = doc.metadata.copy()
            if collection_metadata and "description" in collection_metadata:
                meta["collection_desc"] = collection_metadata["description"]
            enriched_metadatas.append(meta)

        vectors = embeddings.embed_documents(texts)

        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=enriched_metadatas,
            embeddings=vectors
        )

    print(f"\n🎉 完成！持久化至 {settings.CHROMA_PERSIST_DIR}")
    print(f"📊 [{collection_name}] 总片段数: {collection.count()}")


if __name__ == "__main__":
    build_vectorstore(
        collection_name="platform_customer_service",
        collection_metadata={"description": "服务流程、联系方式、投诉建议、账号问题"}
    )