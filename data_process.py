import os
# PyPDFLoader 依然在 langchain-community 中，但需要底层 pypdf 库支持
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
# ⚠️ 核心修复：文本分割器必须从独立新包导入
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from utils import get_embeddings
import config


def load_documents(data_dir="./data/inputs"):
    docs = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        ext = os.path.splitext(fname)[1].lower()

        if ext == ".txt":
            # ⚠️ 关键修复：直接指定 UTF-8 编码，彻底规避 chardet/charset-normalizer 兼容问题
            loader = TextLoader(fpath, encoding="utf-8")
        elif ext == ".csv":
            loader = CSVLoader(fpath, encoding="utf-8")
        elif ext == ".pdf":
            loader = PyPDFLoader(fpath)
        else:
            continue

        try:
            docs.extend(loader.load())
            print(f"   ✅ 成功加载: {fname}")
        except Exception as e:
            print(f"   ❌ 加载失败 {fname}: {e}")

    return docs

def build_vectorstore():
    print("📄 加载文档...")
    docs = load_documents()
    print(f"   共加载 {len(docs)} 个文档块")

    print("✂️  分割文本...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    print("🔢 向量化入库 (text-embedding-v4, 1024维)...")
    embeddings = get_embeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR
    )
    print(f"✅ 完成！持久化至 {config.CHROMA_PERSIST_DIR}")
    return vectordb

if __name__ == "__main__":
    build_vectorstore()