# app/services/rag_service.py
import logging
from typing import List, Dict, Tuple

import chromadb
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from app.tools.llm_tool import get_llm, get_embeddings
from app.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MEDICAL_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的企业级医疗智能助手。请严格根据以下检索到的参考资料回答用户问题。
如果参考资料中没有相关信息，请直接告知用户“当前知识库未收录该信息”，切勿编造。
回答时请保持严谨、客观，并在回答末尾以 [来源: 文件名] 的格式标注参考依据。

<参考资料>
{context}
</参考资料>"""),
    ("human", "{input}")
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个搜索查询优化专家。请将用户的口语化问题改写为更适合在医疗知识库中检索的书面化、关键词丰富的查询语句。只输出改写后的查询，不要任何解释。"),
    ("human", "{query}")
])


def _rrf_fusion(
        docs_list: List[List[Document]],
        top_k: int = 6,
        rrf_k: int = 60
) -> List[Document]:
    """
    ✅ 核心升级：使用 RRF 算法融合 BM25 和 Vector 结果，并自动截断低相关性内容
    Args:
        docs_list: [bm25_docs, vector_docs]
        top_k: 最终保留的文档数量（起到阈值过滤的作用）
        rrf_k: RRF 平滑常数，默认 60
    """
    fused_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for docs in docs_list:
        for rank, doc in enumerate(docs):
            content_hash = hash(doc.page_content)
            if content_hash not in fused_scores:
                fused_scores[content_hash] = 0.0
                doc_map[content_hash] = doc
            # RRF 公式: score += 1 / (k + rank)
            fused_scores[content_hash] += 1.0 / (rrf_k + rank + 1)

    # 按融合分数降序排列，并截取 top_k
    sorted_hashes = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    reranked_docs = [doc_map[h] for h in sorted_hashes[:top_k]]

    return reranked_docs


def _format_docs(docs: List[Document]) -> str:
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source_file", "Unknown")
        formatted.append(f"[文件: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def _debug_retrieval(docs: List[Document]) -> List[Document]:
    logger.info(f"🔍 RRF 融合后保留 {len(docs)} 个高相关片段:")
    for i, doc in enumerate(docs):
        preview = doc.page_content[:80].replace("\n", " ")
        logger.info(f"  [{i}] Source: {doc.metadata.get('source_file')} | Preview: {preview}...")
    return docs


def get_rag_chain():
    # 1. 连接 ChromaDB
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    vectorstore = Chroma(
        client=client,
        collection_name="medical_docs",
        embedding_function=get_embeddings()
    )

    # 从 ChromaDB 中提取原始文档构建 BM25 倒排索引
    collection = client.get_collection("medical_docs")
    raw_data = collection.get(include=["documents", "metadatas"])

    all_docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw_data["documents"], raw_data["metadatas"])
    ]

    if not all_docs:
        raise ValueError("知识库为空，请先运行 data_process.py 进行数据入库！")

    # ✅ 方案一落地：差异化召回配额
    # BM25 保持 k=8，维持对 other.csv 短文本精确问答的高敏感度
    bm25_retriever = BM25Retriever.from_documents(all_docs, k=8)

    # Vector 提升至 k=12，弥补长文本(PDF/TXT)在统计检索中的排名劣势
    # 更大的候选池能让 RRF 融合时有更多机会将高相关长文本顶到前排
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

    llm = get_llm()
    rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()

    # 混合检索与 RRF 融合链路保持不变
    hybrid_retriever = (
            RunnableParallel(
                bm25=bm25_retriever,
                vector=vector_retriever
            )
            | RunnableLambda(lambda x: _rrf_fusion([x["bm25"], x["vector"]], top_k=6))
    )

    rag_chain = (
            {
                "context": rewrite_chain | hybrid_retriever | RunnableLambda(_debug_retrieval) | _format_docs,
                "input": RunnablePassthrough()
            }
            | MEDICAL_RAG_PROMPT
            | llm
            | StrOutputParser()
    )

    return rag_chain, hybrid_retriever