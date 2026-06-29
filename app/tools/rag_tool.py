# app/tools/rag_tool.py
from app.db.chroma_init import medical_collection, support_collection
from app.tools.llm_tool import get_embeddings  # ✅ P1-4 修复：统一使用 llm_tool 的入口
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import chromadb
from app.config.settings import settings


class DualDomainRAG:
    def __init__(self):
        self.embedding_fn = get_embeddings()
        self.medical_coll = medical_collection
        self.support_coll = support_collection
        self._bm25_retriever = None  # ✅ P0-3 新增：延迟初始化 BM25

    def _get_bm25_retriever(self):
        """✅ P0-3 新增：从 ChromaDB 构建 BM25 倒排索引"""
        if self._bm25_retriever is not None:
            return self._bm25_retriever

        raw_data = self.medical_coll.get(include=["documents", "metadatas"])
        all_docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(raw_data["documents"], raw_data["metadatas"])
        ]

        if not all_docs:
            return None

        self._bm25_retriever = BM25Retriever.from_documents(all_docs, k=8)
        return self._bm25_retriever

    def _rrf_fusion(self, docs_list, top_k=6, rrf_k=60):
        """✅ P0-3 新增：RRF 融合算法（从 rag_service.py 迁移）"""
        fused_scores = {}
        doc_map = {}

        for docs in docs_list:
            for rank, doc in enumerate(docs):
                content_hash = hash(doc.page_content)
                if content_hash not in fused_scores:
                    fused_scores[content_hash] = 0.0
                    doc_map[content_hash] = doc
                fused_scores[content_hash] += 1.0 / (rrf_k + rank + 1)

        sorted_hashes = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        return [doc_map[h] for h in sorted_hashes[:top_k]]

    def _safe_query(self, collection, query: str, k: int, threshold: float = 0.6, where_filter: dict = None) -> list:
        """支持 Metadata 过滤的安全向量查询"""
        if not query.strip():
            return []
        try:
            query_embedding = self.embedding_fn.embed_query(query)
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": k,
                "include": ["documents", "metadatas", "distances"]
            }
            if where_filter:
                query_params["where"] = where_filter

            results = collection.query(**query_params)
        except Exception as e:
            print(f"RAG Query Error: {e}")
            return []

        filtered = []
        if results and results["ids"] and results["ids"][0]:
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                if dist <= threshold:
                    filtered.append({
                        "content": doc,
                        "source": meta.get("source_file", "unknown"),
                        "page": meta.get("page", ""),
                        "distance": round(dist, 4)
                    })

        filtered.sort(key=lambda x: x["distance"])
        return filtered

    def retrieve_clinical(self, query: str, year: int = None, population: str = None) -> dict:
        """✅ P0-3 修复：支持混合检索（BM25 + Vector + RRF）的临床检索"""
        # 步骤 1：构建 Metadata 过滤条件（保留原有功能）
        conditions = [{"domain": "medical_clinical_docs"}]
        if year:
            conditions.append({"publish_year": year})
        if population:
            conditions.append({"target_population": population})
        where_filter = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        # 步骤 2：向量检索（k=12，扩大候选池）
        vector_docs = self._safe_query(
            collection=self.medical_coll,
            query=query,
            k=12,  # ✅ 增大候选池，与 rag_service.py 保持一致
            threshold=0.6,
            where_filter=where_filter
        )

        # 步骤 3：尝试 BM25 混合检索（✅ P0-3 核心新增）
        bm25 = self._get_bm25_retriever()
        if bm25 is not None and not year and not population:
            # 仅当无年份/人群过滤时启用 BM25（BM25 不支持 metadata 过滤）
            try:
                bm25_docs = bm25.invoke(query)
                # 将 BM25 结果转换为与向量检索相同的格式
                bm25_formatted = [
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source_file", "unknown"),
                        "page": doc.metadata.get("page", ""),
                        "distance": 0.5  # BM25 无距离概念，赋中等值
                    }
                    for doc in bm25_docs[:8]
                ]

                # RRF 融合
                fused = self._rrf_fusion(
                    [bm25_formatted, vector_docs],
                    top_k=6
                )

                if fused:
                    return {
                        "content": "\n\n---\n\n".join([d["content"] for d in fused]),
                        "sources": list(set([d["source"] for d in fused]))
                    }
            except Exception as e:
                print(f"混合检索失败，降级到纯向量检索: {e}")

        # 步骤 4：Fallback 到纯向量检索（保持原有行为）
        if not vector_docs:
            return {
                "content": "未检索到相关权威临床指南。建议检查查询关键词或放宽年份/人群限制。",
                "sources": []
            }

        return {
            "content": "\n\n---\n\n".join([d["content"] for d in vector_docs[:6]]),
            "sources": list(set([d["source"] for d in vector_docs]))
        }

    def retrieve_support(self, query: str) -> dict:
        """支持返回真实来源的平台服务检索"""
        docs = self._safe_query(
            collection=self.support_coll,
            query=query,
            k=3,
            threshold=0.6
        )

        if not docs:
            return {
                "content": "抱歉，暂未找到相关的平台服务或流程说明。请联系人工客服。",
                "sources": []
            }

        return {
            "content": "\n".join([d["content"] for d in docs]),
            "sources": list(set([d["source"] for d in docs]))
        }


rag_engine = DualDomainRAG()