# app/tools/rag_tool.py
from app.db.chroma_init import medical_collection, support_collection
from app.tools.embeddings import get_embedding_function

class DualDomainRAG:
    def __init__(self):
        self.embedding_fn = get_embedding_function()
        self.medical_coll = medical_collection
        self.support_coll = support_collection

    def _safe_query(self, collection, query: str, k: int, threshold: float = 0.6, where_filter: dict = None) -> list:
        """支持Metadata过滤的安全查询"""
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
            print(f"❌ RAG Query Error: {e}")
            return []

        filtered = []
        if results and results["ids"] and results["ids"][0]:
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                # ✅ 此处逻辑正确，但依赖调用方传入正确的 threshold
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
        """支持按年份/人群过滤的临床检索"""
        conditions = [{"domain": "medical_clinical_docs"}]
        if year:
            conditions.append({"publish_year": year})
        if population:
            conditions.append({"target_population": population})

        where_filter = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        docs = self._safe_query(
            collection=self.medical_coll,
            query=query,
            k=5,
            threshold=0.6,  # ✅ 【核心修复】从 1.8 改为 0.6
            where_filter=where_filter
        )

        if not docs:
            return {
                "content": "未检索到相关权威临床指南。建议检查查询关键词或放宽年份/人群限制。",
                "sources": []
            }

        return {
            "content": "\n\n---\n\n".join([d["content"] for d in docs]),
            "sources": list(set([d["source"] for d in docs]))
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
            # ✅ 提取真实的 source_file 元数据并去重
            "sources": list(set([d["source"] for d in docs]))
        }

rag_engine = DualDomainRAG()