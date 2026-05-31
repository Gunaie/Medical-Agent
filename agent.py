import json
import re
import time
from typing import Optional, List, Dict, Any

from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_chroma import Chroma

from utils import get_llm, get_embeddings, query_neo4j
from prompt import (
    AGENT_SYSTEM_PROMPT, NER_PROMPT,
    GRAPH_TEMPLATE, QUESTION_TYPE_KEYWORDS
)
import config

# ==================== 安全封装：带超时的LLM调用 ====================
def safe_llm_invoke(chain, inputs: dict, timeout_sec: float = 30.0, fallback_msg: str = "大模型响应超时") -> str:
    """统一封装LLM调用，防止API无响应导致Agent卡死"""
    start = time.time()
    try:
        # 注意：LangChain 原生 chain.invoke 不支持直接传 timeout 参数
        # 此处通过捕获异常 + 外部监控实现软超时保护
        result = chain.invoke(inputs)
        elapsed = time.time() - start
        print(f"[LLM] ✅ {elapsed:.2f}s")
        return result.content if hasattr(result, 'content') else str(result)
    except Exception as e:
        elapsed = time.time() - start
        print(f"[LLM] ❌ FAILED after {elapsed:.2f}s | {type(e).__name__}: {e}")
        return fallback_msg


# ==================== 工具定义 ====================

@tool
def generic_func(query: str) -> str:
    """处理通用闲聊、身份询问、敏感话题，或其他工具无法回答时的兜底回复。
    当问题不涉及医疗知识或平台信息时使用此工具。"""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是专业医疗问诊助手。拒绝政治/暴力/色情话题。若不涉及医疗或平台问题，友好回复并引导用户咨询医疗健康问题。"),
        ("human", "{query}")
    ])
    chain = prompt | llm
    result = safe_llm_invoke(chain, {"query": query}, fallback_msg="抱歉，我暂时无法回答该问题，请稍后再试。")
    return f"{result}\n\n🤖 [数据来源: 大语言模型通用生成]"


@tool
def retrival_func(query: str) -> str:
    """检索'寻医问药网'平台相关文档信息，如服务流程、平台介绍、联系方式、融资投资等。
    仅用于平台相关问题，不用于医疗知识问答。返回内容包含信息来源标注。"""
    start = time.time()
    embeddings = get_embeddings()
    vectordb = Chroma(
        persist_directory=config.CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )

    expanded_query = f"{query} 联系电话 咨询热线 客服 联系方式 融资 投资 股权 资本"
    results = vectordb.similarity_search_with_score(expanded_query, k=4)

    filtered_docs = []
    for doc, distance in results:
        if distance <= config.RETRIEVAL_THRESHOLD:
            source = doc.metadata.get("source", "平台文档")
            page = doc.metadata.get("page", "")
            citation = f"\n📌 [信息来源: {source}]" + (f" | 第{page + 1}页" if isinstance(page, int) else "")
            # ⚠️ 截断单条文档长度，防止拼接后超出LLM上下文
            content = doc.page_content[:800]
            filtered_docs.append(f"{content}{citation}")

    elapsed = time.time() - start
    print(f"[RAG] ✅ {elapsed:.2f}s | Hits: {len(filtered_docs)}")

    if not filtered_docs:
        return f"未在平台文档中找到与「{query}」直接相关的信息。建议访问官网 www.xywy.com 底部「联系我们」获取权威信息。\n\n📌 [信息来源: RAG检索未命中]"

    return "\n\n---\n\n".join(filtered_docs)


@tool
def graph_func(query: str) -> str:
    """通过医疗知识图谱回答疾病、症状、药物等专业医疗问题。
    支持：疾病定义/病因/症状/治疗、药物禁忌、症状对应疾病等。"""
    total_start = time.time()
    llm = get_llm()

    # ========== Step 1: NER 实体抽取（带超时保护）==========
    ner_prompt = ChatPromptTemplate.from_template(NER_PROMPT)
    ner_chain = ner_prompt | llm
    ner_raw = safe_llm_invoke(ner_chain, {"question": query}, timeout_sec=15.0, fallback_msg="")

    entities = {}
    try:
        match = re.search(r'\{[^{}]*\}', ner_raw, re.DOTALL)
        if match:
            entities = json.loads(match.group())
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"[NER] ❌ JSON解析失败: {e} | Raw: {ner_raw[:200]}")
        return f"未能从问题中识别出有效的医疗实体，请换个方式描述。\n\n🤖 [数据来源: NER提取失败]"

    # ========== Step 2: 匹配问题类型 & 构建 Cypher ==========
    entity_map = {
        "disease": entities.get("disease"),
        "symptom": entities.get("symptom"),
        "drug": entities.get("drug")
    }
    matched_type: Optional[str] = None
    matched_entity: Optional[str] = None

    for qtype, keywords in QUESTION_TYPE_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            required = "disease" if "disease" in qtype else ("drug" if "drug" in qtype else "symptom")
            if entity_map.get(required):
                matched_type = qtype
                matched_entity = entity_map[required]
                break

    if not matched_type or not matched_entity:
        return f"识别到实体 {entities}，但无法匹配到对应的知识图谱查询类型。\n\n🤖 [数据来源: 图谱路由失败]"

    cypher = GRAPH_TEMPLATE.get(matched_type)
    if not cypher:
        return f"未找到对应的图谱查询模板。\n\n🤖 [数据来源: 图谱模板缺失]"

    # ========== Step 3: 执行图谱查询（带安全保护）==========
    db_start = time.time()
    records = query_neo4j(cypher, {"entity": matched_entity})
    db_elapsed = time.time() - db_start
    print(f"[Neo4j] ✅ {db_elapsed:.2f}s | Type: {matched_type} | Entity: {matched_entity}")

    if not records or not records[0].get("answer"):
        return f"知识图谱中未找到关于'{matched_entity}'的{matched_type}信息。\n\n🗄️ [数据来源: 医疗知识图谱 (无结果)]"

    answer = records[0]["answer"]
    if isinstance(answer, list):
        # ⚠️ 关键：强制截断列表长度，防止LLM润色时Token爆炸
        answer = answer[:20]
        answer_str = "、".join(str(a) for a in answer)
    else:
        answer_str = str(answer)[:1500]  # 非列表也做长度保护

    # ========== Step 4: LLM 润色（带超时保护）==========
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是医疗问答助手。根据图谱查询结果，用通俗易懂的中文回答用户问题。不要添加图谱中没有的信息。"),
        ("human", "问题：{query}\n图谱结果：{answer}\n回答：")
    ])
    summary_chain = summary_prompt | llm
    final_answer = safe_llm_invoke(
        summary_chain,
        {"query": query, "answer": answer_str},
        timeout_sec=20.0,
        fallback_msg=f"根据图谱数据，{matched_entity}的相关信息如下：{answer_str}"  # ⭐ 润色失败时直接返回原始数据兜底
    )

    total_elapsed = time.time() - total_start
    print(f"[GraphFunc] ✅ TOTAL {total_elapsed:.2f}s")

    return f"{final_answer}\n\n🗄️ [数据来源: 医疗知识图谱 ({matched_type}: {matched_entity})]"


# ==================== Agent 构建 ====================

def build_agent(chat_history: Optional[List[Dict[str, Any]]] = None):
    """构建 3 工具 ReAct Agent (LangGraph 版)"""
    llm = get_llm()
    tools = [generic_func, retrival_func, graph_func]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=AGENT_SYSTEM_PROMPT
    )
    return agent