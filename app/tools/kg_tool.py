import re
from typing import List
from langchain.tools import tool
from app.tools.neo4j_tool import neo4j_client

# === 真实关系类型（来自截图）===
RELATION_TYPE_MAP = {
    "症状": ["DISEASE_SYMPTOM"],
    "治疗药物": ["DISEASE_DRUG"],
    "并发症": ["DISEASE_ACCOMPANY"],  # 注意：实际拼写需确认（见下文）
}


# === Cypher 注入防护白名单 ===
# 仅允许以下关系类型参与 Cypher 查询构建，防止注入攻击
VALID_RELATION_TYPES = {
    "DISEASE_SYMPTOM", "DISEASE_DRUG", "DISEASE_ACCOMPANY"
}
# === 症状关键词：覆盖西医+中医表述 ===
SYMPTOM_KEYWORDS = [
    # 引导词
    "症状", "临床表现", "可见", "多见", "常见", "表现为", "主症", "典型表现",
    "主要症状", "伴随症状", "可有", "有时", "常伴", "症见", "症候", "表现",
    # 中医直述词（关键！）
    "面色苍白", "心悸", "头晕", "舌淡", "脉细弱", "气短", "乏力", "纳差",
    "胸闷", "耳鸣", "失眠", "神疲", "自汗", "盗汗"
]


def _extract_symptoms_from_text(text: str) -> List[str]:
    """强化版：支持无引导词的中医症状直述"""
    if not text:
        return []

    # 预处理：统一换行/空格，去除多余标点
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = re.split(r'[。；！？\n\r]+', text)

    symptoms = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 2:
            continue

        # ✅ 方案1：匹配任何关键词（含中医直述词）
        if any(kw in sent for kw in SYMPTOM_KEYWORDS):
            cleaned = re.sub(r'^[\d、.\s]+|^\s*例如[:：]\s*', '', sent)
            cleaned = re.sub(r'[，,；;：:！!？?。.]$', '', cleaned).strip()
            if cleaned and len(cleaned) >= 2:
                symptoms.append(cleaned)

        # ✅ 方案2：无引导词但含典型症状短语（兜底）
        elif any(term in sent for term in [
            "面色苍白", "心悸", "头晕", "舌淡", "脉细弱", "气短", "乏力",
            "胸闷", "耳鸣", "失眠", "神疲", "自汗", "盗汗"
        ]):
            cleaned = re.sub(r'[，,；;：:！!？?。.]$', '', sent).strip()
            if cleaned and len(cleaned) >= 2:
                symptoms.append(cleaned)

    # 去重保序
    seen = set()
    unique = []
    for s in symptoms:
        key = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', '', s).lower()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


@tool
def knowledge_graph_search(entity: str, relation: str = "") -> str:
    """
    检索医疗知识图谱实体关系。
    严格使用真实关系类型：DISEASE_SYMPTOM, DISEASE_DRUG 等。
    支持结构化关系查询，并自动从疾病属性中提取未结构化的症状描述。

    Args:
        entity: 疾病/药品/症状实体名称，如 "乳泣"、"曲霉病"
        relation: 关系类型，如 "症状"、"治疗药物"（可选）
    """
    if not entity or not entity.strip():
        return "[KG] 错误：实体名称不能为空"

    entity = entity.strip()
    related_targets: List[str] = []

    # === Step 1: 结构化查询（真实关系）===
    relation_types = RELATION_TYPE_MAP.get(relation, [])
    if not relation:
        relation_types = ["DISEASE_SYMPTOM"]

    for r_type in relation_types:
        # 白名单校验：防止 Cypher 注入攻击
        if r_type not in VALID_RELATION_TYPES:
            continue

        try:
            target_label = ":Symptom" if r_type == "DISEASE_SYMPTOM" else ""
            if r_type == "DISEASE_DRUG":
                target_label = ":Drug"
            # 使用 format() 方法构建 Cypher（r_type 已通过白名单校验）
            cypher_template = """MATCH (n:Disease {name: $entity})-[r:`{r_type}`]->(m{target_label})
RETURN m.name AS target
LIMIT 15"""
            cypher = cypher_template.format(r_type=r_type, target_label=target_label)
            results = neo4j_client.run_query(cypher, {"entity": entity})
            if results:
                for rec in results:
                    target = (rec.get("target") or "").strip()
                    if target and target not in related_targets:
                        related_targets.append(target)
        except Exception:
            continue

    # === Step 2: Fallback —— 强化版（关键！）===
    is_symptom_query = relation in ["症状", "常见症状", "临床表现"] or not relation
    if is_symptom_query and not related_targets:
        try:
            # 尝试带标签查询（标准路径）
            cypher_tagged = """
            MATCH (n:Disease {name: $entity})
            RETURN n.cause AS cause, n.Cause AS Cause, n.CAUSE AS CAUSE,
                   n.desc AS desc, n.Desc AS Desc, n.DESC AS DESC,
                   n.description AS description, n.Description AS Description,
                   n.prevent AS prevent, n.Prevent AS Prevent, n.PREVENT AS PREVENT
            """
            res = neo4j_client.run_query(cypher_tagged, {"entity": entity})

            if not res:
                # 宽松 fallback：无标签查询（应对极少数异常）
                cypher_fallback = """
                MATCH (n {name: $entity})
                RETURN n.cause AS cause, n.desc AS desc, n.description AS description, n.prevent AS prevent
                """
                res = neo4j_client.run_query(cypher_fallback, {"entity": entity})

            if res:
                props = res[0]
                # 合并所有可能属性
                all_text = " ".join(
                    str(props.get(k, "") or "").strip()
                    for k in [
                        "cause", "Cause", "CAUSE",
                        "desc", "Desc", "DESC",
                        "description", "Description",
                        "prevent", "Prevent", "PREVENT"
                    ]
                )
                extracted = _extract_symptoms_from_text(all_text)
                if extracted:
                    related_targets.extend(extracted)
        except Exception as e:
            print(f"[KG Fallback Error] {e}")  # 仅调试用，正式环境可注释

    # === Step 3: 格式化 ===
    if related_targets:
        if is_symptom_query:
            symptom_list = "\n• ".join(related_targets)
            return f"[KG] {entity} 的常见症状包括：\n• {symptom_list}"
        else:
            items = "\n• ".join(related_targets)
            rel_name = relation or "相关信息"
            return f"[KG] {entity} 的 {rel_name}：\n• {items}"

    return f"[KG] 知识图谱中未找到与「{entity}」相关的{relation or '明确'}信息。"