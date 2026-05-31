# ========== Agent System Prompt (LangGraph create_react_agent 专用) ==========
from langchain_core.prompts import ChatPromptTemplate

AGENT_SYSTEM_PROMPT = """你是一个专业的医疗问诊助手。你可以使用以下工具回答问题：
1. generic_func: 处理闲聊、身份询问、敏感话题拦截，或当其他工具无法回答时的兜底回复
2. retrival_func: 回答关于"寻医问药网"平台信息、服务流程等文档类问题
3. graph_func: 回答疾病定义、病因、症状、治疗、用药、检查、科室、预防等专业医疗知识问题

规则：
- 只要问题涉及疾病、症状、药物、治疗、检查、科室等医疗实体，必须优先调用 graph_func
- 如果用户问题不明确，先用 generic_func 澄清
- 绝不编造医疗信息，若工具均无法回答，诚实告知"暂无相关信息"
- 始终用中文回复

{messages}"""

# ========== 对话摘要 Prompt ==========
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是医疗问答助手。根据图谱查询结果，用通俗易懂的中文回答用户问题。\n"
     "【重要规则】\n"
     "1. 不要添加图谱中没有的信息；\n"
     "2. 必须在回答的最末尾，原样保留以下来源标记，不得修改、删除或改写：\n"
     "   🗄️ [数据来源: 医疗知识图谱 ({matched_type}: {matched_entity})]"),
    ("human", "问题：{query}\n图谱结果：{answer}\n回答：")
])

# ========== 知识图谱 NER Prompt ==========
NER_PROMPT = """从用户问题中提取医疗实体，严格按JSON格式返回，不要包含任何其他文字、解释或Markdown标记：
{{"disease": "疾病名或null", "symptom": "症状名或null", "drug": "药品名或null"}}

示例：
输入：曲霉病可以用哪些药物治疗？
输出：{{"disease": "曲霉病", "symptom": null, "drug": null}}

用户问题：{question}
JSON输出："""

# ========== 知识图谱问答模板（严格对齐Neo4j真实Schema） ==========
GRAPH_TEMPLATE = {
    "disease_define": "MATCH (d:Disease {name: $entity}) RETURN d.desc AS answer",
    "disease_cause": "MATCH (d:Disease {name: $entity}) RETURN d.cause AS answer",
    "disease_prevent": "MATCH (d:Disease {name: $entity}) RETURN d.prevent AS answer",
    "disease_symptom": "MATCH (d:Disease {name: $entity})-[:DISEASE_SYMPTOM]->(s:Symptom) RETURN collect(s.name) AS answer",
    "disease_treatment": "MATCH (d:Disease {name: $entity})-[:DISEASE_DRUG]->(dr:Drug) RETURN collect(dr.name) AS answer",
    "disease_cureway": "MATCH (d:Disease {name: $entity})-[:DISEASE_CUREWAY]->(c:Cureway) RETURN collect(c.name) AS answer",
    "disease_check": "MATCH (d:Disease {name: $entity})-[:DISEASE_CHECK]->(c:Check) RETURN collect(c.name) AS answer",
    "disease_department": "MATCH (d:Disease {name: $entity})-[:DISEASE_DEPARTMENT]->(de:Department) RETURN collect(de.name) AS answer",
    "disease_do_eat": "MATCH (d:Disease {name: $entity})-[:DISEASE_DO_EAT]->(f:Food) RETURN collect(f.name) AS answer",
    "disease_not_eat": "MATCH (d:Disease {name: $entity})-[:DISEASE_NOT_EAT]->(f:Food) RETURN collect(f.name) AS answer",
    "symptom_disease": "MATCH (s:Symptom {name: $entity})<-[:DISEASE_SYMPTOM]-(d:Disease) RETURN collect(d.name) AS answer",
}

# ========== 问题类型关键词映射（全面扩充覆盖） ==========
QUESTION_TYPE_KEYWORDS = {
    "disease_define": ["是什么", "定义", "什么叫", "是指", "含义", "介绍", "简介"],
    "disease_cause": ["原因", "病因", "为什么得", "引起", "导致", "诱发", "发病机制"],
    "disease_prevent": ["怎么预防", "预防措施", "如何避免", "防范", "日常注意"],
    "disease_symptom": ["症状", "表现", "有什么反应", "临床表现", "特征", "体征", "感觉", "不舒服"],
    "disease_treatment": [
        "怎么治", "用什么药", "治疗方案", "药物治疗", "治疗", "用药",
        "吃什么药", "特效药", "首选药", "抗生素", "抗真菌", "疗法", "吃药"
    ],
    "disease_cureway": ["治疗方式", "手术", "保守治疗", "理疗", "化疗", "靶向", "中医治疗"],
    "disease_check": ["做什么检查", "检查项目", "怎么确诊", "化验", "拍片", "CT", "B超", "检测"],
    "disease_department": ["挂什么科", "哪个科", "科室", "去哪个医院", "看什么科"],
    "disease_do_eat": ["能吃什么", "宜吃", "推荐食物", "食疗", "吃什么好"],
    "disease_not_eat": ["不能吃什么", "忌口", "禁忌食物", "不能吃", "避免吃"],
    "symptom_disease": ["什么病", "可能是什么", "是什么引起的", "对应什么病", "预示", "得了什么"],
}

# 在 prompt.py 中补充（可选优化）
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个专业的医疗问答助手。请根据对话历史，将用户的最新问题改写为一个独立、完整、明确的医疗问题。\n"
     "要求：\n"
     "1. 补全代词指代（如'它'、'这个病'）。\n"
     "2. 只输出改写后的问题本身，绝对不要包含任何解释、前缀或标点符号以外的多余内容。\n"
     "3. 如果当前问题已经足够明确，直接原样输出即可。"),
    ("human", "对话历史:\n{chat_history}\n\n当前用户问题: {input}")
])