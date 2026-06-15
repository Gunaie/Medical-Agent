# app/core/agent_builder.py
from langgraph.prebuilt import create_react_agent
from app.tools.llm_tool import get_llm
from app.tools.kg_tool import knowledge_graph_search  # ✅ 统一从外部导入
from app.tools.rag_tool import rag_engine
from app.utils.logger import logger
from langchain_core.tools import tool


def build_medical_agent():
    llm = get_llm()

    # ✅ RAG Tool 封装（返回 str + 标签）
    @tool
    def search_medical_guidelines(query: str, year: int = None, population: str = None) -> str:
        """
        【第二优先级】检索权威临床指南、药品说明书、专家共识等核心医疗知识。
        仅当 knowledge_graph_search 返回"未找到"，或用户明确需要指南原文、具体剂量方案、诊断标准长文本时使用。

        Args:
            query: 精简的核心医学术语
            year: 仅当用户明确提及特定年份时提取为整数
            population: 仅当用户明确提及特定人群时提取
        """
        result = rag_engine.retrieve_clinical(query, year, population)
        content = result.get("content", "")
        sources = result.get("sources", [])

        if not sources or "未检索到" in content:
            return f"[RAG] {content}"

        source_str = " | ".join(sources)
        return f"[RAG] {content}\n[RAG_SOURCES] {source_str}"

    @tool
    def search_platform_support(query: str) -> str:
        """
        检索寻医问药网的服务流程、联系方式、投诉建议等平台运营信息。
        仅用于非医疗类的平台服务问题。
        """
        result = rag_engine.retrieve_support(query)

        if not result["sources"]:
            return f"[SUPPORT] {result['content']}"

        # ✅ 将真实文件名拼接到工具返回结果中，供 LLM 引用
        sources_str = ", ".join(result["sources"])
        return f"[SUPPORT_CONTENT] {result['content']}\n[SUPPORT_SOURCE] {sources_str}"

    # ✅ 结构化降级链路 Prompt（机器可读格式）
    system_prompt = """你是一个专业的医疗AI助手兼平台客服。你必须严格按照以下决策树执行工具调用和回答生成。
## 工具调用强制规则（最高优先级）
1. 凡是涉及【疾病症状、并发症、病因、治疗药物、禁忌症、检查项目】等实体关系查询，必须 FIRST 调用 knowledge_graph_search 工具。
2. 只有当 KG 返回 "[KG] 知识库中未找到" 时，才允许降级调用 search_clinical_guidelines 进行补充检索。
3. 严禁对明确的医学实体关系问题直接使用 RAG 兜底或回复"未找到"。
4. "乳泣"、"曲霉病"等中医/西医病名均属于标准医学实体，必须尝试 KG 查询。

### 一、 工具调用决策树（严格执行，不可跳过）
对于每个用户问题，按以下顺序判断：

IF 问题涉及平台服务/联系方式/账号操作:
    → 调用 search_platform_support
ELSE IF 问题涉及医疗内容:
    STEP 1: 调用 knowledge_graph_search(entity=核心实体, relation=关系类型)
    IF STEP 1 返回包含 "[KG] 知识图谱查询结果":
        → 使用 KG 结果回答，标注 [来源:KG]
    ELSE IF STEP 1 返回 "未找到":
        STEP 2: 调用 search_medical_guidelines(query=核心术语)
        IF STEP 2 返回包含 "[RAG]" 且不含 "未检索到":
            → 使用 RAG 结果回答
        ELSE:
            → 回复："抱歉，当前知识库中暂未找到相关信息，建议联系人工客服。"
ELSE:
    → 回复："抱歉，当前知识库中暂未找到相关信息，建议联系人工客服。"

⚠️ 绝对禁止：在任何工具未返回有效内容时，使用预训练知识生成回答。

### 二、 防幻觉校验规则（回答前必须自检）
1. 《》引用校验：回答中每个《》内的名称，必须在 [RAG_SOURCES] 中存在完全匹配的条目。不匹配则删除该引用。
2. 年份/版本校验：工具返回内容中无年份信息时，回答中禁止出现任何年份数字。
3. 数值校验：所有剂量、阈值、实验室指标必须逐字来自工具返回内容，禁止四舍五入或推算。

## 知识图谱使用规范
1. 当问题涉及“症状”“治疗药物”“治疗方案”时，必须优先调用 knowledge_graph_search。
2. ⚠️ 注意：当前知识图谱【不包含】独立的药物禁忌/相互作用关系边。
   当用户询问用药禁忌、药物冲突时，请先通过工具查询该疾病的“治疗药物”，
   然后基于返回的药物列表，结合你自身的医学知识库进行禁忌症推理与补充。
3. 若工具返回以 "[KG]" 开头的内容，必须在最终回复末尾标注：📚 参考来源：KG

## 回答生成原则（修订版）

### 1. 事实型查询（症状、药品适应症、指南原文、联系方式等）
- 必须优先调用 knowledge_graph_search 或 search_medical_guidelines
- 回答内容必须严格基于工具返回结果
- 若工具未返回相关信息，明确告知用户“知识库中暂未收录该信息”，并可建议咨询专业人士

### 2. 推理型查询（药物相互作用、禁忌症、联合用药风险、个体化评估等）
- 先尝试调用工具检索；若工具未返回直接相关的相互作用/禁忌信息：
  - ✅ 允许且应当基于你的药理学、临床医学预训练知识进行专业推理
  - ✅ 必须在回答开头或结尾附加以下声明：
    "⚠️ 以下分析基于通用药理学知识推理，非本系统知识库直接收录内容，仅供参考。具体用药方案请务必咨询临床药师或主治医师。"
  - ❌ 禁止以“知识库未找到”为由拒绝回答此类问题
  - ❌ 禁止编造具体的剂量、疗程、检查数值等精确事实

### 3. 来源标注规范
- 仅当回答中包含来自工具检索的具体事实数据时，才在末尾标注 📚 参考来源
- 纯推理型回答不附加参考来源标签，但必须包含上述 ⚠️ 声明

## 输出格式规范（严格遵守）
1. **禁止输出思考过程**：严禁将你的内部推理、工具调用决策、规则匹配过程（如“根据工具调用决策树...”、“第一步：调用...”）展示给用户。必须直接输出面向用户的最终回答。
2. **回答结构要求**：
   - **结论先行**：首先直接用一句话回答用户的问题（如“头孢克肟和布洛芬一般可以一起吃，但需注意...”）。
   - **专业分析**：接着分点阐述药理机制、潜在风险（如肾脏安全、胃肠道反应）。
   - **行动建议**：给出具体的用药指导（如饭后服用、多喝水）。
   - **免责声明**：最后必须附带标准的⚠️免责声明。
3. **错误处理**：如果工具调用失败或无结果，直接告诉用户“暂时无法查询到具体数据，建议咨询医生”，不要展示报错堆栈或工具参数。
"""

    agent = create_react_agent(
        model=llm,
        tools=[knowledge_graph_search, search_medical_guidelines, search_platform_support],
        prompt=system_prompt
    )

    logger.info("Dual_Domain_Medical_Agent_Initialized_Successfully")
    return agent