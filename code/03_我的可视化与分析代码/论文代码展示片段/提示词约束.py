SYSTEM_PROMPT = "你是严谨的新闻事实分析助手。"
REPORT_PROMPT_TEMPLATE = """
请基于以下输入，生成中文 Markdown 分析报告。
要求：
1. 只能输出以下四个一级标题，且顺序固定：
   ## 事实核查要点## 逻辑漏洞分析## 图片可信度评估## 最终结论
2. 最终结论必须与模型预测保持一致，只做解释增强，不得改写预测标签。
3. 语句连贯，避免一句话中间断行。
输入信息：
- 标题：{title}
- 正文：{content}
- 模型预测：{prediction}
- 置信度：{confidence}
- 图片信息：{image_notes}

外部检索证据：
{evidence_block}
""".strip()
