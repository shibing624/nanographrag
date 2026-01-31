# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: NanoGraphRAG Prompt 模板
"""

# 实体和关系提取
ENTITY_EXTRACTION_PROMPT = """你是知识图谱专家，从以下文本中提取实体和关系。

文本:
{text}

输出格式 (每行一条):
- Entity: entity||名称||类型||描述
- Relation: relation||源实体||目标实体||关键词||描述

实体类型: 人物、组织、地点、事件、概念、物品等
关键词是描述关系的核心词汇
描述必须使用与原文相同的语言

输出:"""


# RAG 回答生成
RAG_RESPONSE_PROMPT = """基于检索到的知识图谱信息回答问题。

=== 知识图谱数据 ===
{context}

=== 问题 ===
{query}

=== 回答要求 ===
1. 严格基于提供的知识图谱数据回答，不要编造信息
2. 使用 Markdown 格式回答
3. 在回答中引用数据来源(如果有），格式为 [Entities (编号); Relationships (编号); Sources (编号)]
   - Entities 对应实体编号
   - Relationships 对应关系编号  
   - Sources 对应文本来源编号
4. 引用应自然融入回答，放在相关陈述的末尾
5. 如果知识不足以回答，请明确说明
6. 使用与问题相同的语言回答

回答:"""
