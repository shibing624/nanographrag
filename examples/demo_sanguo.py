# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 

graphrag-lite 三国演义示例 (演示异步 API)

演示:
1. 异步文档插入 (ainsert)
2. 异步查询 (aquery)
3. 异步流式输出
"""

import os
import sys
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphrag_lite import GraphRAGLite


async def main():
    # 初始化
    graph = GraphRAGLite(
        storage_path="./tmp/graphrag_sanguo",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        enable_cache=True,
    )
    
    # 加载三国演义文本
    txt_path = os.path.join(os.path.dirname(__file__), "三国演义.txt")
    if not os.path.exists(txt_path):
        print(f"错误: 未找到文件 {txt_path}")
        return
    
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()[:10000]
    
    # 异步插入数据 (已有数据则跳过)
    if not graph.has_data():
        print("=" * 60)
        print("异步构建知识图谱...")
        print("=" * 60)
        result = await graph.ainsert(text, doc_id="三国演义")
        print(f"插入结果: {result}")
    else:
        print("=" * 60)
        print("已有数据，跳过插入")
        print("=" * 60)
    
    # 统计
    stats = graph.get_stats()
    print(f"\n统计: {stats}")
    print(f"实体数量: {len(graph.list_entities())}")
    print(f"关系数量: {len(graph.list_relations())}")
    
    # 异步查询测试
    print("\n" + "=" * 60)
    print("异步查询测试")
    print("=" * 60)
    
    questions = [
        "大乔和曹操是什么关系？",
        # "董卓是什么样的人？",
        # "桃园结义讲了什么故事？",
    ]
    
    for q in questions:
        print(f"\n问题: {q}")
        print("-" * 50)
        
        for mode in ["local", "global", "mix", "naive"]:
            answer = await graph.aquery(q, mode=mode, top_k=5)
            short = answer
            print(f"[{mode:6}] {short}")
    
    # 异步流式输出
    print("\n" + "=" * 60)
    print("异步流式输出 (mix 模式)")
    print("=" * 60)
    print("\n问题: 曹操是什么样的人？他有什么特点？")
    print("-" * 50)
    
    stream = await graph.aquery("曹操是什么样的人？他有什么特点？", mode="mix", stream=True)
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
