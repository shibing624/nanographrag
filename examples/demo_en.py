# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 

graphrag-lite 使用示例

演示:
1. 文档插入与知识图谱构建
2. 四种查询模式 (local/global/mix/naive)
3. 流式输出
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphrag_lite import GraphRAGLite


SAMPLE_TEXT = """
A Christmas Carol by Charles Dickens

Marley was dead: to begin with. There is no doubt whatever about that. 
The register of his burial was signed by the clergyman, the clerk, the undertaker, 
and the chief mourner. Scrooge signed it. And Scrooge's name was good upon 'Change, 
for anything he chose to put his hand to. Old Marley was as dead as a door-nail.

Scrooge knew he was dead? Of course he did. How could it be otherwise? 
Scrooge and he were partners for I don't know how many years. 
Scrooge was his sole executor, his sole administrator, his sole assign, 
his sole residuary legatee, his sole friend, and sole mourner.

Scrooge never painted out Old Marley's name. There it stood, years afterwards, 
above the warehouse door: Scrooge and Marley. The firm was known as Scrooge and Marley. 
Sometimes people new to the business called Scrooge Scrooge, and sometimes Marley, 
but he answered to both names. It was all the same to him.

Oh! But he was a tight-fisted hand at the grindstone, Scrooge! 
a squeezing, wrenching, grasping, scraping, clutching, covetous, old sinner! 
Hard and sharp as flint, from which no steel had ever struck out generous fire; 
secret, and self-contained, and solitary as an oyster.
"""


def main():
    # 初始化
    graph = GraphRAGLite(
        storage_path="./tmp/graphrag_demo_en",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        enable_cache=True,
    )
    
    # 插入数据 (已有数据则跳过)
    if not graph.has_data():
        print("=" * 60)
        print("构建知识图谱...")
        print("=" * 60)
        result = graph.insert(SAMPLE_TEXT, doc_id="christmas_carol")
        print(f"插入结果: {result}")
    else:
        print("=" * 60)
        print("已有数据，跳过插入")
        print("=" * 60)
    
    # 统计
    stats = graph.get_stats()
    print(f"\n统计: {stats}")
    print(f"实体: {graph.list_entities()}")
    print(f"关系: {graph.list_relations()}")
    
    # 查询测试
    print("\n" + "=" * 60)
    print("查询测试")
    print("=" * 60)
    
    questions = [
        "What is the relationship between Scrooge and Marley?",
        "What kind of person is Scrooge?",
    ]
    
    for q in questions:
        print(f"\n问题: {q}")
        print("-" * 50)
        
        for mode in ["local", "global", "mix", "naive"]:
            answer = graph.query(q, mode=mode, top_k=5)
            short = answer
            print(f"[{mode:6}] {short}")
    
    # 流式输出
    print("\n" + "=" * 60)
    print("流式输出 (mix 模式)")
    print("=" * 60)
    print("\n问题: Who is Scrooge?")
    print("-" * 50)
    for chunk in graph.query("Who is Scrooge?", mode="mix", stream=True):
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()
