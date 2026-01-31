# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 

graphrag-lite 中文使用示例

演示:
1. 中文文档插入与知识图谱构建
2. 四种查询模式 (local/global/mix/naive)
3. 流式输出
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphrag_lite import GraphRAGLite


SAMPLE_TEXT = """
《红楼梦》是中国古典四大名著之一，作者曹雪芹。

贾宝玉是荣国府贾政与王夫人所生的次子，自幼聪颖灵慧，却厌恶仕途经济，被寄予厚望却叛逆不羁。
他与林黛玉青梅竹马，两人情投意合，但最终因家族利益和封建礼教的束缚，未能结为连理。

林黛玉是贾母的外孙女，父亲林如海是探花出身，母亲贾敏是贾母之女。
黛玉自幼体弱多病，才华横溢，诗词歌赋无一不精。她性格敏感多疑，与宝玉相知相惜。

薛宝钗是王夫人的外甥女，薛姨妈之女。她容貌丰美，举止娴雅，博学多才，深得贾府上下喜爱。
宝钗最终嫁与宝玉为妻，但宝玉心系黛玉，婚后出家为僧。

王熙凤是贾琏之妻，人称"凤姐"。她精明能干，口才了得，掌管荣国府内务。
虽然手段泼辣，但也有心狠手辣的一面，曾设计害死尤二姐。
"""


def main():
    # 初始化
    graph = GraphRAGLite(
        storage_path="./tmp/graphrag_demo_zh",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        enable_cache=True,
    )
    
    # 插入数据 (已有数据则跳过)
    if not graph.has_data():
        print("=" * 60)
        print("构建知识图谱...")
        print("=" * 60)
        result = graph.insert(SAMPLE_TEXT, doc_id="红楼梦")
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
        "贾宝玉和林黛玉是什么关系？",
        "薛宝钗是什么样的人？",
        "王熙凤在贾府担任什么职务？",
        "孙悟空是谁？"
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
    print("\n问题: 红楼梦的主要人物有哪些？")
    print("-" * 50)
    for chunk in graph.query("红楼梦的主要人物有哪些？", mode="mix", stream=True):
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()
