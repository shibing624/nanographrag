# GraphRAG-Lite

<p align="center">
  <img src="https://github.com/shibing624/graphrag-lite/blob/main/docs/logo.svg" alt="GraphRAG-Lite Logo" width="400">
</p>

<p align="center">
  <b>极简 GraphRAG 实现，约 500 行 Python 代码</b>
</p>

<p align="center">
  <a href="https://badge.fury.io/py/graphrag-lite"><img src="https://badge.fury.io/py/graphrag-lite.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://github.com/shibing624/graphrag-lite/blob/main/README_zh.md"><img src="https://img.shields.io/badge/wechat-group-green.svg?logo=wechat" alt="Chat Group"></a>
</p>

<p align="center">
  <a href="https://github.com/shibing624/graphrag-lite/blob/main/README.md">English</a>
</p>

GraphRAG-Lite 是一个简洁、教学导向的 GraphRAG（基于图的检索增强生成）实现。非常适合学习知识图谱增强 RAG 系统的核心原理。

## 为什么选择 GraphRAG-Lite？

- **阅读即学习**：清晰、文档完善的代码，一个下午就能理解
- **生产级模式**：批量 Embedding、LLM 缓存等真实优化
- **灵活检索**：4 种查询模式适应不同场景
- **依赖精简**：仅需 `openai`、`numpy`、`tiktoken`、`loguru`

## 特性

| 特性 | 说明 |
|------|------|
| **4 种查询模式** | `local`、`global`、`mix`、`naive` - 选择合适的策略 |
| **批量 Embedding** | 智能批处理减少 API 调用 |
| **LLM 缓存** | 避免重复的 LLM 请求 |
| **流式输出** | 实时响应流 |
| **NumPy 加速** | 快速向量相似度搜索 |
| **持久化存储** | 基于 JSON 存储，无需外部数据库 |

## 安装

```bash
pip install graphrag-lite
```

或从源码安装：

```bash
git clone https://github.com/shibing624/graphrag-lite.git
cd graphrag-lite
pip install -e .
```

## 快速开始

```python
import os
from graphrag_lite import GraphRAGLite

# 初始化
graph = GraphRAGLite(
    storage_path="./my_graph",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),  # 可选：兼容 API
)

# 插入文档
graph.insert("""
贾宝玉是《红楼梦》的主人公，与林黛玉青梅竹马。
林黛玉才华横溢，是贾母的外孙女。
薛宝钗最终嫁给了贾宝玉。
""")

# 基于知识图谱上下文查询
answer = graph.query("贾宝玉和林黛玉是什么关系？")
print(answer)
```

## 查询模式

| 模式 | 策略 | 适用场景 |
|------|------|----------|
| `local` | 实体 → 相关关系 | "XX 是谁？"类问题 |
| `global` | 关系 → 相关实体 | "XX 和 YY 什么关系？" |
| `mix` | 实体 + 关系 + 文本块 | **通用场景（推荐）** |
| `naive` | 仅文本块 | 基线对比 |

```python
# 根据问题选择合适的模式
answer = graph.query("贾宝玉是谁？", mode="local")
answer = graph.query("贾宝玉和林黛玉什么关系？", mode="global")
answer = graph.query("介绍一下红楼梦", mode="mix")      # 推荐
answer = graph.query("发生了什么？", mode="naive")
```

## 流式输出

```python
for chunk in graph.query("贾宝玉是谁？", stream=True):
    print(chunk, end="", flush=True)
```

## API 参考

### GraphRAGLite

```python
GraphRAGLite(
    storage_path: str = "./graphrag_data",  # 数据存储目录
    api_key: str = None,                     # OpenAI API 密钥
    base_url: str = None,                    # OpenAI 兼容 API 地址
    model: str = "gpt-4o-mini",              # LLM 模型
    embedding_model: str = "text-embedding-3-small",  # Embedding 模型
    enable_cache: bool = True,               # 启用 LLM 响应缓存
)
```

### 方法

| 方法 | 说明 |
|------|------|
| `insert(text, doc_id=None)` | 插入文档并构建知识图谱 |
| `query(question, mode="mix", top_k=10, stream=False)` | 查询知识图谱 |
| `has_data()` | 检查图谱是否有数据 |
| `get_stats()` | 获取图谱统计信息 |
| `list_entities()` | 列出所有实体 |
| `list_relations()` | 列出所有关系 |
| `clear()` | 清空所有数据 |

## 工作原理

<p align="center">
  <img src="https://github.com/shibing624/graphrag-lite/blob/main/docs/workflow.svg" alt="GraphRAG-Lite 工作流程" width="800">
</p>

**插入流程：**
```
文档 → 分块 → LLM 实体提取 → 批量 Embedding → 存储
```

**查询流程：**
```
问题 → 向量检索 → 上下文构建 → LLM 生成 → 答案
```

## 应用场景

- **学习 GraphRAG**：理解知识图谱如何增强 RAG
- **原型验证**：快速验证 GraphRAG 在你的领域是否有效
- **研究基线**：比较不同检索策略的基准
- **教学材料**：RAG 概念的教学素材

## 社区与支持

*   **GitHub Issues**：[提交 issue](https://github.com/shibing624/graphrag-lite/issues)
*   **微信**：添加 `xuming624`，备注 "llm" 加入群聊

<img src="https://github.com/shibing624/graphrag-lite/blob/main/docs/wechat.jpeg" width="200" />

## 许可证

Apache License 2.0

## 引用

```bibtex
@software{graphrag-lite,
  author = {Xu Ming},
  title = {GraphRAG-Lite: Minimal GraphRAG Implementation},
  year = {2025},
  url = {https://github.com/shibing624/graphrag-lite}
}
```
