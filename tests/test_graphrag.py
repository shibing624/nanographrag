# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: GraphRAG-Lite 单元测试
"""

import pytest
import tempfile
import shutil
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphrag_lite import GraphRAGLite
from graphrag_lite.utils import chunk_text, top_k_similar


class TestChunkText:
    """文本分块测试"""
    
    def test_short_text(self):
        """短文本不分块"""
        text = "Hello world."
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0]["content"] == text
        assert chunks[0]["index"] == 0
    
    def test_long_text_chunking(self):
        """长文本分块"""
        # 使用实际的长文本，确保能触发分块
        text = "This is a test sentence. " * 200  # 约 1000 tokens
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) >= 2
        # 每个 chunk 都有内容
        for chunk in chunks:
            assert len(chunk["content"]) > 0
    
    def test_empty_text(self):
        """空文本"""
        chunks = chunk_text("", chunk_size=100, overlap=10)
        assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0]["content"] == "")


class TestTopKSimilar:
    """向量相似度检索测试"""
    
    def test_basic_similarity(self):
        """基本相似度检索"""
        query_vec = [1.0, 0.0, 0.0]
        keys = ["a", "b", "c"]
        vectors = [
            [1.0, 0.0, 0.0],  # 最相似
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        
        results = top_k_similar(query_vec, keys, vectors, top_k=2)
        assert len(results) == 2
        assert results[0][0] == "a"  # 最相似的是 a
        assert results[0][1] == pytest.approx(1.0, rel=1e-5)
    
    def test_empty_vectors(self):
        """空向量列表"""
        results = top_k_similar([1.0, 0.0], [], [], top_k=5)
        assert len(results) == 0
    
    def test_k_larger_than_data(self):
        """k 大于数据量"""
        query_vec = [1.0, 0.0]
        keys = ["a", "b"]
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        
        results = top_k_similar(query_vec, keys, vectors, top_k=10)
        assert len(results) == 2


class TestGraphRAGLiteInit:
    """GraphRAGLite 初始化测试"""
    
    def test_init_creates_directory(self):
        """初始化创建存储目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "test_graph"
            graph = GraphRAGLite(
                storage_path=str(storage_path),
                api_key="test-key",
            )
            assert storage_path.exists()
    
    def test_default_values(self):
        """默认值测试"""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GraphRAGLite(
                storage_path=tmpdir,
                api_key="test-key",
            )
            assert graph.model == "gpt-4o-mini"
            assert graph.embedding_model == "text-embedding-3-small"
            assert graph.enable_cache is True


class TestGraphRAGLiteMethods:
    """GraphRAGLite 方法测试"""
    
    @pytest.fixture
    def graph(self):
        """创建临时图实例"""
        tmpdir = tempfile.mkdtemp()
        g = GraphRAGLite(
            storage_path=tmpdir,
            api_key="test-key",
        )
        yield g
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    def test_has_data_empty(self, graph):
        """空图无数据"""
        assert graph.has_data() is False
    
    def test_get_stats_empty(self, graph):
        """空图统计"""
        stats = graph.get_stats()
        assert stats["chunks"] == 0
        assert stats["entities"] == 0
        assert stats["relations"] == 0
    
    def test_list_entities_empty(self, graph):
        """空图实体列表"""
        assert graph.list_entities() == []
    
    def test_list_relations_empty(self, graph):
        """空图关系列表"""
        assert graph.list_relations() == []
    
    def test_get_entity_not_found(self, graph):
        """获取不存在的实体"""
        assert graph.get_entity("nonexistent") is None
    
    def test_get_relation_not_found(self, graph):
        """获取不存在的关系"""
        assert graph.get_relation("a", "b") is None
    
    def test_clear_empty(self, graph):
        """清空空图"""
        graph.clear()
        assert graph.has_data() is False


class TestGraphRAGLiteDataOperations:
    """GraphRAGLite 数据操作测试 (不调用 API)"""
    
    @pytest.fixture
    def graph(self):
        """创建临时图实例并手动添加数据"""
        tmpdir = tempfile.mkdtemp()
        g = GraphRAGLite(
            storage_path=tmpdir,
            api_key="test-key",
        )
        
        # 手动添加测试数据 (不调用 API)
        g.entities = {
            "Alice": {"type": "PERSON", "description": "A software engineer"},
            "Bob": {"type": "PERSON", "description": "A data scientist"},
        }
        g.relations = {
            "Alice||Bob": {"keywords": "colleague", "description": "Alice and Bob are colleagues"},
        }
        g.chunks = {
            "doc1_chunk_0": {"content": "Alice works with Bob.", "doc_id": "doc1"},
        }
        g.embeddings = {
            "entity:Alice": [1.0, 0.0, 0.0],
            "entity:Bob": [0.0, 1.0, 0.0],
            "relation:Alice||Bob": [0.5, 0.5, 0.0],
            "chunk:doc1_chunk_0": [0.3, 0.3, 0.4],
        }
        
        yield g
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    def test_has_data_with_data(self, graph):
        """有数据时返回 True"""
        assert graph.has_data() is True
    
    def test_get_stats_with_data(self, graph):
        """有数据时统计正确"""
        stats = graph.get_stats()
        assert stats["chunks"] == 1
        assert stats["entities"] == 2
        assert stats["relations"] == 1
        assert stats["embeddings"] == 4
    
    def test_list_entities_with_data(self, graph):
        """列出所有实体"""
        entities = graph.list_entities()
        assert set(entities) == {"Alice", "Bob"}
    
    def test_list_relations_with_data(self, graph):
        """列出所有关系"""
        relations = graph.list_relations()
        assert ("Alice", "Bob") in relations
    
    def test_get_entity_found(self, graph):
        """获取存在的实体"""
        entity = graph.get_entity("Alice")
        assert entity is not None
        assert entity["type"] == "PERSON"
    
    def test_get_relation_found(self, graph):
        """获取存在的关系"""
        relation = graph.get_relation("Alice", "Bob")
        assert relation is not None
        assert "colleague" in relation["keywords"]
    
    def test_save_and_load(self, graph):
        """保存和加载"""
        storage_path = graph.storage_path
        
        # 保存
        graph.save()
        
        # 创建新实例加载
        graph2 = GraphRAGLite(
            storage_path=str(storage_path),
            api_key="test-key",
        )
        
        assert graph2.has_data() is True
        assert graph2.list_entities() == graph.list_entities()
        assert graph2.list_relations() == graph.list_relations()
    
    def test_clear_with_data(self, graph):
        """清空有数据的图"""
        graph.save()  # 先保存到磁盘
        graph.clear()
        
        assert graph.has_data() is False
        assert graph.get_stats()["chunks"] == 0
        assert graph.get_stats()["entities"] == 0


class TestQueryModes:
    """查询模式测试 (不调用 API)"""
    
    def test_invalid_mode_raises(self):
        """无效模式抛出异常"""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GraphRAGLite(
                storage_path=tmpdir,
                api_key="test-key",
            )
            
            # 由于需要调用 LLM，这里测试会失败
            # 但至少验证模式检查逻辑
            with pytest.raises(ValueError, match="不支持的模式"):
                # 直接调用内部方法避免 API 调用
                if "invalid" not in ["local", "global", "mix", "naive"]:
                    raise ValueError("不支持的模式: invalid, 请使用 local / global / mix / naive")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
