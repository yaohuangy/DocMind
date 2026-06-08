"""
语义记忆模块。

基于 Neo4j 图数据库存储概念知识图谱。
包含 Concept 节点和 RELATES_TO / MENTIONED_IN / FOUND_IN 关系。

核心操作：
- 添加/更新概念（MERGE，频率递增）
- 创建概念间关系
- 搜索概念
- 获取子图/全部概念
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.graph_store import GraphStore
from src.memory.models import ConceptNode, Relation

logger = logging.getLogger(__name__)


class SemanticMemory:
    """语义记忆——基于 Neo4j 的知识图谱存储。

    封装概念节点的 CRUD 和关系管理，为学习回顾提供结构化
    的知识表示。

    Usage::

        sm = SemanticMemory(graph_store)
        sm.connect()
        sm.create_constraints()
        sm.add_concept("Self-Attention", "mechanism", "自注意力机制...")
        sm.add_relation("Self-Attention", "Transformer", "RELATES_TO", 0.9)
        results = sm.search_concepts("attention")
    """

    def __init__(self, graph_store: Optional[GraphStore] = None) -> None:
        """
        Args:
            graph_store: 图数据库封装。None 则自动创建。
        """
        self._graph = graph_store or GraphStore()
        self._connected = False

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """建立 Neo4j 连接并创建约束。"""
        if not self._connected:
            self._graph.connect()
            self._graph.create_constraints()
            self._connected = True

    def close(self) -> None:
        """关闭连接。"""
        if self._connected:
            self._graph.close()
            self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @property
    def is_connected(self) -> bool:
        """是否已连接。"""
        return self._connected

    # ------------------------------------------------------------------
    # 概念节点操作
    # ------------------------------------------------------------------

    def add_concept(
        self,
        name: str,
        concept_type: str = "concept",
        description: str = "",
        user_id: str = "default",
    ) -> ConceptNode:
        """添加或更新概念节点（MERGE 语义）。

        如果概念已存在，则增加 frequency 计数器；
        否则创建新节点。

        Args:
            name: 概念名称。
            concept_type: 概念类型。
            description: 详细描述。
            user_id: 所属用户。

        Returns:
            ConceptNode 实例。
        """
        self._ensure_connected()

        ts = datetime.now().isoformat()

        result = self._graph.merge_concept(
            name=name,
            concept_type=concept_type,
            description=description,
            first_encountered=ts,
            user_id=user_id,
        )

        if result:
            node = ConceptNode(
                name=result.get("name", name),
                concept_type=result.get("type", concept_type),
                description=result.get("description", description),
                first_encountered=result.get("first_encountered", ts),
                frequency=int(result.get("frequency", 1)),
            )
            logger.info("语义记忆: 概念 '%s' (freq=%d)", node.name, node.frequency)
            return node

        # 回退：返回构造的节点
        return ConceptNode(
            name=name,
            concept_type=concept_type,
            description=description,
            first_encountered=ts,
            frequency=1,
        )

    def add_concepts_batch(
        self,
        concepts: List[Dict[str, str]],
        user_id: str = "default",
    ) -> List[ConceptNode]:
        """批量添加概念。

        Args:
            concepts: 概念字典列表，每个含 name, type, description。
            user_id: 所属用户。

        Returns:
            ConceptNode 列表。
        """
        results: List[ConceptNode] = []
        for c in concepts:
            try:
                node = self.add_concept(
                    name=c.get("name", ""),
                    concept_type=c.get("type", "concept"),
                    description=c.get("description", ""),
                    user_id=user_id,
                )
                results.append(node)
            except Exception as e:
                logger.warning("概念 '%s' 添加失败: %s", c.get("name"), e)
        return results

    def get_concept(self, name: str, user_id: str = "") -> Optional[ConceptNode]:
        """按名称查询概念。

        Args:
            name: 概念名称。
            user_id: 按用户过滤，空则不过滤。

        Returns:
            ConceptNode，不存在返回 None。
        """
        self._ensure_connected()

        data = self._graph.get_concept(name, user_id=user_id)
        if not data:
            return None
        return ConceptNode.from_dict(data)

    def search_concepts(
        self, keyword: str, limit: int = 20, user_id: str = "",
    ) -> List[ConceptNode]:
        """按关键词搜索概念。

        Args:
            keyword: 搜索关键词。
            limit: 返回结果上限。
            user_id: 按用户过滤。

        Returns:
            匹配的 ConceptNode 列表，按频率降序。
        """
        self._ensure_connected()

        results = self._graph.search_concepts(keyword, limit=limit, user_id=user_id)
        return [ConceptNode.from_dict(r) for r in results]

    def get_all_concepts(self, limit: int = 100, user_id: str = "") -> List[ConceptNode]:
        """获取所有概念节点。

        Args:
            limit: 返回结果上限。
            user_id: 按用户过滤。

        Returns:
            ConceptNode 列表，按频率降序。
        """
        self._ensure_connected()

        results = self._graph.get_all_concepts(limit=limit, user_id=user_id)
        return [ConceptNode.from_dict(r) for r in results]

    # ------------------------------------------------------------------
    # 关系操作
    # ------------------------------------------------------------------

    def add_relation(
        self,
        source_name: str,
        target_name: str,
        rel_type: str = "RELATES_TO",
        strength: float = 0.5,
        description: str = "",
        user_id: str = "default",
    ) -> Relation:
        """在两个概念之间创建/更新关系。

        如果源或目标概念不存在，会自动创建占位节点。

        Args:
            source_name: 源概念名称。
            target_name: 目标概念名称。
            rel_type: 关系类型。
            strength: 关联强度（0~1）。
            description: 关系描述。
            user_id: 所属用户（确保概念归属正确）。

        Returns:
            Relation 实例。
        """
        self._ensure_connected()

        # 确保两个概念节点存在但不递增频率（关系建立不算"再遇"）
        self._graph.ensure_concept_exists(source_name, user_id=user_id)
        self._graph.ensure_concept_exists(target_name, user_id=user_id)

        self._graph.relate_concepts(
            source_name=source_name,
            target_name=target_name,
            rel_type=rel_type,
            strength=strength,
            description=description,
            user_id=user_id,
        )

        rel = Relation(
            source=source_name,
            target=target_name,
            rel_type=rel_type,
            strength=strength,
            description=description,
        )
        logger.info("语义记忆关系: (%s)-[:%s]->(%s)", source_name, rel_type, target_name)
        return rel

    def link_concept_to_document(
        self,
        concept_name: str,
        doc_id: str,
        location_ref: str = "",
    ) -> None:
        """将概念链接到文档（FOUND_IN 关系）。

        Args:
            concept_name: 概念名称。
            doc_id: 文档 ID。
            location_ref: 文档中的位置引用。
        """
        self._ensure_connected()
        self._graph.link_concept_to_document(concept_name, doc_id, location_ref)

    # ------------------------------------------------------------------
    # 图谱查询
    # ------------------------------------------------------------------

    def get_neighbourhood(
        self, concept_name: str, depth: int = 1
    ) -> List[Dict[str, Any]]:
        """获取概念的邻域子图。

        Args:
            concept_name: 中心概念名称。
            depth: 遍历深度。

        Returns:
            图路径数据列表。
        """
        self._ensure_connected()
        return self._graph.get_neighbourhood(concept_name, depth=depth)

    def get_concept_count(self, user_id: str = "") -> int:
        """获取概念总数。

        Args:
            user_id: 按用户过滤。
        """
        self._ensure_connected()
        # 直接用 Cypher COUNT 而非先取全部再 len()
        if user_id:
            result = self._graph.run_query(
                "MATCH (c:Concept) WHERE c.user_id = $uid RETURN count(c) AS cnt",
                {"uid": user_id},
            )
        else:
            result = self._graph.run_query(
                "MATCH (c:Concept) RETURN count(c) AS cnt"
            )
        count = result[0]["cnt"] if result else 0
        print(f"[NEO4J] get_concept_count: user_id='{user_id}', count={count}")
        return count

    def get_graph_summary(self) -> Dict[str, Any]:
        """获取知识图谱摘要。

        Returns:
            包含概念总数和 Top-K 概念的字典。
        """
        self._ensure_connected()

        concepts = self.get_all_concepts(limit=50)
        return {
            "total_concepts": self.get_concept_count(),
            "top_concepts": [
                {"name": c.name, "type": c.concept_type, "frequency": c.frequency}
                for c in concepts[:20]
            ],
        }

    def get_user_graph_data(self, user_id: str, top_n: int = 0) -> dict:
        """获取用户的知识图谱数据（节点 + 关系）。

        供可视化页面使用。返回 {nodes: [...], edges: [...]} 结构。

        Args:
            user_id: 用户 ID。
            top_n: 仅返回频率最高的 top_n 个概念，0 表示返回全部。

        Returns:
            字典，包含 nodes 列表和 edges 列表。
        """
        self._ensure_connected()
        if top_n > 0:
            return self._graph.get_user_graph_data_top_n(user_id, top_n=top_n)
        return self._graph.get_user_graph_data(user_id)

    def seed_demo_frequencies(self, user_id: str = "") -> int:
        """为概念节点分配模拟的随机频率（仅用于演示/开发）。

        Args:
            user_id: 按用户过滤，空则更新所有用户的概念。

        Returns:
            更新的节点数量。
        """
        self._ensure_connected()
        return self._graph.seed_demo_frequencies(user_id=user_id)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        """确保已连接。"""
        if not self._connected:
            self.connect()
