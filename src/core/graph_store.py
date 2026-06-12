"""
Neo4j 图数据库封装模块。

为语义记忆（知识图谱）提供底层图存储和查询能力。
节点类型：Concept, Document, Session, Note
关系类型：RELATES_TO, MENTIONED_IN, FOUND_IN, ABOUT

Usage::

    store = GraphStore()
    store.connect()
    store.create_constraints()
    store.merge_concept("Self-Attention", "mechanism", "自注意力机制...")
    store.close()
"""

import logging
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from src.core.config import Neo4jConfig, get_config

logger = logging.getLogger(__name__)


class GraphStore:
    """Neo4j 图数据库驱动封装。

    管理连接生命周期，提供节点和关系的 CRUD 操作。
    支持自动创建约束（唯一性）。

    Usage::

        store = GraphStore()
        try:
            store.connect()
            store.merge_concept(...)
        finally:
            store.close()
    """

    def __init__(self, config: Neo4jConfig | None = None):
        """
        Args:
            config: Neo4j 配置。为 None 时自动从全局配置加载。
        """
        if config is None:
            config = get_config().neo4j
        self._config = config
        self._driver = None

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """是否已连接。"""
        return self._driver is not None

    def connect(self) -> None:
        """建立与 Neo4j 的连接。

        Raises:
            ServiceUnavailable: 无法连接到 Neo4j 服务。
        """
        if self._driver is not None:
            return
        self._driver = GraphDatabase.driver(
            self._config.uri,
            auth=(self._config.user, self._config.password),
        )
        # 验证连接
        try:
            self._driver.verify_connectivity()
            logger.info("Neo4j 连接成功: %s", self._config.uri)
        except Exception:
            self._driver = None
            raise ServiceUnavailable(
                f"无法连接到 Neo4j: {self._config.uri}"
            )

    def close(self) -> None:
        """关闭 Neo4j 连接。"""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j 连接已关闭")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # 底层查询接口
    # ------------------------------------------------------------------

    def run_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        database: str | None = None,
    ) -> list[dict[str, Any]]:
        """执行 Cypher 查询并返回结果列表。

        Args:
            query: Cypher 查询语句。
            params: 查询参数。
            database: 目标数据库名，默认使用配置中的 database。

        Returns:
            字典列表，每个字典对应一行结果。
        """
        if not self._driver:
            raise RuntimeError("Neo4j 未连接，请先调用 connect()")

        db = database or self._config.database

        with self._driver.session(database=db) as session:
            result = session.run(query, params or {})
            records = [record.data() for record in result]
            return records

    def execute_write(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        database: str | None = None,
    ) -> list[dict[str, Any]]:
        """执行写事务（Cypher 查询）。

        Args:
            query: Cypher 语句。
            params: 查询参数。
            database: 目标数据库。

        Returns:
            结果字典列表。
        """
        if not self._driver:
            raise RuntimeError("Neo4j 未连接，请先调用 connect()")

        db = database or self._config.database

        def _tx(tx):
            result = tx.run(query, params or {})
            return [record.data() for record in result]

        with self._driver.session(database=db) as session:
            return session.execute_write(_tx)

    # ------------------------------------------------------------------
    # Schema 初始化
    # ------------------------------------------------------------------

    def create_constraints(self) -> None:
        """创建 Neo4j 约束与索引（幂等，已存在则忽略）。

        重要：不再对 name 单独做 UNIQUE 约束，因为概念按 (name, user_id) 多用户隔离。
        旧约束 concept_name_unique 会自动尝试删除以升级 schema。
        """
        # 删除旧版单属性唯一约束（升级兼容）
        try:
            self.run_query("DROP CONSTRAINT concept_name_unique IF EXISTS")
            logger.info("旧版 concept_name_unique 约束已删除（升级 schema）")
        except Neo4jError:
            pass  # 不存在或已删除
        except Exception:
            pass

        # 创建复合索引以加速 (name, user_id) 上的 MERGE 查找
        try:
            self.run_query(
                "CREATE INDEX concept_name_user_id IF NOT EXISTS "
                "FOR (c:Concept) ON (c.name, c.user_id)"
            )
            logger.info("Neo4j 复合索引 concept_name_user_id 已就绪")
        except Neo4jError as e:
            logger.warning("创建复合索引失败（可能已存在）: %s", e)

    # ------------------------------------------------------------------
    # 概念节点 CRUD
    # ------------------------------------------------------------------

    def merge_concept(
        self,
        name: str,
        concept_type: str = "concept",
        description: str = "",
        first_encountered: str = "",
        user_id: str = "default",
    ) -> dict[str, Any]:
        """创建或更新概念节点（MERGE 语义）。

        Args:
            name: 概念名称。
            concept_type: 类型。
            description: 详细描述。
            first_encountered: 首次遇到的时间戳。
            user_id: 所属用户。

        Returns:
            概念节点的属性字典。
        """
        result = self.execute_write(
            """
            MERGE (c:Concept {name: $name, user_id: $user_id})
            ON CREATE SET
                c.type = $type,
                c.description = $description,
                c.first_encountered = $first_encountered,
                c.frequency = 1
            ON MATCH SET
                c.type = CASE WHEN $type <> '' THEN $type ELSE c.type END,
                c.description = CASE WHEN $description <> '' THEN $description ELSE c.description END,
                c.frequency = c.frequency + 1
            RETURN c { .name, .type, .description, .frequency, .user_id } AS concept
            """,
            {
                "name": name,
                "type": concept_type,
                "description": description,
                "first_encountered": first_encountered,
                "user_id": user_id,
            },
        )
        if result:
            return result[0].get("concept", {})
        return {}

    def get_concept(self, name: str, user_id: str = "") -> dict[str, Any] | None:
        """按名称（和可选用户）查询概念节点。

        Args:
            name: 概念名称。
            user_id: 按用户过滤，空则不过滤。

        Returns:
            概念属性字典，未找到返回 None。
        """
        if user_id:
            results = self.run_query(
                "MATCH (c:Concept {name: $name, user_id: $user_id}) "
                "RETURN c { .name, .type, .description, .frequency, .user_id } AS concept",
                {"name": name, "user_id": user_id},
            )
        else:
            results = self.run_query(
                "MATCH (c:Concept {name: $name}) "
                "RETURN c { .name, .type, .description, .frequency, .user_id } AS concept",
                {"name": name},
            )
        return results[0]["concept"] if results else None

    def search_concepts(
        self, keyword: str, limit: int = 20, user_id: str = "",
    ) -> list[dict[str, Any]]:
        """按关键词搜索概念（模糊匹配）。

        Args:
            keyword: 搜索关键词。
            limit: 返回结果上限。
            user_id: 按用户过滤，空则不过滤。

        Returns:
            匹配的概念列表。
        """
        if user_id:
            results = self.run_query(
                """
                MATCH (c:Concept)
                WHERE (c.name CONTAINS $keyword OR c.description CONTAINS $keyword)
                  AND c.user_id = $user_id
                RETURN c { .name, .type, .description, .frequency } AS concept
                ORDER BY c.frequency DESC
                LIMIT $limit
                """,
                {"keyword": keyword, "limit": limit, "user_id": user_id},
            )
        else:
            results = self.run_query(
                """
                MATCH (c:Concept)
                WHERE c.name CONTAINS $keyword
                   OR c.description CONTAINS $keyword
                RETURN c { .name, .type, .description, .frequency } AS concept
                ORDER BY c.frequency DESC
                LIMIT $limit
                """,
                {"keyword": keyword, "limit": limit},
            )
        return [r["concept"] for r in results]

    def delete_concept(self, name: str) -> bool:
        """删除概念节点及其所有关系。

        Args:
            name: 概念名称。

        Returns:
            是否实际删除了节点。
        """
        result = self.execute_write(
            "MATCH (c:Concept {name: $name}) "
            "DETACH DELETE c "
            "RETURN count(c) AS deleted",
            {"name": name},
        )
        deleted = result[0].get("deleted", 0) if result else 0
        return deleted > 0

    def ensure_concept_exists(self, name: str, user_id: str = "default") -> None:
        """确保概念节点存在（仅创建，不递增频率）。

        供 add_relation 内部使用——关系建立不应算作概念"再遇"。

        Args:
            name: 概念名称。
            user_id: 所属用户。
        """
        self.execute_write(
            """
            MERGE (c:Concept {name: $name, user_id: $user_id})
            ON CREATE SET
                c.type = 'concept',
                c.description = '',
                c.first_encountered = '',
                c.frequency = 0
            """,
            {"name": name, "user_id": user_id},
        )

    # ------------------------------------------------------------------
    # 关系操作
    # ------------------------------------------------------------------

    def relate_concepts(
        self,
        source_name: str,
        target_name: str,
        rel_type: str = "RELATES_TO",
        strength: float = 0.5,
        description: str = "",
        user_id: str = "",
    ) -> None:
        """在两个概念之间创建/更新关系。

        Args:
            source_name: 源概念名称。
            target_name: 目标概念名称。
            rel_type: 关系类型名称。
            strength: 关联强度（0~1）。
            description: 关系描述。
            user_id: 所属用户（确保不会跨用户创建关系）。
        """
        if user_id:
            self.execute_write(
                f"""
                MATCH (a:Concept {{name: $source, user_id: $user_id}})
                MATCH (b:Concept {{name: $target, user_id: $user_id}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r.strength = $strength,
                    r.description = $description
                RETURN type(r) AS rel
                """,
                {
                    "source": source_name,
                    "target": target_name,
                    "strength": strength,
                    "description": description,
                    "user_id": user_id,
                },
            )
        else:
            self.execute_write(
                f"""
                MATCH (a:Concept {{name: $source}})
                MATCH (b:Concept {{name: $target}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r.strength = $strength,
                    r.description = $description
                RETURN type(r) AS rel
                """,
                {
                    "source": source_name,
                    "target": target_name,
                    "strength": strength,
                    "description": description,
                },
            )
        logger.info("创建关系: (%s)-[:%s]->(%s)", source_name, rel_type, target_name)

    # ------------------------------------------------------------------
    # 辅助节点操作
    # ------------------------------------------------------------------

    def record_document(
        self, doc_id: str, name: str, source: str, doc_format: str
    ) -> None:
        """在图中记录文档节点（用于构建 FOUND_IN 关系）。

        Args:
            doc_id: 文档 ID。
            name: 文档名。
            source: 文档来源路径或 URL。
            doc_format: 格式（pdf / web / docx 等）。
        """
        self.execute_write(
            """
            MERGE (d:Document {doc_id: $doc_id})
            SET d.name = $name, d.source = $source, d.format = $format
            """,
            {"doc_id": doc_id, "name": name, "source": source, "format": doc_format},
        )

    def link_concept_to_document(
        self, concept_name: str, doc_id: str, location_ref: str = ""
    ) -> None:
        """将概念链接到文档（FOUND_IN 关系）。

        Args:
            concept_name: 概念名称。
            doc_id: 文档 ID。
            location_ref: 位置引用（如 "第3页"、"段落: 架构概述"）。
        """
        self.execute_write(
            """
            MATCH (c:Concept {name: $concept})
            MATCH (d:Document {doc_id: $doc_id})
            MERGE (c)-[r:FOUND_IN {location_ref: $location_ref}]->(d)
            """,
            {"concept": concept_name, "doc_id": doc_id, "location_ref": location_ref},
        )

    # ------------------------------------------------------------------
    # 图谱查询
    # ------------------------------------------------------------------

    def get_neighbourhood(
        self, concept_name: str, depth: int = 1
    ) -> list[dict[str, Any]]:
        """获取概念的邻域子图。

        Args:
            concept_name: 中心概念名称。
            depth: 遍历深度。

        Returns:
            包含节点和关系的字典列表。
        """
        results = self.run_query(
            """
            MATCH path = (c:Concept {name: $name})-[*1..$depth]-(related)
            RETURN path LIMIT 100
            """,
            {"name": concept_name, "depth": depth},
        )
        return results

    def get_all_concepts(self, limit: int = 100, user_id: str = "") -> list[dict[str, Any]]:
        """获取所有概念节点（按频率降序）。

        Args:
            limit: 返回结果上限。
            user_id: 按用户过滤，空则不过滤。

        Returns:
            概念列表。
        """
        if user_id:
            results = self.run_query(
                """
                MATCH (c:Concept)
                WHERE c.user_id = $user_id
                RETURN c { .name, .type, .description, .frequency } AS concept
                ORDER BY c.frequency DESC
                LIMIT $limit
                """,
                {"limit": limit, "user_id": user_id},
            )
        else:
            results = self.run_query(
                """
                MATCH (c:Concept)
                RETURN c { .name, .type, .description, .frequency } AS concept
                ORDER BY c.frequency DESC
                LIMIT $limit
                """,
                {"limit": limit},
            )
        return [r["concept"] for r in results]

    def get_user_graph_data(
        self, user_id: str
    ) -> dict[str, Any]:
        """获取用户的知识图谱数据（节点 + 关系），供可视化使用。

        返回当前用户在 Neo4j 中的所有 Concept 节点及 RELATES_TO 关系，
        格式为 {nodes: [...], edges: [...]}，可直接供 pyvis 等库消费。

        Args:
            user_id: 用户 ID。

        Returns:
            字典，包含 nodes 列表和 edges 列表。
            - nodes: [{name, type, frequency, description}, ...]
            - edges: [{source, target, strength, description}, ...]
        """
        # 查询该用户的所有 Concept 节点
        node_results = self.run_query(
            """
            MATCH (c:Concept {user_id: $user_id})
            RETURN c.name AS name, c.type AS type,
                   c.frequency AS frequency, c.description AS description
            ORDER BY c.frequency DESC
            """,
            {"user_id": user_id},
        )
        nodes = [
            {
                "name": r["name"],
                "type": r.get("type", "concept"),
                "frequency": int(r.get("frequency", 0)),
                "description": r.get("description", ""),
            }
            for r in node_results
        ]

        # 查询该用户概念之间的 RELATES_TO 关系（去重：仅保留单向 a < b）
        edge_results = self.run_query(
            """
            MATCH (a:Concept {user_id: $user_id})-[r:RELATES_TO]->(b:Concept {user_id: $user_id})
            WHERE a.name < b.name
            RETURN a.name AS source, b.name AS target,
                   r.strength AS strength, r.description AS description
            """,
            {"user_id": user_id},
        )
        edges = [
            {
                "source": r["source"],
                "target": r["target"],
                "strength": float(r.get("strength", 0.5)),
                "description": r.get("description", ""),
            }
            for r in edge_results
        ]

        return {"nodes": nodes, "edges": edges}

    def seed_demo_frequencies(self, user_id: str = "") -> int:
        """为概念节点分配模拟的随机频率（仅用于演示/开发）。

        将现有 Concept 节点的 frequency 更新为 1~15 之间的随机值，
        模拟多次交互积累的效果。仅影响 frequency=1 的节点（首次分配），
        避免覆盖真实累积数据（frequency > 1 的节点保持不变）。

        Args:
            user_id: 按用户过滤，空则更新所有用户的概念。

        Returns:
            更新的节点数量。
        """
        import random

        # 先查出所有需要更新的概念（frequency == 1 表示从未重复遇到）
        if user_id:
            results = self.run_query(
                """
                MATCH (c:Concept)
                WHERE c.user_id = $user_id AND (c.frequency IS NULL OR c.frequency <= 1)
                RETURN c.name AS name, c.user_id AS uid
                """,
                {"user_id": user_id},
            )
        else:
            results = self.run_query(
                """
                MATCH (c:Concept)
                WHERE c.frequency IS NULL OR c.frequency <= 1
                RETURN c.name AS name, c.user_id AS uid
                """
            )

        if not results:
            logger.info("seed_demo_frequencies: 无需更新的概念（所有概念 frequency > 1）")
            return 0

        updated = 0
        for r in results:
            name = r["name"]
            uid = r.get("uid", "default")
            # 随机频率：模拟 1~15 次遇到
            freq = random.choices(
                population=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
                weights=[5, 10, 15, 15, 15, 10, 8, 5, 3, 2, 1],
                k=1,
            )[0]
            self.execute_write(
                """
                MATCH (c:Concept {name: $name, user_id: $uid})
                SET c.frequency = $freq
                """,
                {"name": name, "uid": uid, "freq": freq},
            )
            updated += 1

        logger.info("seed_demo_frequencies: 已更新 %d 个概念的频率", updated)
        return updated

    def get_user_graph_data_top_n(
        self, user_id: str, top_n: int = 10
    ) -> dict[str, Any]:
        """获取用户的知识图谱数据（仅 Top-N 概念），供可视化使用。

        与 get_user_graph_data 类似，但仅返回频率最高的 top_n 个概念节点
        及其之间的 RELATES_TO 关系。

        Args:
            user_id: 用户 ID。
            top_n: 返回的概念节点数量上限。

        Returns:
            字典，包含 nodes 列表和 edges 列表。
        """
        # 查询该用户频率最高的 top_n 个 Concept 节点
        node_results = self.run_query(
            """
            MATCH (c:Concept {user_id: $user_id})
            RETURN c.name AS name, c.type AS type,
                   c.frequency AS frequency, c.description AS description
            ORDER BY c.frequency DESC
            LIMIT $top_n
            """,
            {"user_id": user_id, "top_n": top_n},
        )
        nodes = [
            {
                "name": r["name"],
                "type": r.get("type", "concept"),
                "frequency": int(r.get("frequency", 0)),
                "description": r.get("description", ""),
            }
            for r in node_results
        ]

        # 查询这些 top_n 概念之间的 RELATES_TO 关系
        top_names = [n["name"] for n in nodes]
        edges: list[dict[str, Any]] = []
        if len(top_names) >= 2:
            edge_results = self.run_query(
                """
                MATCH (a:Concept {user_id: $user_id})-[r:RELATES_TO]->(b:Concept {user_id: $user_id})
                WHERE a.name IN $names AND b.name IN $names AND a.name < b.name
                RETURN a.name AS source, b.name AS target,
                       r.strength AS strength, r.description AS description
                """,
                {"user_id": user_id, "names": top_names},
            )
            edges = [
                {
                    "source": r["source"],
                    "target": r["target"],
                    "strength": float(r.get("strength", 0.5)),
                    "description": r.get("description", ""),
                }
                for r in edge_results
            ]

        return {"nodes": nodes, "edges": edges}
