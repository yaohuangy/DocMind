"""
SQLite 元数据存储模块。

管理结构化元数据：文档列表和会话统计。
提供 documents 表和 sessions 表的完整 CRUD 操作。

表结构（见 spec §10.4）：

- ``documents``: doc_id, name, source, format, num_chunks, num_pages, char_count, loaded_at
- ``sessions``: session_id, user_id, start_time, end_time, num_questions, num_notes, num_documents
"""

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config import SQLiteConfig, get_config

logger = logging.getLogger(__name__)


class MetadataStore:
    """SQLite 元数据存储。

    线程安全（每线程独立连接），支持上下文管理器协议。

    Usage::

        store = MetadataStore()
        store.ensure_tables()
        store.add_document("doc_001", "paper.pdf", "/path/to/paper.pdf", "pdf", 42, 10, 3200)
        docs = store.list_documents()
        store.close()
    """

    def __init__(self, config: Optional[SQLiteConfig] = None):
        """
        Args:
            config: SQLite 配置。为 None 时自动从全局配置加载。
        """
        if config is None:
            config = get_config().sqlite
        self._db_path = config.path
        # 确保父目录存在
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        # Thread-local connection
        self._local = threading.local()

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    @property
    def _conn(self) -> sqlite3.Connection:
        """获取当前线程的数据库连接（延迟创建）。"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
            )
            self._local.conn.row_factory = sqlite3.Row
            # 启用 WAL 模式以支持并发读
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def close(self) -> None:
        """关闭当前线程的数据库连接。"""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Schema 初始化
    # ------------------------------------------------------------------

    def ensure_tables(self) -> None:
        """创建所有元数据表（幂等，IF NOT EXISTS）。"""
        conn = self._conn

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                format TEXT NOT NULL,
                user_id TEXT NOT NULL DEFAULT 'default',
                num_chunks INTEGER DEFAULT 0,
                num_pages INTEGER DEFAULT 0,
                char_count INTEGER DEFAULT 0,
                loaded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                num_questions INTEGER DEFAULT 0,
                num_notes INTEGER DEFAULT 0,
                num_documents INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer_preview TEXT NOT NULL,
                method TEXT NOT NULL,
                rating TEXT NOT NULL,
                latency_sec REAL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_history (
                user_id TEXT NOT NULL,
                messages_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id)
            );
        """)
        conn.commit()

        # 迁移：为旧数据库添加 user_id 列（如果不存在）
        try:
            conn.execute("ALTER TABLE documents ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'")
            conn.commit()
            logger.info("已为 documents 表添加 user_id 列")
        except Exception:
            pass  # 列已存在，忽略

        logger.info("SQLite 元数据表已就绪: %s", self._db_path)

    # ------------------------------------------------------------------
    # Documents CRUD
    # ------------------------------------------------------------------

    def add_document(
        self,
        doc_id: str,
        name: str,
        source: str,
        doc_format: str,
        user_id: str = "default",
        num_chunks: int = 0,
        num_pages: int = 0,
        char_count: int = 0,
        loaded_at: str = "",
    ) -> None:
        """添加/更新文档记录。

        Args:
            doc_id: 文档唯一 ID。
            name: 文档名称。
            source: 文档来源。
            doc_format: 格式。
            user_id: 所属用户。
            num_chunks: 分块数。
            num_pages: 页数。
            char_count: 总字符数。
            loaded_at: 加载时间。
        """
        from datetime import datetime

        loaded_at = loaded_at or datetime.now().isoformat()

        self._conn.execute(
            """
            INSERT INTO documents (doc_id, name, source, format, user_id, num_chunks, num_pages, char_count, loaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                num_chunks = excluded.num_chunks,
                num_pages = excluded.num_pages,
                char_count = excluded.char_count,
                loaded_at = excluded.loaded_at
            """,
            (doc_id, name, source, doc_format, user_id, num_chunks, num_pages, char_count, loaded_at),
        )
        self._conn.commit()
        logger.info("文档已记录: %s (%s, %d chunks, user=%s)", name, doc_format, num_chunks, user_id)

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """按 ID 查询单个文档。

        Args:
            doc_id: 文档 ID。

        Returns:
            文档字典，不存在时返回 None。
        """
        row = self._conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_documents(
        self,
        doc_format: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """列出文档，可选按格式和用户过滤。

        Args:
            doc_format: 按格式过滤。
            user_id: 按用户过滤。

        Returns:
            文档字典列表，按加载时间降序排列。
        """
        query = "SELECT * FROM documents WHERE 1=1"
        params: List[Any] = []

        if doc_format:
            query += " AND format = ?"
            params.append(doc_format)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        query += " ORDER BY loaded_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: str) -> bool:
        """删除文档记录。

        Args:
            doc_id: 文档 ID。

        Returns:
            是否实际删除了记录。
        """
        cursor = self._conn.execute(
            "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
        )
        self._conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("文档已删除: %s", doc_id)
        return deleted

    def get_document_count(self, doc_format: Optional[str] = None, user_id: Optional[str] = None) -> int:
        """获取文档总数，可选按格式和用户过滤。

        Args:
            doc_format: 格式过滤。
            user_id: 用户过滤。

        Returns:
            文档数量。
        """
        query = "SELECT COUNT(*) as cnt FROM documents WHERE 1=1"
        params: List[Any] = []
        if doc_format:
            query += " AND format = ?"
            params.append(doc_format)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        row = self._conn.execute(query, params).fetchone()
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Sessions CRUD
    # ------------------------------------------------------------------

    def start_session(
        self,
        session_id: str,
        user_id: str = "default_user",
        start_time: str = "",
    ) -> None:
        """开始新会话。

        Args:
            session_id: 会话唯一 ID。
            user_id: 用户 ID。
            start_time: 开始时间 ISO 字符串。
        """
        from datetime import datetime

        start_time = start_time or datetime.now().isoformat()

        self._conn.execute(
            """
            INSERT INTO sessions (session_id, user_id, start_time)
            VALUES (?, ?, ?)
            """,
            (session_id, user_id, start_time),
        )
        self._conn.commit()
        logger.info("会话开始: %s", session_id)

    def update_session(
        self,
        session_id: str,
        end_time: Optional[str] = None,
        num_questions: Optional[int] = None,
        num_notes: Optional[int] = None,
        num_documents: Optional[int] = None,
    ) -> None:
        """更新会话统计信息。

        Args:
            session_id: 会话 ID。
            end_time: 结束时间。
            num_questions: 提问次数。
            num_notes: 笔记数。
            num_documents: 文档数。
        """
        fields: List[str] = []
        values: List[Any] = []

        if end_time is not None:
            fields.append("end_time = ?")
            values.append(end_time)
        if num_questions is not None:
            fields.append("num_questions = ?")
            values.append(num_questions)
        if num_notes is not None:
            fields.append("num_notes = ?")
            values.append(num_notes)
        if num_documents is not None:
            fields.append("num_documents = ?")
            values.append(num_documents)

        if fields:
            values.append(session_id)
            query = f"UPDATE sessions SET {', '.join(fields)} WHERE session_id = ?"
            self._conn.execute(query, values)
            self._conn.commit()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """查询会话信息。

        Args:
            session_id: 会话 ID。

        Returns:
            会话字典，不存在时返回 None。
        """
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_sessions(
        self, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """列出最近的会话。

        Args:
            limit: 返回数量上限。

        Returns:
            会话字典列表，按开始时间降序。
        """
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session_count(self, user_id: Optional[str] = None) -> int:
        """获取会话总数。

        Args:
            user_id: 按用户过滤，None 则返回全部。

        Returns:
            会话数量。
        """
        if user_id:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM sessions WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM sessions"
            ).fetchone()
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # 反馈记录
    # ------------------------------------------------------------------

    def add_feedback(
        self,
        user_id: str,
        question: str,
        answer_preview: str,
        method: str,
        rating: str,
        latency_sec: float = 0.0,
    ) -> None:
        """记录一条用户反馈。

        Args:
            user_id: 用户名。
            question: 问题文本。
            answer_preview: 答案前 200 字。
            method: 检索方法。
            rating: 'useful' 或 'not_useful'。
            latency_sec: 检索延迟秒数。
        """
        from datetime import datetime
        created_at = datetime.now().isoformat()
        self._conn.execute(
            """
            INSERT INTO feedback (user_id, question, answer_preview, method, rating, latency_sec, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, question, answer_preview[:200], method, rating, latency_sec, created_at),
        )
        self._conn.commit()
        # 验证写入
        count = self._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        print(f"[FEEDBACK WRITE] user={user_id}, method={method}, rating={rating}, total_rows={count}")

    def get_feedback_stats(
        self,
        user_id: Optional[str] = None,
        method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """获取反馈统计数据。

        Args:
            user_id: 按用户过滤，None 返回全部。
            method: 按检索方法过滤。

        Returns:
            {
                "total": int,
                "useful": int,
                "not_useful": int,
                "satisfaction_rate": float,
                "by_method": {method: {"total": n, "useful": n, "rate": f}},
                "avg_latency": float,
                "recent": [dict, ...],
            }
        """
        query = "SELECT * FROM feedback WHERE 1=1"
        params: List[Any] = []
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if method:
            query += " AND method = ?"
            params.append(method)

        rows = self._conn.execute(query + " ORDER BY created_at DESC", params).fetchall()
        records = [dict(r) for r in rows]
        print(f"[FEEDBACK READ] user_id={user_id}, method={method}, rows={len(records)}")
        for r in records[:3]:
            print(f"  row: {r['user_id']} | {r['rating']} | {r['method']} | {r['created_at'][:19]}")

        total = len(records)
        useful = sum(1 for r in records if r["rating"] == "useful")
        not_useful = total - useful

        # 按方法分
        by_method: Dict[str, Dict[str, Any]] = {}
        for r in records:
            m = r["method"]
            if m not in by_method:
                by_method[m] = {"total": 0, "useful": 0, "latencies": []}
            by_method[m]["total"] += 1
            if r["rating"] == "useful":
                by_method[m]["useful"] += 1
            if r.get("latency_sec"):
                by_method[m]["latencies"].append(r["latency_sec"])

        for m in by_method:
            bm = by_method[m]
            bm["rate"] = bm["useful"] / bm["total"] if bm["total"] > 0 else 0.0
            lats = bm.pop("latencies")
            bm["avg_latency"] = sum(lats) / len(lats) if lats else 0.0

        all_latencies = [r["latency_sec"] for r in records if r.get("latency_sec")]
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0

        return {
            "total": total,
            "useful": useful,
            "not_useful": not_useful,
            "satisfaction_rate": useful / total if total > 0 else 0.0,
            "by_method": by_method,
            "avg_latency": avg_latency,
            "recent": records[:10],
        }

    # ------------------------------------------------------------------
    # 会话消息持久化
    # ------------------------------------------------------------------

    def save_conversation(self, user_id: str, messages: List[Dict[str, Any]]) -> None:
        """持久化当前会话的聊天记录。

        Args:
            user_id: 用户名。
            messages: 消息列表 [{"role": "user/assistant", "content": "...", ...}]
        """
        import json as _json
        from datetime import datetime
        messages_json = _json.dumps(messages, ensure_ascii=False, default=str)
        updated_at = datetime.now().isoformat()
        self._conn.execute(
            """
            INSERT INTO conversation_history (user_id, messages_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                messages_json = excluded.messages_json,
                updated_at = excluded.updated_at
            """,
            (user_id, messages_json, updated_at),
        )
        self._conn.commit()

    def load_conversation(self, user_id: str) -> List[Dict[str, Any]]:
        """加载持久化的会话聊天记录。

        Args:
            user_id: 用户名。

        Returns:
            消息列表，不存在返回空列表。
        """
        import json as _json
        row = self._conn.execute(
            "SELECT messages_json FROM conversation_history WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row:
            try:
                return _json.loads(row["messages_json"])
            except Exception:
                return []
        return []

    def clear_conversation(self, user_id: str) -> None:
        """清除持久化的会话记录。

        Args:
            user_id: 用户名。
        """
        self._conn.execute(
            "DELETE FROM conversation_history WHERE user_id = ?", (user_id,)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # 统计聚合
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """获取全局统计快照。

        Returns:
            包含文档总数、会话总数、各格式文档数等。
        """
        doc_count = self.get_document_count()
        session_count = self.get_session_count()

        # 各格式计数
        format_rows = self._conn.execute(
            "SELECT format, COUNT(*) as cnt FROM documents GROUP BY format"
        ).fetchall()
        formats = {r["format"]: r["cnt"] for r in format_rows}

        return {
            "total_documents": doc_count,
            "total_sessions": session_count,
            "documents_by_format": formats,
        }
