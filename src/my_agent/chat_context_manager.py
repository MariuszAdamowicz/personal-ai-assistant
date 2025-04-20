import sqlite3
from typing import Callable, List, Dict, Optional


class ChatContextManager:
    def __init__(
        self,
        db_path: str = "chat_history.db",
        session_id: str = "default",
        max_history_messages: int = 20,
        summarize_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
        group_size_level_0: int = 5,
        group_size_higher_levels: int = 5,
        max_summary_level: int = 3
    ):
        self.db_path = db_path
        self.session_id = session_id
        self.max_history_messages = max_history_messages
        self.summarize_fn = summarize_fn
        self.group_size_level_0 = group_size_level_0
        self.group_size_higher_levels = group_size_higher_levels
        self.max_summary_level = max_summary_level
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                level INTEGER,
                start_msg_id INTEGER,
                end_msg_id INTEGER,
                summary_text TEXT
            )
        """)
        self.conn.commit()

    def add_message(self, role: str, content: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO messages (session_id, role, content)
            VALUES (?, ?, ?)
        """, (self.session_id, role, content))
        self.conn.commit()
        self._update_summaries()

    def _get_last_messages(self, n: int) -> List[Dict[str, str]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT role, content FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (self.session_id, n))
        rows = cursor.fetchall()
        return list(reversed([{"role": role, "content": content} for role, content in rows]))

    def _get_total_message_count(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM messages WHERE session_id = ?
        """, (self.session_id,))
        return cursor.fetchone()[0]

    def _update_summaries(self):
        if self.summarize_fn is None:
            raise ValueError("summarize_fn is not set")

        cursor = self.conn.cursor()
        total_message_count = self._get_total_message_count()
        max_msg_id_to_summarize = total_message_count - self.max_history_messages

        # Level 0 — message summarize
        cursor.execute("""
            SELECT MAX(end_msg_id) FROM summaries
            WHERE session_id = ? AND level = 0
        """, (self.session_id,))
        last_end = cursor.fetchone()[0] or 0

        cursor.execute("""
            SELECT id, role, content FROM messages
            WHERE session_id = ? AND id > ?
            ORDER BY id ASC
            LIMIT ?
        """, (self.session_id, last_end, self.group_size_level_0))
        group = cursor.fetchall()

        if len(group) == self.group_size_level_0 and group[-1][0] <= max_msg_id_to_summarize:
            msgs = [{"role": r, "content": c} for _, r, c in group]
            summary = self.summarize_fn(msgs)
            start_id = group[0][0]
            end_id = group[-1][0]
            cursor.execute("""
                INSERT INTO summaries (session_id, level, start_msg_id, end_msg_id, summary_text)
                VALUES (?, ?, ?, ?, ?)
            """, (self.session_id, 0, start_id, end_id, summary))
            self.conn.commit()

        # upper levels — summaries summary
        for level in range(1, self.max_summary_level + 1):
            cursor.execute("""
                SELECT MAX(end_msg_id) FROM summaries
                WHERE session_id = ? AND level = ?
            """, (self.session_id, level))
            last_end = cursor.fetchone()[0] or 0

            cursor.execute("""
                SELECT id, start_msg_id, end_msg_id, summary_text
                FROM summaries
                WHERE session_id = ? AND level = ? AND end_msg_id > ?
                ORDER BY start_msg_id ASC
                LIMIT ?
            """, (self.session_id, level - 1, last_end, self.group_size_higher_levels))
            group = cursor.fetchall()

            if len(group) == self.group_size_higher_levels:
                texts = [{"role": "system", "content": row[3]} for row in group]
                summary = self.summarize_fn(texts)
                start_id = group[0][1]
                end_id = group[-1][2]
                cursor.execute("""
                    INSERT INTO summaries (session_id, level, start_msg_id, end_msg_id, summary_text)
                    VALUES (?, ?, ?, ?, ?)
                """, (self.session_id, level, start_id, end_id, summary))
                self.conn.commit()

    def get_summary_until(self, msg_id: int) -> Optional[str]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT summary_text FROM summaries
            WHERE session_id = ? AND end_msg_id <= ?
            ORDER BY level DESC, end_msg_id DESC
            LIMIT 1
        """, (self.session_id, msg_id))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_context(self, system_prompt: str = None) -> List[Dict[str, str]]:
        total_messages = self._get_total_message_count()

        context = []
        if system_prompt:
            context.append({"role": "system", "content": system_prompt})

        if total_messages > self.max_history_messages:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            """, (self.session_id, self.max_history_messages))
            last_ids = [row[0] for row in cursor.fetchall()]
            first_to_exclude = min(last_ids)
            summary = self.get_summary_until(first_to_exclude - 1)
            if summary:
                context.append({"role": "system", "content": summary})

        context.extend(self._get_last_messages(self.max_history_messages))
        return context
