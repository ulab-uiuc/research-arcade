import os
from typing import Optional, List, Tuple
import psycopg2
import psycopg2.extras
import pandas as pd  # only used in CSV import helper

class SQLArxivParagraphs:
    def __init__(self, sql_args):
        """
        sql_args should provide: host, port, dbname, user, password, autocommit (bool)
        """
        self.host, self.port, self.dbname, self.user, self.password, self.autocommit = (
            sql_args.host,
            sql_args.port,
            sql_args.dbname,
            sql_args.user,
            sql_args.password,
            sql_args.autocommit,
        )

    def _get_connection(self):
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
        )
        conn.autocommit = self.autocommit
        return conn

    # -------------------------
    # DDL
    # -------------------------
    def create_paragraphs_table(self):
        """
        Creates table with a composite uniqueness on (paragraph_id, paper_arxiv_id, paper_section)
        to mirror the CSV conflict check.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_paragraphs (
                    id SERIAL PRIMARY KEY,
                    paragraph_id VARCHAR(255) NOT NULL,
                    content TEXT,
                    paper_arxiv_id VARCHAR(100) NOT NULL,
                    paper_section VARCHAR(255) NOT NULL
                )
            """)
            # Composite unique index for conflict prevention
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS ux_arxiv_paragraphs_unique
                ON arxiv_paragraphs (paragraph_id, paper_arxiv_id, paper_section)
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # CRUD
    # -------------------------
    def insert_paragraph(self, paragraph_id, content, paper_arxiv_id, paper_section) -> Optional[int]:
        """
        Insert a paragraph. Returns generated id or None if conflicts with the composite unique constraint.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_paragraphs (paragraph_id, content, paper_arxiv_id, paper_section)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (paragraph_id, paper_arxiv_id, paper_section) DO NOTHING
                RETURNING id
                """,
                (paragraph_id, content, paper_arxiv_id, paper_section)
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def delete_paragraph_by_id(self, id: int) -> bool:
        """
        Delete a paragraph by id. Returns True if a row was deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM arxiv_paragraphs WHERE id = %s RETURNING id", (id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def delete_paragraph_by_paper_arxiv_id(self, paper_arxiv_id: str) -> int:
        """
        Delete all paragraphs for a given paper_arxiv_id. Returns number of rows deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paragraphs WHERE paper_arxiv_id = %s RETURNING id",
                (paper_arxiv_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return len(rows) if rows else 0
        finally:
            conn.close()

    def delete_paragraph_by_paper_section(self, paper_arxiv_id: str, paper_section: str) -> int:
        """
        Delete all paragraphs for a given (paper_arxiv_id, paper_section).
        Returns number of rows deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM arxiv_paragraphs
                WHERE paper_arxiv_id = %s AND paper_section = %s
                RETURNING id
                """,
                (paper_arxiv_id, paper_section)
            )
            rows = cur.fetchall()
            cur.close()
            return len(rows) if rows else 0
        finally:
            conn.close()

    def update_paragraph(self, id: int, paragraph_id=None, content=None, paper_arxiv_id=None, paper_section=None) -> bool:
        """
        Partial update by id. Only non-None fields are updated.
        Returns True if a row was updated.
        """
        sets: List[str] = []
        vals: List = []

        if paragraph_id is not None:
            sets.append("paragraph_id = %s")
            vals.append(paragraph_id)
        if content is not None:
            sets.append("content = %s")
            vals.append(content)
        if paper_arxiv_id is not None:
            sets.append("paper_arxiv_id = %s")
            vals.append(paper_arxiv_id)
        if paper_section is not None:
            sets.append("paper_section = %s")
            vals.append(paper_section)

        if not sets:
            return False

        sql = f"UPDATE arxiv_paragraphs SET {', '.join(sets)} WHERE id = %s RETURNING id"
        vals.append(id)

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(sql, tuple(vals))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def get_paragraph_by_id(self, id: int, return_all: bool = False):
        """
        If return_all=False: returns a single tuple
           (id, paragraph_id, content, paper_arxiv_id, paper_section)
        If return_all=True: returns a list of such tuples.
        Returns None if no row found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, paragraph_id, content, paper_arxiv_id, paper_section "
                "FROM arxiv_paragraphs WHERE id = %s",
                (id,)
            )
            rows = cur.fetchall() if return_all else cur.fetchone()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_paragraphs_by_arxiv_id(self, arxiv_id: str):
        """
        Returns a list of tuples
          (id, paragraph_id, content, paper_arxiv_id, paper_section)
        or None if no rows exist.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, paragraph_id, content, paper_arxiv_id, paper_section "
                "FROM arxiv_paragraphs WHERE paper_arxiv_id = %s ORDER BY id",
                (arxiv_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_paragraphs_by_paper_section(self, paper_arxiv_id: str, paper_section: str):
        """
        Returns a list of tuples
          (id, paragraph_id, content, paper_arxiv_id, paper_section)
        filtered by (paper_arxiv_id, paper_section), or None if no rows.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, paragraph_id, content, paper_arxiv_id, paper_section "
                "FROM arxiv_paragraphs WHERE paper_arxiv_id = %s AND paper_section = %s "
                "ORDER BY id",
                (paper_arxiv_id, paper_section)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def check_paragraph_exists(self, id: int) -> bool:
        """
        Returns True if a paragraph with the given id exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM arxiv_paragraphs WHERE id = %s LIMIT 1", (id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    # -------------------------
    # Bulk import from CSV
    # -------------------------
    def construct_paragraph_table_from_csv(self, csv_file: str) -> bool:
        """
        Imports rows from a CSV with columns:
            ['paragraph_id', 'content', 'paper_arxiv_id', 'paper_section']
        Ignores any 'id' column; DB assigns SERIAL ids.
        Skips rows that violate the composite uniqueness (via ON CONFLICT DO NOTHING).
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        df = pd.read_csv(csv_file)
        required_cols = ['paragraph_id', 'content', 'paper_arxiv_id', 'paper_section']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Error: External CSV is missing required columns: {missing}")
            return False

        rows: List[Tuple] = list(df[required_cols].itertuples(index=False, name=None))
        if not rows:
            print("No rows to import.")
            return True

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            # Use execute_values with ON CONFLICT to bulk-insert
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO arxiv_paragraphs (paragraph_id, content, paper_arxiv_id, paper_section)
                VALUES %s
                ON CONFLICT (paragraph_id, paper_arxiv_id, paper_section) DO NOTHING
                """,
                rows,
                page_size=1000
            )
            cur.close()
        finally:
            conn.close()

        print(f"Successfully imported {len(rows)} paragraphs from {csv_file}")
        return True
