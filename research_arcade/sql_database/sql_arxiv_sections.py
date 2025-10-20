import os
from typing import Optional, List, Tuple
import psycopg2
import psycopg2.extras
import pandas as pd  # used only in CSV import helper

class SQLArxivSections:
    def __init__(self, host: str, dbname: str, user: str, password: str, port: str):
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.autocommit = port
        self.autocommit = True

    def _get_connection(self):
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password
        )
        conn.autocommit = self.autocommit
        return conn

    # -------------------------
    # DDL
    # -------------------------
    def create_sections_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_sections (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    title VARCHAR(512),
                    appendix BOOLEAN,
                    paper_arxiv_id VARCHAR(100) NOT NULL
                )
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # CRUD
    # -------------------------
    def insert_section(self, content, title, is_appendix, paper_arxiv_id) -> Optional[int]:
        """
        Insert a section; returns generated id or None.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_sections (content, title, appendix, paper_arxiv_id)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (content, title, is_appendix, paper_arxiv_id)
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def delete_section_by_id(self, id: int) -> bool:
        """
        Delete by id; returns True if a row was deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM arxiv_sections WHERE id = %s RETURNING id", (id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def delete_section_by_paper_arxiv_id(self, paper_arxiv_id: str) -> int:
        """
        Delete all sections for a paper; returns number of deleted rows.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM arxiv_sections WHERE paper_arxiv_id = %s RETURNING id", (paper_arxiv_id,))
            rows = cur.fetchall()
            cur.close()
            return len(rows) if rows else 0
        finally:
            conn.close()

    def update_section(self, id: int, content=None, title=None, is_appendix=None, paper_arxiv_id=None) -> bool:
        """
        Partial update; only non-None fields are updated.
        Returns True if a row was updated.
        """
        sets = []
        vals: List = []
        if content is not None:
            sets.append("content = %s")
            vals.append(content)
        if title is not None:
            sets.append("title = %s")
            vals.append(title)
        if is_appendix is not None:
            sets.append("appendix = %s")
            vals.append(is_appendix)
        if paper_arxiv_id is not None:
            sets.append("paper_arxiv_id = %s")
            vals.append(paper_arxiv_id)

        if not sets:
            return False

        sql = f"UPDATE arxiv_sections SET {', '.join(sets)} WHERE id = %s RETURNING id"
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

    def get_section_by_id(self, id: int, return_all: bool = False):
        """
        If return_all=False: returns a single tuple
           (id, content, title, appendix, paper_arxiv_id)
        If return_all=True: returns a list of such tuples.
        Returns None if no row found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, content, title, appendix, paper_arxiv_id FROM arxiv_sections WHERE id = %s",
                (id,)
            )
            rows = cur.fetchall() if return_all else cur.fetchone()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_sections_by_arxiv_id(self, arxiv_id: str):
        """
        Returns a list of tuples:
          (id, content, title, appendix, paper_arxiv_id)
        or None if no rows.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, content, title, appendix, paper_arxiv_id "
                "FROM arxiv_sections WHERE paper_arxiv_id = %s ORDER BY id",
                (arxiv_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def check_section_exists(self, id: int) -> bool:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM arxiv_sections WHERE id = %s LIMIT 1", (id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    # -------------------------
    # Bulk import from CSV
    # -------------------------
    def construct_sections_table_from_csv(self, csv_file: str) -> bool:
        """
        Imports rows from a CSV with columns:
          ['content', 'title', 'appendix', 'paper_arxiv_id']
        Ignores any 'id' in the CSV; DB assigns SERIAL ids.
        Returns True on success, False on validation error or missing file.
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        df = pd.read_csv(csv_file)
        required_cols = ['content', 'title', 'appendix', 'paper_arxiv_id']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Error: External CSV is missing required columns: {missing}")
            return False

        # Normalize boolean-like 'appendix' column if needed (optional)
        if df['appendix'].dtype == object:
            true_vals = {'true', '1', 't', 'yes', 'y'}
            df['appendix'] = df['appendix'].astype(str).str.lower().isin(true_vals)

        rows: List[Tuple] = list(df[required_cols].itertuples(index=False, name=None))
        if not rows:
            print("No rows to import.")
            return True

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO arxiv_sections (content, title, appendix, paper_arxiv_id)
                VALUES %s
                """,
                rows,
                page_size=1000
            )
            cur.close()
        finally:
            conn.close()

        print(f"Successfully imported {len(rows)} sections from {csv_file}")
        return True
