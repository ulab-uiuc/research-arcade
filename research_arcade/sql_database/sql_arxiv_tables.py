import os
import csv
from typing import Optional, List, Tuple
import psycopg2
import psycopg2.extras
import pandas as pd  # only used inside construct helper; feel free to swap to csv.DictReader if you prefer

class SQLArxivTable:
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
            password=self.password
        )
        conn.autocommit = self.autocommit
        return conn

    # -------------------------
    # DDL
    # -------------------------
    def create_tables_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_tables (
                    id SERIAL PRIMARY KEY,
                    paper_arxiv_id VARCHAR(100) NOT NULL,
                    path VARCHAR(1024),
                    caption TEXT,
                    label VARCHAR(255),
                    table_text TEXT
                )
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # CRUD
    # -------------------------
    def insert_table(self, paper_arxiv_id, path=None, caption=None, label=None, table_text=None) -> Optional[int]:
        """
        Insert a table row; returns generated id (int) or None if not inserted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_tables (paper_arxiv_id, path, caption, label, table_text)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (paper_arxiv_id, path, caption, label, table_text)
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def delete_table_by_id(self, id: int) -> bool:
        """
        Delete by id; returns True if a row was deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM arxiv_tables WHERE id = %s RETURNING id", (id,))
            deleted = cur.fetchone() is not None
            cur.close()
            return deleted
        finally:
            conn.close()

    def update_table(self, id: int, paper_arxiv_id=None, path=None, caption=None, label=None, table_text=None) -> bool:
        """
        Partial update. Only non-None fields are updated.
        Returns True if a row was updated.
        """
        sets = []
        vals = []
        if paper_arxiv_id is not None:
            sets.append("paper_arxiv_id = %s")
            vals.append(paper_arxiv_id)
        if path is not None:
            sets.append("path = %s")
            vals.append(path)
        if caption is not None:
            sets.append("caption = %s")
            vals.append(caption)
        if label is not None:
            sets.append("label = %s")
            vals.append(label)
        if table_text is not None:
            sets.append("table_text = %s")
            vals.append(table_text)

        if not sets:
            return False  # nothing to update

        sql = f"UPDATE arxiv_tables SET {', '.join(sets)} WHERE id = %s RETURNING id"
        vals.append(id)

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(sql, tuple(vals))
            updated = cur.fetchone() is not None
            cur.close()
            return updated
        finally:
            conn.close()

    def get_table_by_id(self, id: int, return_all: bool = False):
        """
        If return_all=False: returns a single tuple
           (id, paper_arxiv_id, path, caption, label, table_text)
        If return_all=True: returns a list of such tuples.
        Returns None if no row found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, paper_arxiv_id, path, caption, label, table_text FROM arxiv_tables WHERE id = %s",
                (id,)
            )
            rows = cur.fetchall() if return_all else cur.fetchone()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def check_table_exists(self, id: int) -> bool:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM arxiv_tables WHERE id = %s LIMIT 1", (id,))
            exists = cur.fetchone() is not None
            cur.close()
            return exists
        finally:
            conn.close()

    # -------------------------
    # Bulk import from CSV
    # -------------------------
    def construct_tables_table_from_csv(self, csv_file: str) -> bool:
        """
        Imports rows from a CSV with columns:
          ['paper_arxiv_id', 'path', 'caption', 'label', 'table_text']
        Ignores any 'id' from the CSV and lets DB assign SERIAL ids.
        Returns True on success, False on validation error or missing file.
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        df = pd.read_csv(csv_file)
        required_cols = ['paper_arxiv_id', 'path', 'caption', 'label', 'table_text']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Error: External CSV is missing required columns: {missing}")
            return False

        rows: List[Tuple] = list(
            df[required_cols].itertuples(index=False, name=None)
        )
        if not rows:
            print("No rows to import.")
            return True

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO arxiv_tables (paper_arxiv_id, path, caption, label, table_text)
                VALUES %s
                """,
                rows,
                page_size=1000
            )
            cur.close()
        finally:
            conn.close()

        print(f"Successfully imported {len(rows)} tables from {csv_file}")
        return True
