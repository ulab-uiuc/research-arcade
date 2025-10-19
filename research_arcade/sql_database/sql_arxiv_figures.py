import os
from typing import Optional, List, Tuple
import psycopg2
import psycopg2.extras
import pandas as pd  # only used for CSV import

class SQLArxivFigure:
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
    def create_figures_table(self):
        """
        Creates the arxiv_figures table.
        We add a partial UNIQUE index on name (WHERE name IS NOT NULL) to match the CSV
        behavior where we disallow duplicate non-null names.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_figures (
                    id SERIAL PRIMARY KEY,
                    paper_arxiv_id VARCHAR(100) NOT NULL,
                    path VARCHAR(1024),
                    caption TEXT,
                    label VARCHAR(255),
                    name VARCHAR(255)
                )
            """)
            # Partial unique: only enforce uniqueness when name IS NOT NULL
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS ux_arxiv_figures_name_notnull
                ON arxiv_figures (name)
                WHERE name IS NOT NULL
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # CRUD
    # -------------------------
    def insert_figure(self, paper_arxiv_id, path, caption=None, label=None, name=None) -> Optional[int]:
        """
        Insert a figure row; returns generated id or None on (name) conflict (when name is not NULL).
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_figures (paper_arxiv_id, path, caption, label, name)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT ON CONSTRAINT ux_arxiv_figures_name_notnull DO NOTHING
                RETURNING id
                """,
                (paper_arxiv_id, path, caption, label, name)
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def delete_figure_by_id(self, id: int) -> bool:
        """
        Delete by id; returns True if a row was deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM arxiv_figures WHERE id = %s RETURNING id", (id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def update_figure(self, id: int, paper_arxiv_id=None, path=None, caption=None, label=None, name=None) -> bool:
        """
        Partial update by id. Only non-None fields are updated.
        Returns True if a row was updated.
        """
        sets: List[str] = []
        vals: List = []

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
        if name is not None:
            sets.append("name = %s")
            vals.append(name)

        if not sets:
            return False

        sql = f"UPDATE arxiv_figures SET {', '.join(sets)} WHERE id = %s RETURNING id"
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

    def get_figure_by_id(self, id: int, return_all: bool = False):
        """
        If return_all=False: returns a single tuple
           (id, paper_arxiv_id, path, caption, label, name)
        If return_all=True: returns a list of such tuples.
        Returns None if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, paper_arxiv_id, path, caption, label, name "
                "FROM arxiv_figures WHERE id = %s",
                (id,)
            )
            rows = cur.fetchall() if return_all else cur.fetchone()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def check_figure_exists(self, id: int) -> bool:
        """
        Returns True if a figure with the given id exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM arxiv_figures WHERE id = %s LIMIT 1", (id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    # -------------------------
    # Bulk import from CSV
    # -------------------------
    def construct_figure_table_from_csv(self, csv_file: str) -> bool:
        """
        Imports rows from a CSV with columns:
          ['paper_arxiv_id', 'path', 'caption', 'label', 'name']
        Ignores any 'id' column; DB assigns SERIAL ids.
        Skips conflicts on non-null 'name'.
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        df = pd.read_csv(csv_file)
        required_cols = ['paper_arxiv_id', 'path', 'caption', 'label', 'name']
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
            # Use upsert-like bulk insert; conflict only triggers if name IS NOT NULL and duplicates.
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO arxiv_figures (paper_arxiv_id, path, caption, label, name)
                VALUES %s
                ON CONFLICT ON CONSTRAINT ux_arxiv_figures_name_notnull DO NOTHING
                """,
                rows,
                page_size=1000
            )
            cur.close()
        finally:
            conn.close()

        print(f"Successfully imported {len(rows)} figures from {csv_file}")
        return True
