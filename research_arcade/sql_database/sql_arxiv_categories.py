import os
from typing import Optional, List, Tuple
import psycopg2
import psycopg2.extras
import pandas as pd  # only used for CSV import

class SQLArxivCategory:
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
            password=self.password,
        )
        conn.autocommit = self.autocommit
        return conn

    # -------------------------
    # DDL
    # -------------------------
    def create_categories_table(self):
        """
        Creates arxiv_categories with UNIQUE(name) to mirror CSV conflict rule.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_categories (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    description TEXT
                )
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # CRUD
    # -------------------------
    def insert_category(self, name: str, description: Optional[str] = None) -> Optional[int]:
        """
        Insert a category; returns generated id or None if name conflicts.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_categories (name, description)
                VALUES (%s, %s)
                ON CONFLICT (name) DO NOTHING
                RETURNING id
                """,
                (name, description)
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def delete_category_by_id(self, id: int) -> bool:
        """
        Delete by id; returns True if a row was deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM arxiv_categories WHERE id = %s RETURNING id", (id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def update_category(self, id: int, name: Optional[str] = None, description: Optional[str] = None) -> bool:
        """
        Partial update by id. Only non-None fields are updated.
        Returns True if a row was updated.
        """
        sets: List[str] = []
        vals: List = []
        if name is not None:
            sets.append("name = %s")
            vals.append(name)
        if description is not None:
            sets.append("description = %s")
            vals.append(description)

        if not sets:
            return False

        sql = f"UPDATE arxiv_categories SET {', '.join(sets)} WHERE id = %s RETURNING id"
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

    def get_category_by_id(self, id: int, return_all: bool = False):
        """
        If return_all=False: returns a single tuple (id, name, description)
        If return_all=True: returns a list of such tuples.
        Returns None if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, name, description FROM arxiv_categories WHERE id = %s",
                (id,)
            )
            rows = cur.fetchall() if return_all else cur.fetchone()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def check_category_exists(self, id: int) -> bool:
        """
        True if a category with the given id exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM arxiv_categories WHERE id = %s LIMIT 1", (id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    # -------------------------
    # Bulk import from CSV
    # -------------------------
    def construct_category_table_from_csv(self, csv_file: str) -> bool:
        """
        Imports rows from a CSV with required column: ['name']
        Optional column: ['description']
        Uses ON CONFLICT (name) DO NOTHING to skip duplicates.
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        df = pd.read_csv(csv_file)
        required_cols = ['name']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Error: External CSV is missing required columns: {missing}")
            return False

        if 'description' not in df.columns:
            df['description'] = None

        rows: List[Tuple] = list(df[['name', 'description']].itertuples(index=False, name=None))
        if not rows:
            print("No rows to import.")
            return True

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO arxiv_categories (name, description)
                VALUES %s
                ON CONFLICT (name) DO NOTHING
                """,
                rows,
                page_size=1000
            )
            cur.close()
        finally:
            conn.close()

        print(f"Successfully imported {len(rows)} categories from {csv_file}")
        return True
