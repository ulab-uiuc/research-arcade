import os
from typing import Optional, List, Tuple
import psycopg2
import psycopg2.extras
import pandas as pd  # only used for CSV import
import json

class SQLArxivPapers:
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
    def create_papers_table(self):
        """
        Creates the arxiv_papers table.
        arxiv_id is UNIQUE to mirror your CSV conflict rule.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_papers (
                    id SERIAL PRIMARY KEY,
                    arxiv_id VARCHAR(64) NOT NULL UNIQUE,
                    base_arxiv_id VARCHAR(64) NOT NULL,
                    version VARCHAR(32) NOT NULL,
                    title TEXT NOT NULL,
                    abstract TEXT,
                    submit_date TIMESTAMP NULL,
                    metadata JSONB
                )
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # CRUD
    # -------------------------
    def insert_paper(self, arxiv_id, base_arxiv_id, version, title,
                     abstract=None, submit_date=None, metadata=None) -> Optional[int]:
        """
        Insert a paper; returns generated id or None if arxiv_id already exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_papers
                  (arxiv_id, base_arxiv_id, version, title, abstract, submit_date, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (arxiv_id) DO NOTHING
                RETURNING id
                """,
                (
                    arxiv_id,
                    base_arxiv_id,
                    version,
                    title,
                    abstract,
                    submit_date,  # can be a string parsable by PG or a datetime
                    psycopg2.extras.Json(metadata) if metadata is not None else None
                )
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def delete_paper_by_arxiv_id(self, arxiv_id: str) -> bool:
        """
        Delete by arxiv_id; returns True if a row was deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM arxiv_papers WHERE arxiv_id = %s RETURNING id", (arxiv_id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def delete_paper_by_year(self, year: int) -> int:
        """
        Delete all papers whose submit_date falls in the given year.
        Returns number of rows deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM arxiv_papers
                WHERE submit_date IS NOT NULL
                  AND EXTRACT(YEAR FROM submit_date)::int = %s
                RETURNING id
                """,
                (year,)
            )
            rows = cur.fetchall()
            cur.close()
            return len(rows) if rows else 0
        finally:
            conn.close()

    def update_paper(self, arxiv_id: str, base_arxiv_id=None, version=None, title=None,
                     abstract=None, submit_date=None, metadata=None) -> bool:
        """
        Partial update by arxiv_id. Only non-None fields are updated.
        Returns True if a row was updated.
        """
        sets: List[str] = []
        vals: List = []

        if base_arxiv_id is not None:
            sets.append("base_arxiv_id = %s")
            vals.append(base_arxiv_id)
        if version is not None:
            sets.append("version = %s")
            vals.append(version)
        if title is not None:
            sets.append("title = %s")
            vals.append(title)
        if abstract is not None:
            sets.append("abstract = %s")
            vals.append(abstract)
        if submit_date is not None:
            sets.append("submit_date = %s")
            vals.append(submit_date)
        if metadata is not None:
            sets.append("metadata = %s")
            vals.append(psycopg2.extras.Json(metadata))

        if not sets:
            return False

        sql = f"UPDATE arxiv_papers SET {', '.join(sets)} WHERE arxiv_id = %s RETURNING id"
        vals.append(arxiv_id)

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(sql, tuple(vals))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def get_paper_by_arxiv_id(self, arxiv_id: str, return_all: bool = False):
        """
        If return_all=False: returns a single tuple
          (id, arxiv_id, base_arxiv_id, version, title, abstract, submit_date, metadata)
        If return_all=True: returns a list of such tuples.
        Returns None if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, arxiv_id, base_arxiv_id, version, title, abstract, submit_date, metadata "
                "FROM arxiv_papers WHERE arxiv_id = %s",
                (arxiv_id,)
            )
            rows = cur.fetchall() if return_all else cur.fetchone()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def check_paper_exists(self, arxiv_id: str) -> bool:
        """
        Returns True if a paper with the given arxiv_id exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM arxiv_papers WHERE arxiv_id = %s LIMIT 1", (arxiv_id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    # -------------------------
    # Bulk import from CSV
    # -------------------------
    def construct_papers_table_from_csv(self, csv_file: str) -> bool:
        """
        Imports rows from a CSV with required columns:
          ['arxiv_id', 'base_arxiv_id', 'version', 'title']
        Optional columns (auto-filled to NULL if missing):
          ['abstract', 'submit_date', 'metadata']
        Ignores any 'id' in the CSV; DB assigns SERIAL ids.
        Skips conflicts on arxiv_id (ON CONFLICT DO NOTHING).
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        df = pd.read_csv(csv_file)

        required = ['arxiv_id', 'base_arxiv_id', 'version', 'title']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Error: External CSV is missing required columns: {missing}")
            return False

        # Ensure optional columns exist
        for c in ['abstract', 'submit_date', 'metadata']:
            if c not in df.columns:
                df[c] = None

        # If metadata is present as strings/dicts, keep it as-is; psycopg2.Json handles serialization.
        rows: List[Tuple] = []
        for _, r in df.iterrows():
            meta_val = r['metadata']
            # If metadata column is a string that looks like JSON, try to keep it raw; psycopg2.Json can take dicts as well.
            if isinstance(meta_val, str):
                try:
                    # Best-effort parse to dict for proper JSONB storage
                    meta_val = json.loads(meta_val)
                except Exception:
                    # Leave as string; PG will accept it as text → JSONB cast may fail if not valid JSON.
                    # Safer approach: leave it None if invalid JSON strings are expected.
                    pass

            rows.append((
                r['arxiv_id'],
                r['base_arxiv_id'],
                r['version'],
                r['title'],
                r['abstract'],
                r['submit_date'],
                meta_val
            ))

        if not rows:
            print("No rows to import.")
            return True

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO arxiv_papers
                  (arxiv_id, base_arxiv_id, version, title, abstract, submit_date, metadata)
                VALUES %s
                ON CONFLICT (arxiv_id) DO NOTHING
                """,
                [
                    (
                        arxiv_id,
                        base_arxiv_id,
                        version,
                        title,
                        abstract,
                        submit_date,
                        psycopg2.extras.Json(metadata) if metadata is not None else None
                    )
                    for (arxiv_id, base_arxiv_id, version, title, abstract, submit_date, metadata)
                    in rows
                ],
                page_size=1000
            )
            cur.close()
        finally:
            conn.close()

        print(f"Successfully imported {len(rows)} papers from {csv_file}")
        return True
