import os
from typing import Optional, List, Tuple
import psycopg2
import psycopg2.extras
import pandas as pd
import json


class SQLArxivCitation:
    def __init__(self, host: str, dbname: str, user: str, password: str, port: str):
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.port = port
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
    def create_citations_table(self):
        """
        Creates arxiv_citations table with UNIQUE constraint on (citing_arxiv_id, cited_arxiv_id).
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_citations (
                    id SERIAL PRIMARY KEY,
                    citing_arxiv_id VARCHAR(255) NOT NULL,
                    cited_arxiv_id VARCHAR(255) NOT NULL,
                    bib_title TEXT,
                    bib_key VARCHAR(255),
                    author_cited_paper TEXT,
                    citing_sections JSONB DEFAULT '[]'::jsonb,
                    citing_paragraphs JSONB DEFAULT '[]'::jsonb,
                    UNIQUE(citing_arxiv_id, cited_arxiv_id)
                )
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # CRUD
    # -------------------------
    def insert_citation(
        self,
        citing_arxiv_id: str,
        cited_arxiv_id: str,
        bib_title: Optional[str] = None,
        bib_key: Optional[str] = None,
        author_cited_paper: Optional[str] = None,
        citing_sections: Optional[List] = None
    ) -> Optional[int]:
        """
        Insert a citation; returns generated id or None if:
        - citing_arxiv_id == cited_arxiv_id (self-citation)
        - conflict on (citing_arxiv_id, cited_arxiv_id)
        """
        if citing_arxiv_id == cited_arxiv_id:
            return None

        citing_sections_json = json.dumps(citing_sections) if citing_sections else '[]'
        
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_citations 
                    (citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, 
                     author_cited_paper, citing_sections, citing_paragraphs)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, '[]'::jsonb)
                ON CONFLICT (citing_arxiv_id, cited_arxiv_id) DO NOTHING
                RETURNING id
                """,
                (citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, 
                 author_cited_paper, citing_sections_json)
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def delete_citation_by_id(self, id: int) -> bool:
        """
        Delete by id; returns True if a row was deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM arxiv_citations WHERE id = %s RETURNING id", (id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()
    
    def update_citation(
        self,
        id: int,
        citing_arxiv_id: Optional[str] = None,
        cited_arxiv_id: Optional[str] = None,
        bib_title: Optional[str] = None,
        bib_key: Optional[str] = None,
        author_cited_paper: Optional[str] = None,
        citing_sections: Optional[List] = None,
        citing_paragraphs: Optional[List] = None
    ) -> bool:
        """
        Partial update by id. Only non-None fields are updated.
        Returns True if a row was updated.
        """
        sets: List[str] = []
        vals: List = []
        
        if citing_arxiv_id is not None:
            sets.append("citing_arxiv_id = %s")
            vals.append(citing_arxiv_id)
        if cited_arxiv_id is not None:
            sets.append("cited_arxiv_id = %s")
            vals.append(cited_arxiv_id)
        if bib_title is not None:
            sets.append("bib_title = %s")
            vals.append(bib_title)
        if bib_key is not None:
            sets.append("bib_key = %s")
            vals.append(bib_key)
        if author_cited_paper is not None:
            sets.append("author_cited_paper = %s")
            vals.append(author_cited_paper)
        if citing_sections is not None:
            sets.append("citing_sections = %s::jsonb")
            vals.append(json.dumps(citing_sections))
        if citing_paragraphs is not None:
            sets.append("citing_paragraphs = %s::jsonb")
            vals.append(json.dumps(citing_paragraphs))

        if not sets:
            return False

        sql = f"UPDATE arxiv_citations SET {', '.join(sets)} WHERE id = %s RETURNING id"
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

    def get_citation_by_id(self, id: int, return_all: bool = False):
        """
        If return_all=False: returns a single tuple
        If return_all=True: returns a list of tuples.
        Returns None if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key,
                       author_cited_paper, citing_sections, citing_paragraphs
                FROM arxiv_citations WHERE id = %s
                """,
                (id,)
            )
            rows = cur.fetchall() if return_all else cur.fetchone()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_citations_by_citing_paper(self, citing_arxiv_id: str) -> List[Tuple]:
        """
        Get all citations made by a specific paper.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key,
                       author_cited_paper, citing_sections, citing_paragraphs
                FROM arxiv_citations WHERE citing_arxiv_id = %s
                """,
                (citing_arxiv_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else []
        finally:
            conn.close()

    def get_citations_by_cited_paper(self, cited_arxiv_id: str) -> List[Tuple]:
        """
        Get all citations of a specific paper (papers that cite this one).
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key,
                       author_cited_paper, citing_sections, citing_paragraphs
                FROM arxiv_citations WHERE cited_arxiv_id = %s
                """,
                (cited_arxiv_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else []
        finally:
            conn.close()

    def check_citation_exists(self, citing_arxiv_id: str, cited_arxiv_id: str) -> bool:
        """
        True if a citation from citing_arxiv_id to cited_arxiv_id exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT 1 FROM arxiv_citations 
                WHERE citing_arxiv_id = %s AND cited_arxiv_id = %s 
                LIMIT 1
                """,
                (citing_arxiv_id, cited_arxiv_id)
            )
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    # -------------------------
    # Bulk import from CSV
    # -------------------------
    def construct_citation_table_from_csv(self, csv_file: str) -> bool:
        """
        Imports rows from a CSV with required columns: 
        ['citing_arxiv_id', 'cited_arxiv_id']
        Optional columns: ['bib_title', 'bib_key', 'author_cited_paper', 
                          'citing_sections', 'citing_paragraphs']
        Uses ON CONFLICT to skip duplicates.
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        df = pd.read_csv(csv_file)
        required_cols = ['citing_arxiv_id', 'cited_arxiv_id']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Error: CSV is missing required columns: {missing}")
            return False

        # Set defaults for optional columns
        for col in ['bib_title', 'bib_key', 'author_cited_paper']:
            if col not in df.columns:
                df[col] = None
        
        for col in ['citing_sections', 'citing_paragraphs']:
            if col not in df.columns:
                df[col] = '[]'
            else:
                # Convert to JSON string if not already
                df[col] = df[col].apply(lambda x: x if isinstance(x, str) else json.dumps(x) if x else '[]')

        # Filter out self-citations
        df = df[df['citing_arxiv_id'] != df['cited_arxiv_id']]

        rows: List[Tuple] = list(df[[
            'citing_arxiv_id', 'cited_arxiv_id', 'bib_title', 'bib_key',
            'author_cited_paper', 'citing_sections', 'citing_paragraphs'
        ]].itertuples(index=False, name=None))
        
        if not rows:
            print("No rows to import.")
            return True

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO arxiv_citations 
                    (citing_arxiv_id, cited_arxiv_id, bib_title, bib_key,
                     author_cited_paper, citing_sections, citing_paragraphs)
                VALUES %s
                ON CONFLICT (citing_arxiv_id, cited_arxiv_id) DO NOTHING
                """,
                rows,
                page_size=1000
            )
            cur.close()
        finally:
            conn.close()

        print(f"Successfully imported {len(rows)} citations from {csv_file}")
        return True