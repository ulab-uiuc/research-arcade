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
            password=self.password
        )
        conn.autocommit = self.autocommit
        return conn

    # -------------------------
    # DDL
    # -------------------------
    def create_citations_table(self):
        """
        Creates the arxiv_citations table.
        Enforces no self-citations and unique (citing_arxiv_id, cited_arxiv_id) pairs.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_citations (
                    id SERIAL PRIMARY KEY,
                    citing_arxiv_id VARCHAR(100) NOT NULL,
                    cited_arxiv_id VARCHAR(100) NOT NULL,
                    bib_title TEXT,
                    bib_key VARCHAR(255),
                    author_cited_paper TEXT,
                    citing_sections TEXT[],
                    citing_paragraphs TEXT[],
                    CONSTRAINT no_self_citation CHECK (citing_arxiv_id != cited_arxiv_id),
                    CONSTRAINT unique_citation UNIQUE (citing_arxiv_id, cited_arxiv_id)
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
        bib_title: str = None,
        bib_key: str = None, 
        author_cited_paper: str = None,
        citing_sections: List[str] = None,
        citing_paragraphs: List[str] = None
    ) -> Optional[int]:
        """
        Insert a citation row; returns generated id or None on conflict.
        Automatically prevents self-citations and duplicate citations.
        """
        if citing_arxiv_id == cited_arxiv_id:
            return None
            
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_citations 
                (citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, 
                 author_cited_paper, citing_sections, citing_paragraphs)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT ON CONSTRAINT unique_citation DO NOTHING
                RETURNING id
                """,
                (
                    citing_arxiv_id, 
                    cited_arxiv_id, 
                    bib_title, 
                    bib_key, 
                    author_cited_paper,
                    citing_sections or [],
                    citing_paragraphs or []
                )
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
        citing_arxiv_id: str = None,
        cited_arxiv_id: str = None,
        bib_title: str = None,
        bib_key: str = None,
        author_cited_paper: str = None,
        citing_sections: List[str] = None,
        citing_paragraphs: List[str] = None
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
            sets.append("citing_sections = %s")
            vals.append(citing_sections)
        if citing_paragraphs is not None:
            sets.append("citing_paragraphs = %s")
            vals.append(citing_paragraphs)

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
           (id, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, 
            author_cited_paper, citing_sections, citing_paragraphs)
        If return_all=True: returns a list of such tuples.
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

    def get_citations_by_paper(self, arxiv_id: str, as_citing: bool = True):
        """
        Get all citations for a paper.
        If as_citing=True: returns citations where this paper cites others
        If as_citing=False: returns citations where others cite this paper
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            if as_citing:
                cur.execute(
                    """
                    SELECT id, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key,
                           author_cited_paper, citing_sections, citing_paragraphs
                    FROM arxiv_citations WHERE citing_arxiv_id = %s
                    """,
                    (arxiv_id,)
                )
            else:
                cur.execute(
                    """
                    SELECT id, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key,
                           author_cited_paper, citing_sections, citing_paragraphs
                    FROM arxiv_citations WHERE cited_arxiv_id = %s
                    """,
                    (arxiv_id,)
                )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else []
        finally:
            conn.close()

    def check_citation_exists(self, citing_arxiv_id: str, cited_arxiv_id: str) -> bool:
        """
        Returns True if a citation already exists between these two papers.
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
        Imports rows from a CSV with columns:
          ['citing_arxiv_id', 'cited_arxiv_id', 'bib_title', 'bib_key', 
           'author_cited_paper', 'citing_sections', 'citing_paragraphs']
        Ignores any 'id' column; DB assigns SERIAL ids.
        Skips conflicts and self-citations.
        
        Note: citing_sections and citing_paragraphs should be JSON arrays in CSV,
        which will be parsed and stored as PostgreSQL arrays.
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        df = pd.read_csv(csv_file)
        required_cols = ['citing_arxiv_id', 'cited_arxiv_id', 'bib_title', 
                        'bib_key', 'author_cited_paper', 'citing_sections', 
                        'citing_paragraphs']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Error: CSV is missing required columns: {missing}")
            return False

        # Parse JSON arrays from CSV
        rows = []
        for _, row in df.iterrows():
            # Skip self-citations
            if row['citing_arxiv_id'] == row['cited_arxiv_id']:
                continue
                
            citing_sections = json.loads(row['citing_sections']) if pd.notna(row['citing_sections']) else []
            citing_paragraphs = json.loads(row['citing_paragraphs']) if pd.notna(row['citing_paragraphs']) else []
            
            rows.append((
                row['citing_arxiv_id'],
                row['cited_arxiv_id'],
                row['bib_title'] if pd.notna(row['bib_title']) else None,
                row['bib_key'] if pd.notna(row['bib_key']) else None,
                row['author_cited_paper'] if pd.notna(row['author_cited_paper']) else None,
                citing_sections,
                citing_paragraphs
            ))

        if not rows:
            print("No valid rows to import.")
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
                ON CONFLICT ON CONSTRAINT unique_citation DO NOTHING
                """,
                rows,
                page_size=1000
            )
            cur.close()
        finally:
            conn.close()

        print(f"Successfully imported {len(rows)} citations from {csv_file}")
        return True