import psycopg2

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..arxiv_utils.multi_input.multi_download import MultiDownload
from ..arxiv_utils.graph_constructor.node_processor import NodeConstructor
from ..arxiv_utils.utils import arxiv_id_processor
class SQLArxivPaperAuthor:
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
    def create_paper_authors_table(self):
        """
        Create the arxiv_paper_authors table with composite uniqueness on (paper_arxiv_id, author_id).
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_paper_authors (
                    paper_arxiv_id VARCHAR(100) NOT NULL,
                    author_id VARCHAR(100) NOT NULL,
                    author_sequence INTEGER NOT NULL
                )
            """)
            # Create composite unique index to prevent duplicates
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS ux_arxiv_paper_authors_unique
                ON arxiv_paper_authors (paper_arxiv_id, author_id)
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # Insert
    # -------------------------
    def insert_paper_author(self, paper_arxiv_id, author_id, author_sequence) -> bool:
        """
        Insert a (paper_arxiv_id, author_id, author_sequence) record.
        Returns True if inserted, False if already exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_paper_authors (paper_arxiv_id, author_id, author_sequence)
                VALUES (%s, %s, %s)
                ON CONFLICT ON CONSTRAINT ux_arxiv_paper_authors_unique DO NOTHING
                """,
                (paper_arxiv_id, author_id, author_sequence)
            )
            inserted = cur.rowcount > 0  # 1 if inserted, 0 if conflict
            cur.close()
            return inserted
        finally:
            conn.close()
