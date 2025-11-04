import psycopg2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..arxiv_utils.multi_input.multi_download import MultiDownload
from ..arxiv_utils.graph_constructor.node_processor import NodeConstructor
from ..arxiv_utils.utils import arxiv_id_processor
class SQLArxivPaperFigure:
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
    def create_paper_figures_table(self):
        """
        Create the arxiv_paper_figures table with a composite uniqueness constraint.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_paper_figures (
                    paper_arxiv_id VARCHAR(100) NOT NULL,
                    figure_id INTEGER NOT NULL
                )
            """)
            # Composite unique index for conflict prevention
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS ux_arxiv_paper_figures_unique
                ON arxiv_paper_figures (paper_arxiv_id, figure_id)
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # Insert
    # -------------------------
    def insert_paper_figure(self, paper_arxiv_id, figure_id) -> bool:
        """
        Insert a (paper_arxiv_id, figure_id) record.
        Returns True if inserted, False if the pair already exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_paper_figures (paper_arxiv_id, figure_id)
                VALUES (%s, %s)
                ON CONFLICT ON CONSTRAINT ux_arxiv_paper_figures_unique DO NOTHING
                """,
                (paper_arxiv_id, figure_id)
            )
            inserted = cur.rowcount > 0
            cur.close()
            return inserted
        finally:
            conn.close()
