import os
from typing import Optional, List, Tuple
import psycopg2
import psycopg2.extras
import pandas as pd
import json
from typing import Optional

from research_arcade.research_arcade.arxiv_utils.get_paragraph_global_id import get_paragraph_global_id_sql

class SQLArxivParagraphFigure:
    def __init__(self, host: str, dbname: str, user: str, password: str, port: str):
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.port = port
        self.autocommit = True
        self.create_paragraph_figures_table()

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

    def create_paragraph_figures_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_paragraph_figures (
                    id SERIAL PRIMARY KEY,
                    paragraph_id INTEGER NOT NULL,
                    figure_id INTEGER NOT NULL,
                    paper_arxiv_id VARCHAR(100) NOT NULL,
                    paper_section_id INTEGER NOT NULL,
                    paragraph_global_id INTEGER NOT NULL
                )
            """)
            cur.close()
        finally:
            conn.close()

    def insert_paragraph_figure_table(self, paragraph_id, figure_id, paper_arxiv_id, paper_section_id, paragraph_global_id=None) -> Optional[int]:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            if paragraph_global_id is None:
                paragraph_global_id = get_paragraph_global_id_sql(
                    conn=conn,
                    paper_arxiv_id=paper_arxiv_id,
                    paper_section_id=paper_section_id,
                    paragraph_id=paragraph_id
                )
            cur.execute(
                """
                INSERT INTO arxiv_paragraph_figures
                (paragraph_id, figure_id, paper_arxiv_id, paper_section_id, paragraph_global_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (paragraph_id, figure_id, paper_arxiv_id, paper_section_id, paragraph_global_id)
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def get_paragraph_neighboring_figures(self, paragraph_id: int) -> Optional[pd.DataFrame]:
        conn = self._get_connection()
        try:
            query = "SELECT * FROM arxiv_paragraph_figures WHERE paragraph_id = %s"
            df = pd.read_sql(query, conn, params=(paragraph_id,))
            return None if df.empty else df.reset_index(drop=True)
        finally:
            conn.close()

    def get_figure_neighboring_paragraphs(self, figure_id: int) -> Optional[pd.DataFrame]:
        conn = self._get_connection()
        try:
            query = "SELECT * FROM arxiv_paragraph_figures WHERE figure_id = %s"
            df = pd.read_sql(query, conn, params=(figure_id,))
            return None if df.empty else df.reset_index(drop=True)
        finally:
            conn.close()

    def delete_paragraph_figure_by_figure_id(self, figure_id: int) -> bool:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM arxiv_paragraph_figures WHERE figure_id = %s RETURNING id", (figure_id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def construct_paragraph_figures_table_from_api(self, arxiv_ids, dest_dir=None):
        """
        Translates the logic from the CSV version to SQL. 
        Matches existing references of type 'figure' to their corresponding figure and section nodes.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            for arxiv_id in arxiv_ids:
                # 1. Fetch relevant references from the references table
                cur.execute("""
                    SELECT paragraph_id, paper_section, reference_label, paragraph_global_id
                    FROM arxiv_paragraph_references 
                    WHERE paper_arxiv_id = %s AND reference_type = 'figure'
                """, (arxiv_id,))
                refs = cur.fetchall()

                for para_id_str, section_title, ref_label, paragraph_global_id in refs:
                    label = f"\\label{{{ref_label}}}"
                    if paragraph_global_id is None:

                        # 2. Find Global Paragraph ID
                        cur.execute("""
                            SELECT id FROM arxiv_paragraphs 
                            WHERE paper_arxiv_id = %s AND paper_section = %s AND paragraph_id = %s
                        """, (arxiv_id, section_title, para_id_str))
                        p_res = cur.fetchone()
                        paragraph_global_id = p_res[0] if p_res else None
                        
                    # 3. Find Figure ID
                    cur.execute("""
                        SELECT id FROM arxiv_figures 
                        WHERE paper_arxiv_id = %s AND label = %s
                    """, (arxiv_id, label))
                    f_res = cur.fetchone()

                    # 4. Find Section ID
                    cur.execute("""
                        SELECT id FROM arxiv_sections 
                        WHERE paper_arxiv_id = %s AND title = %s
                    """, (arxiv_id, section_title))
                    s_res = cur.fetchone()

                    if paragraph_global_id and f_res and s_res:
                        self.insert_paragraph_figure_table(paragraph_global_id=paragraph_global_id, figure_id=f_res[0], paper_arxiv_id=arxiv_id, paper_section_id=s_res[0], paragraph_id=para_id_str)
            cur.close()
        finally:
            conn.close()