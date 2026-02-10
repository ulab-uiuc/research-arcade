import os
from typing import Optional, List, Tuple
import psycopg2
import psycopg2.extras
import pandas as pd
import json
from typing import Optional
from ..arxiv_utils.get_paragraph_global_id import get_paragraph_global_id_sql

class SQLArxivParagraphEquation:
    def __init__(self, host: str, dbname: str, user: str, password: str, port: str):
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.port = port
        self.autocommit = True
        self.create_paragraph_equations_table()

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

    def create_paragraph_equations_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_paragraph_equations (
                    id SERIAL PRIMARY KEY,
                    paragraph_id INTEGER NOT NULL,
                    equation_id INTEGER NOT NULL,
                    paper_arxiv_id VARCHAR(100) NOT NULL,
                    paper_section_id INTEGER NOT NULL,
                    paragraph_global_id INTEGER NOT NULL
                )
            """)
            cur.close()
        finally:
            conn.close()

    def insert_paragraph_equation_table(self, paragraph_id, equation_id, paper_arxiv_id, paper_section_id, paragraph_global_id=None) -> Optional[int]:
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
                print(f"Finding paragraph_global_id for paper_arxiv_id={paper_arxiv_id}, paper_section_id={paper_section_id}, paragraph_id={paragraph_id} => {paragraph_global_id}")
            cur.execute(
                """
                INSERT INTO arxiv_paragraph_equations
                (paragraph_id, equation_id, paper_arxiv_id, paper_section_id, paragraph_global_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (paragraph_id, equation_id, paper_arxiv_id, paper_section_id, paragraph_global_id)
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def get_paragraph_neighboring_equations(self, paragraph_id: int) -> Optional[pd.DataFrame]:
        conn = self._get_connection()
        try:
            query = "SELECT * FROM arxiv_paragraph_equations WHERE paragraph_id = %s"
            df = pd.read_sql(query, conn, params=(paragraph_id,))
            return None if df.empty else df.reset_index(drop=True)
        finally:
            conn.close()
    def get_equation_neighboring_paragraphs(self, equation_id: int) -> Optional[pd.DataFrame]:
        conn = self._get_connection()
        try:
            query = "SELECT * FROM arxiv_paragraph_equations WHERE equation_id = %s"
            df = pd.read_sql(query, conn, params=(equation_id,))
            return None if df.empty else df.reset_index(drop=True)
        finally:
            conn.close()

    def delete_paragraph_equation_by_equation_id(self, equation_id: int) -> bool:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM arxiv_paragraph_equations WHERE equation_id = %s RETURNING id", (equation_id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()
    
    def construct_paragraph_equations_table_from_api(self, arxiv_ids, dest_dir=None):
        """
        Matches existing references of type 'equation' to their corresponding equation and section nodes.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            for arxiv_id in arxiv_ids:
                cur.execute("""
                    SELECT paragraph_id, paper_section, reference_label, paragraph_global_id
                    FROM arxiv_paragraph_references 
                    WHERE paper_arxiv_id = %s AND reference_type = 'equation'
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


                    cur.execute("""
                        SELECT id FROM arxiv_paragraphs 
                        WHERE paper_arxiv_id = %s AND paper_section = %s AND paragraph_id = %s
                    """, (arxiv_id, section_title, para_id_str))
                    p_res = cur.fetchone()

                    cur.execute("""
                        SELECT id FROM arxiv_equations 
                        WHERE paper_arxiv_id = %s AND label = %s
                    """, (arxiv_id, label))
                    e_res = cur.fetchone()

                    cur.execute("""
                        SELECT id FROM arxiv_sections 
                        WHERE paper_arxiv_id = %s AND title = %s
                    """, (arxiv_id, section_title))
                    s_res = cur.fetchone()

                    if p_res and e_res and s_res:
                        self.insert_paragraph_equation_table(paragraph_id=para_id_str, equation_id=e_res[0], paper_arxiv_id=arxiv_id, paper_section_id=s_res[0], paragraph_global_id=paragraph_global_id)
            cur.close()
        finally:
            conn.close()
