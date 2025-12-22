import os
from typing import Optional, List, Tuple
import psycopg2
import psycopg2.extras
import pandas as pd
import json

class SQLArxivParagraphTable:
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

    def create_paragraph_tables_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_paragraph_tables (
                    id SERIAL PRIMARY KEY,
                    paragraph_id INTEGER NOT NULL,
                    table_id INTEGER NOT NULL,
                    paper_arxiv_id VARCHAR(100) NOT NULL,
                    paper_section_id INTEGER NOT NULL
                )
            """)
            cur.close()
        finally:
            conn.close()

    def insert_paragraph_table_table(self, paragraph_id, table_id, paper_arxiv_id, paper_section_id) -> Optional[int]:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_paragraph_tables
                (paragraph_id, table_id, paper_arxiv_id, paper_section_id)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (paragraph_id, table_id, paper_arxiv_id, paper_section_id)
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def get_paragraph_neighboring_tables(self, paragraph_id: int) -> Optional[pd.DataFrame]:
        conn = self._get_connection()
        try:
            query = "SELECT * FROM arxiv_paragraph_tables WHERE paragraph_id = %s"
            df = pd.read_sql(query, conn, params=(paragraph_id,))
            return None if df.empty else df.reset_index(drop=True)
        finally:
            conn.close()

    def delete_paragraph_table_by_table_id(self, table_id: int) -> bool:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM arxiv_paragraph_tables WHERE table_id = %s RETURNING id", (table_id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def construct_paragraph_tables_table_from_api(self, arxiv_ids):
        """
        Matches existing references of type 'table' to their corresponding table and section nodes.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            for arxiv_id in arxiv_ids:
                cur.execute("""
                    SELECT paragraph_id, paper_section, reference_label 
                    FROM arxiv_paragraph_references 
                    WHERE paper_arxiv_id = %s AND reference_type = 'table'
                """, (arxiv_id,))
                refs = cur.fetchall()

                for para_id_str, section_title, ref_label in refs:
                    label = f"\\label{{{ref_label}}}"
                    
                    cur.execute("""
                        SELECT id FROM arxiv_paragraphs 
                        WHERE paper_arxiv_id = %s AND paper_section = %s AND paragraph_id = %s
                    """, (arxiv_id, section_title, para_id_str))
                    p_res = cur.fetchone()
                    
                    cur.execute("""
                        SELECT id FROM arxiv_tables 
                        WHERE paper_arxiv_id = %s AND label = %s
                    """, (arxiv_id, label))
                    t_res = cur.fetchone()

                    cur.execute("""
                        SELECT id FROM arxiv_sections 
                        WHERE paper_arxiv_id = %s AND title = %s
                    """, (arxiv_id, section_title))
                    s_res = cur.fetchone()

                    if p_res and t_res and s_res:
                        self.insert_paragraph_table_table(p_res[0], t_res[0], arxiv_id, s_res[0])
            cur.close()
        finally:
            conn.close()