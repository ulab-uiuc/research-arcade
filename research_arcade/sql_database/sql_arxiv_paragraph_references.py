import os
from typing import Optional, List, Tuple
import psycopg2
import psycopg2.extras
import pandas as pd
import json

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..arxiv_utils.multi_input.multi_download import MultiDownload
from ..arxiv_utils.graph_constructor.node_processor import NodeConstructor
from ..arxiv_utils.utils import arxiv_id_processor


class SQLArxivParagraphReference:
    def __init__(self, host: str, dbname: str, user: str, password: str, port: str):
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.port = port
        self.autocommit = True
        self.create_paragraph_references_table()

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
    def create_paragraph_references_table(self):
        """
        Creates the paragraph_references table.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_paragraph_references (
                    id SERIAL PRIMARY KEY,
                    paragraph_id INT NOT NULL,
                    paper_section VARCHAR(255) NOT NULL,
                    paper_arxiv_id VARCHAR(100) NOT NULL,
                    reference_label VARCHAR(255),
                    reference_type VARCHAR(100)
                )
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # CRUD
    # -------------------------
    def insert_paragraph_reference(self, paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type) -> Optional[int]:
        """
        Insert a paragraph reference. Returns generated id.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_paragraph_references
                (paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type)
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def delete_paragraph_reference_by_id(self, id: int) -> bool:
        """
        Delete by id. Returns True if deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM arxiv_paragraph_references WHERE id = %s RETURNING id", (id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def update_paragraph_reference(self, id: int, paragraph_id=None, paper_section=None, paper_arxiv_id=None, reference_label=None, reference_type=None) -> bool:
        """
        Partial update by id. Only non-None fields are updated.
        Returns True if updated.
        """
        sets, vals = [], []

        if paragraph_id is not None:
            sets.append("paragraph_id = %s")
            vals.append(paragraph_id)
        if paper_section is not None:
            sets.append("paper_section = %s")
            vals.append(paper_section)
        if paper_arxiv_id is not None:
            sets.append("paper_arxiv_id = %s")
            vals.append(paper_arxiv_id)
        if reference_label is not None:
            sets.append("reference_label = %s")
            vals.append(reference_label)
        if reference_type is not None:
            sets.append("reference_type = %s")
            vals.append(reference_type)

        if not sets:
            return False

        sql = f"UPDATE arxiv_paragraph_references SET {', '.join(sets)} WHERE id = %s RETURNING id"
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

    def get_paragraph_reference_by_id(self, id: int, return_all: bool = False):
        """
        If return_all=False: returns a single tuple
           (id, paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type)
        If return_all=True: returns a list of such tuples.
        Returns None if no row found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type "
                "FROM arxiv_paragraph_references WHERE id = %s",
                (id,)
            )
            rows = cur.fetchall() if return_all else cur.fetchone()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_paragraph_references_by_paragraph_id(self, paragraph_id: str):
        """
        Returns a list of tuples for a given paragraph_id.
        Each tuple: (id, paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type)
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type "
                "FROM arxiv_paragraph_references WHERE paragraph_id = %s ORDER BY id",
                (paragraph_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_all_paragraph_references(self):
        """Get all paragraph references from the database."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type "
                "FROM arxiv_paragraph_references"
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_paragraph_neighboring_references(self, paragraph_id: int):
        """Get all references for a given paragraph."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type "
                "FROM arxiv_paragraph_references WHERE paragraph_id = %s",
                (paragraph_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_reference_neighboring_paragraphs(self, reference_id: int):
        """Get all paragraphs that reference a given reference_id."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type "
                "FROM arxiv_paragraph_references WHERE id = %s",
                (reference_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def delete_paragraph_reference_by_paragraph_id(self, paragraph_id: int) -> int:
        """Delete all references for a given paragraph. Returns count of deleted rows."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paragraph_references WHERE paragraph_id = %s",
                (paragraph_id,)
            )
            count = cur.rowcount
            cur.close()
            return count
        finally:
            conn.close()

    def delete_paragraph_reference_by_reference_id(self, reference_id: int) -> int:
        """Delete a reference by its id. Returns count of deleted rows."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paragraph_references WHERE id = %s",
                (reference_id,)
            )
            count = cur.rowcount
            cur.close()
            return count
        finally:
            conn.close()

    def check_paragraph_reference_exists(self, id: int) -> bool:
        """
        Returns True if a paragraph reference with the given id exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM arxiv_paragraph_references WHERE id = %s LIMIT 1", (id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    # -------------------------
    # Bulk import from CSV
    # -------------------------
    def construct_paragraph_references_table_from_csv(self, csv_file: str) -> bool:
        """
        Imports rows from a CSV with columns:
          ['paragraph_id', 'paper_section', 'paper_arxiv_id', 'reference_label', 'reference_type']
        Ignores any 'id' column; DB assigns SERIAL ids.
        Returns True on success, False on validation error or missing file.
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        df = pd.read_csv(csv_file)
        required_cols = ['paragraph_id', 'paper_section', 'paper_arxiv_id', 'reference_label', 'reference_type']
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
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO arxiv_paragraph_references
                (paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type)
                VALUES %s
                """,
                rows,
                page_size=1000
            )
            cur.close()
        finally:
            conn.close()

        print(f"Successfully imported {len(rows)} paragraph references from {csv_file}")
        return True

    def construct_table_from_csv(self, csv_file: str) -> bool:
        """Alias for construct_paragraph_references_table_from_csv for consistency."""
        return self.construct_paragraph_references_table_from_csv(csv_file)

    def construct_table_from_json(self, json_file):
        """
        Construct the paragraph-reference relationships from an external JSON file.
        
        Args:
            json_file: Path to the JSON file
            
        Expected JSON format:
            [
                {
                    "paragraph_id": 1,
                    "paper_section": "introduction",
                    "paper_arxiv_id": "1706.03762v7",
                    "reference_label": "fig:1",
                    "reference_type": "figure"
                },
                ...
            ]
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(json_file):
            print(f"Error: JSON file {json_file} does not exist.")
            return False

        try:
            # Load JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(json_data, dict):
                if 'paragraph_references' in json_data:
                    relations_list = json_data['paragraph_references']
                else:
                    relations_list = [json_data]
            elif isinstance(json_data, list):
                relations_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not relations_list:
                print("Error: No paragraph-reference data found in JSON file")
                return False
            
            # Convert to list of tuples for bulk insert
            rows = []
            for relation in relations_list:
                if 'paragraph_id' not in relation or 'paper_section' not in relation or 'paper_arxiv_id' not in relation or 'reference_label' not in relation or 'reference_type' not in relation:
                    print(f"Warning: Skipping relation missing required fields: {relation}")
                    continue
                    
                rows.append((
                    relation['paragraph_id'],
                    relation['paper_section'],
                    relation['paper_arxiv_id'],
                    relation['reference_label'],
                    relation['reference_type']
                ))

            if not rows:
                print("No valid paragraph-reference records to import")
                return False

            conn = self._get_connection()
            try:
                cur = conn.cursor()
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO arxiv_paragraph_references
                    (paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type)
                    VALUES %s
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} paragraph-reference relationships from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing paragraph-reference relationships from JSON: {e}")
            return False