"""
The edge between paragraph and section where the section is referenced by the paragraph,
in contrast to the edge between paragraph and section where the section is the parent of the paragraph.

This tracks cross-sectional references like "as shown in Section 3" or "see Section 2.1".
"""

import os
from typing import Optional, List, Tuple
import psycopg2
import psycopg2.extras
import pandas as pd
import json

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..arxiv_utils.utils import get_paragraph_num
from ..arxiv_utils.get_paragraph_global_id import get_paragraph_global_id_sql


class SQLArxivParagraphSectionReference:
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
    def create_paragraph_section_references_table(self):
        """
        Creates the paragraph_section_references table for tracking
        cross-sectional references (e.g., "as shown in Section 3").
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_paragraph_section_references (
                    id SERIAL PRIMARY KEY,
                    paragraph_id INT NOT NULL,
                    referenced_section_id INT NOT NULL,
                    paper_arxiv_id VARCHAR(100) NOT NULL,
                    paper_section_id INT NOT NULL,
                    paragraph_global_id INT
                )
            """)
            # Create indexes for efficient querying
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_para_sec_ref_paragraph_id 
                ON arxiv_paragraph_section_references (paragraph_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_para_sec_ref_referenced_section_id 
                ON arxiv_paragraph_section_references (referenced_section_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_para_sec_ref_paper_arxiv_id 
                ON arxiv_paragraph_section_references (paper_arxiv_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_para_sec_ref_paragraph_global_id 
                ON arxiv_paragraph_section_references (paragraph_global_id)
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # CRUD Operations
    # -------------------------
    def insert_paragraph_section_reference(
        self,
        paragraph_id: int,
        referenced_section_id: int,
        paper_arxiv_id: str,
        paper_section_id: int,
        paragraph_global_id: int = None
    ) -> Optional[int]:
        """
        Insert a paragraph-section reference edge.
        Returns generated id or None on failure.
        """
        if paragraph_global_id is None:
            paragraph_global_id = get_paragraph_global_id_sql(
                host=self.host,
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                port=self.port,
                paper_arxiv_id=paper_arxiv_id,
                paper_section_id=paper_section_id,
                paragraph_id=paragraph_id
            )

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_paragraph_section_references 
                (paragraph_id, referenced_section_id, paper_arxiv_id, paper_section_id, paragraph_global_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (paragraph_id, referenced_section_id, paper_arxiv_id, paper_section_id, paragraph_global_id)
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def get_all_paragraph_section_references(self) -> Optional[pd.DataFrame]:
        """Get all paragraph-section references as a DataFrame."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM arxiv_paragraph_section_references")
            rows = cur.fetchall()
            cur.close()
            
            if not rows:
                return None
            
            columns = ['id', 'paragraph_id', 'referenced_section_id', 
                       'paper_arxiv_id', 'paper_section_id', 'paragraph_global_id']
            return pd.DataFrame(rows, columns=columns)
        finally:
            conn.close()

    def get_paragraph_neighboring_referenced_sections(self, paragraph_id: int) -> Optional[pd.DataFrame]:
        """Get all sections referenced by a given paragraph."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM arxiv_paragraph_section_references WHERE paragraph_id = %s",
                (paragraph_id,)
            )
            rows = cur.fetchall()
            cur.close()
            
            if not rows:
                return None
            
            columns = ['id', 'paragraph_id', 'referenced_section_id', 
                       'paper_arxiv_id', 'paper_section_id', 'paragraph_global_id']
            return pd.DataFrame(rows, columns=columns)
        finally:
            conn.close()

    def get_paragraph_global_id_neighboring_referenced_sections(self, paragraph_global_id: int) -> Optional[pd.DataFrame]:
        """Get all sections referenced by a given paragraph (using global ID)."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM arxiv_paragraph_section_references WHERE paragraph_global_id = %s",
                (paragraph_global_id,)
            )
            rows = cur.fetchall()
            cur.close()
            
            if not rows:
                return None
            
            columns = ['id', 'paragraph_id', 'referenced_section_id', 
                       'paper_arxiv_id', 'paper_section_id', 'paragraph_global_id']
            return pd.DataFrame(rows, columns=columns)
        finally:
            conn.close()

    def get_section_neighboring_referencing_paragraphs(self, referenced_section_id: int) -> Optional[pd.DataFrame]:
        """Get all paragraphs that reference a given section."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM arxiv_paragraph_section_references WHERE referenced_section_id = %s",
                (referenced_section_id,)
            )
            rows = cur.fetchall()
            cur.close()
            
            if not rows:
                return None
            
            columns = ['id', 'paragraph_id', 'referenced_section_id', 
                       'paper_arxiv_id', 'paper_section_id', 'paragraph_global_id']
            return pd.DataFrame(rows, columns=columns)
        finally:
            conn.close()

    def get_references_by_paper(self, paper_arxiv_id: str) -> Optional[pd.DataFrame]:
        """Get all paragraph-section references for a specific paper."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM arxiv_paragraph_section_references WHERE paper_arxiv_id = %s",
                (paper_arxiv_id,)
            )
            rows = cur.fetchall()
            cur.close()
            
            if not rows:
                return None
            
            columns = ['id', 'paragraph_id', 'referenced_section_id', 
                       'paper_arxiv_id', 'paper_section_id', 'paragraph_global_id']
            return pd.DataFrame(rows, columns=columns)
        finally:
            conn.close()

    def get_references_from_section(self, paper_section_id: int) -> Optional[pd.DataFrame]:
        """Get all section references made from paragraphs within a given section."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM arxiv_paragraph_section_references WHERE paper_section_id = %s",
                (paper_section_id,)
            )
            rows = cur.fetchall()
            cur.close()
            
            if not rows:
                return None
            
            columns = ['id', 'paragraph_id', 'referenced_section_id', 
                       'paper_arxiv_id', 'paper_section_id', 'paragraph_global_id']
            return pd.DataFrame(rows, columns=columns)
        finally:
            conn.close()

    # -------------------------
    # Delete Operations
    # -------------------------
    def delete_paragraph_section_reference_by_id(self, id: int) -> bool:
        """Delete by id. Returns True if deleted."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paragraph_section_references WHERE id = %s RETURNING id",
                (id,)
            )
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def delete_paragraph_section_reference_by_section_id(self, referenced_section_id: int) -> bool:
        """Delete all references to a given section. Returns True if any deleted."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paragraph_section_references WHERE referenced_section_id = %s RETURNING id",
                (referenced_section_id,)
            )
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def delete_paragraph_section_reference_by_paragraph_id(self, paragraph_id: int) -> int:
        """Delete all references from a given paragraph. Returns count of deleted rows."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paragraph_section_references WHERE paragraph_id = %s",
                (paragraph_id,)
            )
            count = cur.rowcount
            cur.close()
            return count
        finally:
            conn.close()

    def delete_paragraph_section_reference_by_paragraph_global_id(self, paragraph_global_id: int) -> int:
        """Delete all references from a given paragraph (using global ID). Returns count of deleted rows."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paragraph_section_references WHERE paragraph_global_id = %s",
                (paragraph_global_id,)
            )
            count = cur.rowcount
            cur.close()
            return count
        finally:
            conn.close()

    def delete_paragraph_section_reference_by_paragraph_section_id(
        self,
        paragraph_id: int,
        referenced_section_id: int
    ) -> int:
        """Delete specific paragraph-section reference pair. Returns count of deleted rows."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM arxiv_paragraph_section_references 
                WHERE paragraph_id = %s AND referenced_section_id = %s
                """,
                (paragraph_id, referenced_section_id)
            )
            count = cur.rowcount
            cur.close()
            return count
        finally:
            conn.close()

    # -------------------------
    # Bulk Import Operations
    # -------------------------
    def construct_table_from_csv(self, csv_file: str, dest_dir: str = None) -> bool:
        """
        Construct the paragraph-section reference relationships from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file
            dest_dir: Optional destination directory (unused, kept for API consistency)
            
        Expected CSV format:
            - Required columns: paragraph_id, referenced_section_id,
              paper_arxiv_id, paper_section_id
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        try:
            external_df = pd.read_csv(csv_file, dtype={'paper_arxiv_id': str})

            required_cols = ['paragraph_id', 'referenced_section_id', 'paper_arxiv_id', 'paper_section_id']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: External CSV is missing required columns: {missing_cols}")
                return False

            # Add paragraph_global_id if not present
            if 'paragraph_global_id' not in external_df.columns:
                external_df['paragraph_global_id'] = None

            # Prepare rows for bulk insert
            rows = [
                (
                    row['paragraph_id'],
                    row['referenced_section_id'],
                    row['paper_arxiv_id'],
                    row['paper_section_id'],
                    row['paragraph_global_id'] if pd.notna(row['paragraph_global_id']) else None
                )
                for _, row in external_df.iterrows()
            ]

            conn = self._get_connection()
            try:
                cur = conn.cursor()
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO arxiv_paragraph_section_references 
                    (paragraph_id, referenced_section_id, paper_arxiv_id, paper_section_id, paragraph_global_id)
                    VALUES %s
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} paragraph-section references from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing paragraph-section references from CSV: {e}")
            return False

    def construct_table_from_json(self, json_file: str) -> bool:
        """
        Construct the paragraph-section reference relationships from an external JSON file.
        
        Args:
            json_file: Path to the JSON file
            
        Expected JSON format:
            [
                {
                    "paragraph_id": 1,
                    "referenced_section_id": 5,
                    "paper_arxiv_id": "1706.03762v7",
                    "paper_section_id": 2,
                    "paragraph_global_id": 123  # optional
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
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(json_data, dict):
                if 'paragraph_section_references' in json_data:
                    relations_list = json_data['paragraph_section_references']
                else:
                    relations_list = [json_data]
            elif isinstance(json_data, list):
                relations_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not relations_list:
                print("Error: No paragraph-section reference data found in JSON file")
                return False

            # Check for required fields
            required_cols = ['paragraph_id', 'referenced_section_id', 'paper_arxiv_id', 'paper_section_id']
            if relations_list:
                missing_cols = [col for col in required_cols if col not in relations_list[0]]
                if missing_cols:
                    print(f"Error: JSON data is missing required fields: {missing_cols}")
                    return False

            # Prepare rows for bulk insert
            rows = [
                (
                    item['paragraph_id'],
                    item['referenced_section_id'],
                    item['paper_arxiv_id'],
                    item['paper_section_id'],
                    item.get('paragraph_global_id', None)
                )
                for item in relations_list
            ]

            conn = self._get_connection()
            try:
                cur = conn.cursor()
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO arxiv_paragraph_section_references 
                    (paragraph_id, referenced_section_id, paper_arxiv_id, paper_section_id, paragraph_global_id)
                    VALUES %s
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} paragraph-section references from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing paragraph-section references from JSON: {e}")
            return False

    def construct_paragraph_section_references_table_from_api(
        self,
        arxiv_ids: List[str],
        dest_dir: str = None
    ):
        """
        Construct paragraph-section reference edges from preprocessed data.
        
        First select all refs in the paragraph-reference that are sections.
        Then for these refs, we match them with their corresponding paragraphs and referenced sections.
        Finally we save the edges.
        
        Args:
            arxiv_ids: List of arXiv paper IDs to process
            dest_dir: Optional destination directory (unused, kept for API consistency)
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            
            for arxiv_id in arxiv_ids:
                # Get all section references for this paper
                cur.execute(
                    """
                    SELECT paragraph_id, paper_section, reference_label, paragraph_global_id
                    FROM arxiv_paragraph_references 
                    WHERE paper_arxiv_id = %s AND reference_type = 'section'
                    """,
                    (arxiv_id,)
                )
                section_refs = cur.fetchall()
                
                for para_ref in section_refs:
                    paragraph_id = para_ref[0]
                    paper_section = para_ref[1]
                    reference_label = para_ref[2]
                    paragraph_global_id = para_ref[3]
                    
                    # Get paragraph global ID if not provided
                    if paragraph_global_id is None:
                        cur.execute(
                            """
                            SELECT id FROM arxiv_paragraphs 
                            WHERE paper_arxiv_id = %s AND paper_section = %s AND paragraph_id = %s
                            """,
                            (arxiv_id, paper_section, paragraph_id)
                        )
                        para_result = cur.fetchone()
                        
                        if para_result is None:
                            print(f"Warning: Paragraph not found for arxiv_id={arxiv_id}, "
                                  f"section={paper_section}, para_id={paragraph_id}")
                            continue
                        
                        paragraph_global_id = para_result[0]
                    
                    # Get the section ID of the section containing this paragraph
                    cur.execute(
                        """
                        SELECT id FROM arxiv_sections 
                        WHERE paper_arxiv_id = %s AND title = %s
                        """,
                        (arxiv_id, paper_section)
                    )
                    section_result = cur.fetchone()
                    
                    if section_result is None:
                        print(f"Warning: Section not found for arxiv_id={arxiv_id}, section={paper_section}")
                        continue
                    
                    paper_section_id = section_result[0]
                    
                    # ============================================================
                    # Find the referenced section ID
                    # ============================================================
                    # The reference_label could be in various formats:
                    # - "sec:introduction", "sec:methods", "sec:3"
                    # - "Section 3", "Section 3.1"
                    # - Just a number "3" or "3.1"
                    
                    referenced_section_id = None
                    
                    # Construct possible label formats to search for
                    label_variants = [
                        f"\\label{{{reference_label}}}",
                        reference_label,
                        f"sec:{reference_label}",
                    ]
                    
                    # Try matching by label
                    for label_variant in label_variants:
                        cur.execute(
                            """
                            SELECT id FROM arxiv_sections 
                            WHERE paper_arxiv_id = %s AND label = %s
                            """,
                            (arxiv_id, label_variant)
                        )
                        ref_section_result = cur.fetchone()
                        if ref_section_result:
                            referenced_section_id = ref_section_result[0]
                            break
                    
                    # Fallback: try matching by title (case-insensitive partial match)
                    if referenced_section_id is None:
                        search_term = reference_label.lower().replace('sec:', '').strip()
                        cur.execute(
                            """
                            SELECT id FROM arxiv_sections 
                            WHERE paper_arxiv_id = %s AND LOWER(title) LIKE %s
                            """,
                            (arxiv_id, f"%{search_term}%")
                        )
                        ref_section_result = cur.fetchone()
                        if ref_section_result:
                            referenced_section_id = ref_section_result[0]
                    
                    if referenced_section_id is None:
                        print(f"Warning: Referenced section not found for label={reference_label}, "
                              f"arxiv_id={arxiv_id}")
                        continue
                    
                    # Insert the paragraph-section reference edge
                    cur.execute(
                        """
                        INSERT INTO arxiv_paragraph_section_references 
                        (paragraph_id, referenced_section_id, paper_arxiv_id, paper_section_id, paragraph_global_id)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (paragraph_id, referenced_section_id, arxiv_id, paper_section_id, paragraph_global_id)
                    )
            
            cur.close()
        finally:
            conn.close()
            