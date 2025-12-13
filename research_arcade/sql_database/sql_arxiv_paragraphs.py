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
from ..arxiv_utils.utils import get_paragraph_num
from ..arxiv_utils.paper_collector.paper_graph_processor import PaperGraphProcessor


class SQLArxivParagraphs:
    def __init__(self, host: str, dbname: str, user: str, password: str, port: str):
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.port = port
        self.autocommit = True
        self.create_paragraphs_table()

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
    def create_paragraphs_table(self):
        """
        Creates table with a composite uniqueness on (paragraph_id, paper_arxiv_id, paper_section)
        to mirror the CSV conflict check.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_paragraphs (
                    id SERIAL PRIMARY KEY,
                    paragraph_id VARCHAR(255) NOT NULL,
                    content TEXT,
                    paper_arxiv_id VARCHAR(100) NOT NULL,
                    paper_section VARCHAR(255) NOT NULL
                )
            """)
            # Composite unique index for conflict prevention
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS ux_arxiv_paragraphs_unique
                ON arxiv_paragraphs (paragraph_id, paper_arxiv_id, paper_section)
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # CRUD
    # -------------------------
    def insert_paragraph(self, paragraph_id, content, paper_arxiv_id, paper_section, section_id=None, paragraph_in_paper_id=None) -> Optional[int]:
        """
        Insert a paragraph. Returns generated id or None if conflicts with the composite unique constraint.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_paragraphs (paragraph_id, content, paper_arxiv_id, paper_section)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT ON CONSTRAINT ux_arxiv_paragraphs_unique DO NOTHING
                RETURNING id
                """,
                (paragraph_id, content, paper_arxiv_id, paper_section)
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def delete_paragraph_by_id(self, id: int) -> bool:
        """
        Delete a paragraph by id. Returns True if a row was deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM arxiv_paragraphs WHERE id = %s RETURNING id", (id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def delete_paragraph_by_paper_arxiv_id(self, paper_arxiv_id: str) -> int:
        """
        Delete all paragraphs for a given paper_arxiv_id. Returns number of rows deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paragraphs WHERE paper_arxiv_id = %s RETURNING id",
                (paper_arxiv_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return len(rows) if rows else 0
        finally:
            conn.close()

    def delete_paragraph_by_paper_section(self, paper_arxiv_id: str, paper_section: str) -> int:
        """
        Delete all paragraphs for a given (paper_arxiv_id, paper_section).
        Returns number of rows deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM arxiv_paragraphs
                WHERE paper_arxiv_id = %s AND paper_section = %s
                RETURNING id
                """,
                (paper_arxiv_id, paper_section)
            )
            rows = cur.fetchall()
            cur.close()
            return len(rows) if rows else 0
        finally:
            conn.close()

    def update_paragraph(self, id: int, paragraph_id=None, content=None, paper_arxiv_id=None, paper_section=None) -> bool:
        """
        Partial update by id. Only non-None fields are updated.
        Returns True if a row was updated.
        """
        sets: List[str] = []
        vals: List = []

        if paragraph_id is not None:
            sets.append("paragraph_id = %s")
            vals.append(paragraph_id)
        if content is not None:
            sets.append("content = %s")
            vals.append(content)
        if paper_arxiv_id is not None:
            sets.append("paper_arxiv_id = %s")
            vals.append(paper_arxiv_id)
        if paper_section is not None:
            sets.append("paper_section = %s")
            vals.append(paper_section)

        if not sets:
            return False

        sql = f"UPDATE arxiv_paragraphs SET {', '.join(sets)} WHERE id = %s RETURNING id"
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

    def get_paragraph_by_id(self, id: int, return_all: bool = False) -> Optional[pd.DataFrame]:
        """
        Returns a DataFrame with the paragraph(s) matching the given id.
        If return_all=False (default): returns DataFrame with single row or None
        If return_all=True: returns DataFrame with all matching rows or None
        
        Returns None if no row found.
        """
        conn = self._get_connection()
        try:
            query = """
                SELECT id, paragraph_id, content, paper_arxiv_id, paper_section
                FROM arxiv_paragraphs 
                WHERE id = %s
            """
            df = pd.read_sql(query, conn, params=(id,))
            
            if df.empty:
                return None
            
            return df
        finally:
            conn.close()

    def get_paragraphs_by_arxiv_id(self, arxiv_id: str) -> Optional[pd.DataFrame]:
        """
        Returns a DataFrame with all paragraphs for a given arxiv_id.
        Returns None if no rows exist.
        """
        conn = self._get_connection()
        try:
            query = """
                SELECT id, paragraph_id, content, paper_arxiv_id, paper_section
                FROM arxiv_paragraphs 
                WHERE paper_arxiv_id = %s 
                ORDER BY id
            """
            df = pd.read_sql(query, conn, params=(arxiv_id,))
            
            return None if df.empty else df
        finally:
            conn.close()

    def get_paragraphs_by_paper_section(self, paper_arxiv_id: str, paper_section: str) -> Optional[pd.DataFrame]:
        """
        Returns a DataFrame with all paragraphs for a given (paper_arxiv_id, paper_section).
        Returns None if no rows exist.
        """
        conn = self._get_connection()
        try:
            query = """
                SELECT id, paragraph_id, content, paper_arxiv_id, paper_section
                FROM arxiv_paragraphs 
                WHERE paper_arxiv_id = %s AND paper_section = %s
                ORDER BY id
            """
            df = pd.read_sql(query, conn, params=(paper_arxiv_id, paper_section))
            
            return None if df.empty else df
        finally:
            conn.close()

    def get_id_by_arxiv_id_section_paragraph_id(self, paper_arxiv_id: str, paper_section: str, paragraph_id) -> Optional[int]:
        """
        Get the database id for a paragraph given its paper_arxiv_id, paper_section, and paragraph_id.
        Returns the id or None if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id 
                FROM arxiv_paragraphs 
                WHERE paper_arxiv_id = %s 
                  AND paper_section = %s 
                  AND paragraph_id = %s
                LIMIT 1
                """,
                (paper_arxiv_id, paper_section, str(paragraph_id))
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def check_paragraph_exists(self, id: int) -> bool:
        """
        Returns True if a paragraph with the given id exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM arxiv_paragraphs WHERE id = %s LIMIT 1", (id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def get_all_paragraphs(self, is_all_features=True) -> Optional[pd.DataFrame]:
        """
        Get all paragraphs from the database.
        Returns DataFrame or None if empty.
        """
        conn = self._get_connection()
        try:
            query = """
                SELECT id, paragraph_id, content, paper_arxiv_id, paper_section
                FROM arxiv_paragraphs
                ORDER BY id
            """
            df = pd.read_sql(query, conn)
            
            return None if df.empty else df
        finally:
            conn.close()

    # -------------------------
    # Bulk import from CSV
    # -------------------------
    def construct_paragraph_table_from_csv(self, csv_file: str) -> bool:
        """
        Imports rows from a CSV with columns:
            ['paragraph_id', 'content', 'paper_arxiv_id', 'paper_section']
        Ignores any 'id' column; DB assigns SERIAL ids.
        Skips rows that violate the composite uniqueness (via ON CONFLICT DO NOTHING).
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        df = pd.read_csv(csv_file)
        required_cols = ['paragraph_id', 'content', 'paper_arxiv_id', 'paper_section']
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
            # Use execute_values with ON CONFLICT to bulk-insert
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO arxiv_paragraphs (paragraph_id, content, paper_arxiv_id, paper_section)
                VALUES %s
                ON CONFLICT ON CONSTRAINT ux_arxiv_paragraphs_unique DO NOTHING
                """,
                rows,
                page_size=1000
            )
            cur.close()
        finally:
            conn.close()

        print(f"Successfully imported {len(rows)} paragraphs from {csv_file}")
        return True

    def construct_table_from_csv(self, csv_file: str) -> bool:
        """Alias for construct_paragraph_table_from_csv for consistency."""
        return self.construct_paragraph_table_from_csv(csv_file)

    def construct_table_from_json(self, json_file):
        """
        Construct the paragraphs table from an external JSON file.
        
        Args:
            json_file: Path to the JSON file containing paragraph data
            
        Expected JSON format:
            [
                {
                    "paragraph_id": 0,
                    "content": "This paper introduces the Transformer...",
                    "paper_arxiv_id": "1706.03762v7",
                    "paper_section": "introduction"
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
                if 'paragraphs' in json_data:
                    paragraphs_list = json_data['paragraphs']
                else:
                    paragraphs_list = [json_data]
            elif isinstance(json_data, list):
                paragraphs_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not paragraphs_list:
                print("Error: No paragraph data found in JSON file")
                return False
            
            # Convert to list of tuples for bulk insert
            rows = []
            for paragraph in paragraphs_list:
                if 'paragraph_id' not in paragraph or 'content' not in paragraph or 'paper_arxiv_id' not in paragraph or 'paper_section' not in paragraph:
                    print(f"Warning: Skipping paragraph missing required fields: {paragraph}")
                    continue
                    
                rows.append((
                    paragraph['paragraph_id'],
                    paragraph['content'],
                    paragraph['paper_arxiv_id'],
                    paragraph['paper_section']
                ))

            if not rows:
                print("No valid paragraph records to import")
                return False

            conn = self._get_connection()
            try:
                cur = conn.cursor()
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO arxiv_paragraphs (paragraph_id, content, paper_arxiv_id, paper_section)
                    VALUES %s
                    ON CONFLICT ON CONSTRAINT ux_arxiv_paragraphs_unique DO NOTHING
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} paragraphs from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing paragraphs from JSON: {e}")
            return False

    def construct_paragraphs_table_from_api(self, arxiv_ids, dest_dir):
        """
        Construct the paragraphs table from API-generated data.
        Downloads papers, builds graphs, and extracts paragraphs.
        """
        downloaded_paper_ids = []
        md = MultiDownload()
        
        data_dir_path = f"{dest_dir}/output"
        figures_dir_path = f"{dest_dir}/output/images"
        output_dir_path = f"{dest_dir}/output/paragraphs"
        pgp = PaperGraphProcessor(data_dir=data_dir_path, figures_dir=figures_dir_path, output_dir=output_dir_path)

        papers = []
        for arxiv_id in arxiv_ids:
            paper_dir = f"{dest_dir}/{arxiv_id}/{arxiv_id}_metadata.json"

            if not os.path.exists(paper_dir):
                downloaded_paper_ids.append(arxiv_id)

        for arxiv_id in downloaded_paper_ids:
            try:
                md.download_arxiv(input=arxiv_id, input_type="id", output_type="latex", dest_dir=dest_dir)
                print(f"paper with id {arxiv_id} downloaded")
                downloaded_paper_ids.append(arxiv_id)
            except RuntimeError as e:
                print(f"[ERROR] Failed to download {arxiv_id}: {e}")
                continue
        
        for arxiv_id in arxiv_ids:
            # Search if the corresponding paper graph exists
            json_path = f"{dest_dir}/output/{arxiv_id}.json"
            if not os.path.exists(json_path):
                try:
                    # Build corresponding graph
                    md.build_paper_graph(
                        input=arxiv_id,
                        input_type="id",
                        dest_dir=dest_dir
                    )
                except Exception as e:
                    print(f"[Warning] Failed to process papers: {e}")
                    continue
            
            papers.append(arxiv_id)

        paper_paths = []
        # We first build paper node
        # We loop through the provided arxiv ids of paper.
        for arxiv_id in papers:
            paper_paths.append(f"{dest_dir}/output/{arxiv_id}.json")
        pgp.process_papers(paper_paths)

        # Build the paragraphs
        paragraph_path = f"{dest_dir}/output/paragraphs/text_nodes.jsonl"
        with open(paragraph_path) as f:
            data = [json.loads(line) for line in f]
        
        # Use arxiv_id + section name as key
        # Find the smallest paragraph_id generated by knowledge debugger
        # Subtract all paragraph id of the same section (of the same paper) with the smallest one to ensure that order starts with zero
        section_min_paragraph = {}

        for paragraph in data:
            paragraph_id = paragraph.get('id')
            # Extract paragraph_id
            id_number = get_paragraph_num(paragraph_id)
            paper_arxiv_id = paragraph.get('paper_id')
            paper_section = paragraph.get('section')
            if (paper_arxiv_id, paper_section) not in section_min_paragraph:
                section_min_paragraph[(paper_arxiv_id, paper_section)] = int(id_number)
            else:
                section_min_paragraph[(paper_arxiv_id, paper_section)] = min(section_min_paragraph[(paper_arxiv_id, paper_section)], int(id_number))

        for paragraph in data:
            paragraph_id = paragraph.get('id')
            content = paragraph.get('content')
            paper_arxiv_id = paragraph.get('paper_id')
            paper_section = paragraph.get('section')
            id_number = get_paragraph_num(paragraph_id)
            id_zero_based = id_number - section_min_paragraph[(paper_arxiv_id, paper_section)]
            self.insert_paragraph(paragraph_id=id_zero_based, content=content, paper_arxiv_id=paper_arxiv_id, paper_section=paper_section)

