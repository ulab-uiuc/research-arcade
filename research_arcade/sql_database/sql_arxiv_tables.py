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


class SQLArxivTable:
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
    def create_tables_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_tables (
                    id SERIAL PRIMARY KEY,
                    paper_arxiv_id VARCHAR(100) NOT NULL,
                    path VARCHAR(1024),
                    caption TEXT,
                    label VARCHAR(255),
                    table_text TEXT
                )
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # CRUD
    # -------------------------
    def insert_table(self, paper_arxiv_id, path=None, caption=None, label=None, table_text=None) -> Optional[int]:
        """
        Insert a table row; returns generated id (int) or None if not inserted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_tables (paper_arxiv_id, path, caption, label, table_text)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (paper_arxiv_id, path, caption, label, table_text)
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def delete_table_by_id(self, id: int) -> bool:
        """
        Delete by id; returns True if a row was deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM arxiv_tables WHERE id = %s RETURNING id", (id,))
            deleted = cur.fetchone() is not None
            cur.close()
            return deleted
        finally:
            conn.close()

    def update_table(self, id: int, paper_arxiv_id=None, path=None, caption=None, label=None, table_text=None) -> bool:
        """
        Partial update. Only non-None fields are updated.
        Returns True if a row was updated.
        """
        sets = []
        vals = []
        if paper_arxiv_id is not None:
            sets.append("paper_arxiv_id = %s")
            vals.append(paper_arxiv_id)
        if path is not None:
            sets.append("path = %s")
            vals.append(path)
        if caption is not None:
            sets.append("caption = %s")
            vals.append(caption)
        if label is not None:
            sets.append("label = %s")
            vals.append(label)
        if table_text is not None:
            sets.append("table_text = %s")
            vals.append(table_text)

        if not sets:
            return False  # nothing to update

        sql = f"UPDATE arxiv_tables SET {', '.join(sets)} WHERE id = %s RETURNING id"
        vals.append(id)

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(sql, tuple(vals))
            updated = cur.fetchone() is not None
            cur.close()
            return updated
        finally:
            conn.close()

    def get_table_by_id(self, id: int, return_all: bool = False):
        """
        If return_all=False: returns a single tuple
           (id, paper_arxiv_id, path, caption, label, table_text)
        If return_all=True: returns a list of such tuples.
        Returns None if no row found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, paper_arxiv_id, path, caption, label, table_text FROM arxiv_tables WHERE id = %s",
                (id,)
            )
            rows = cur.fetchall() if return_all else cur.fetchone()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_all_tables(self, is_all_features=True):
        """Get all tables from the database."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, paper_arxiv_id, path, caption, label, table_text FROM arxiv_tables"
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def check_table_exists(self, id: int) -> bool:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM arxiv_tables WHERE id = %s LIMIT 1", (id,))
            exists = cur.fetchone() is not None
            cur.close()
            return exists
        finally:
            conn.close()

    # -------------------------
    # Bulk import from CSV
    # -------------------------
    def construct_tables_table_from_csv(self, csv_file: str) -> bool:
        """
        Imports rows from a CSV with columns:
          ['paper_arxiv_id', 'path', 'caption', 'label', 'table_text']
        Ignores any 'id' from the CSV and lets DB assign SERIAL ids.
        Returns True on success, False on validation error or missing file.
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        df = pd.read_csv(csv_file)
        required_cols = ['paper_arxiv_id']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Error: External CSV is missing required columns: {missing}")
            return False

        # Add optional columns if missing
        for col in ['path', 'caption', 'label', 'table_text']:
            if col not in df.columns:
                df[col] = None

        rows: List[Tuple] = list(
            df[['paper_arxiv_id', 'path', 'caption', 'label', 'table_text']].itertuples(index=False, name=None)
        )
        if not rows:
            print("No rows to import.")
            return True

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO arxiv_tables (paper_arxiv_id, path, caption, label, table_text)
                VALUES %s
                """,
                rows,
                page_size=1000
            )
            cur.close()
        finally:
            conn.close()

        print(f"Successfully imported {len(rows)} tables from {csv_file}")
        return True

    def construct_table_from_csv(self, csv_file: str) -> bool:
        """Alias for construct_tables_table_from_csv for consistency."""
        return self.construct_tables_table_from_csv(csv_file)

    def construct_table_from_json(self, json_file):
        """
        Construct the tables table from an external JSON file.
        
        Args:
            json_file: Path to the JSON file containing table data
            
        Expected JSON format:
            [
                {
                    "paper_arxiv_id": "1706.03762v7",
                    "path": "/path/to/table1.tex",
                    "caption": "Model performance comparison",
                    "label": "tab:performance",
                    "table_text": "Table content..."
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
                if 'tables' in json_data:
                    tables_list = json_data['tables']
                else:
                    tables_list = [json_data]
            elif isinstance(json_data, list):
                tables_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not tables_list:
                print("Error: No table data found in JSON file")
                return False
            
            # Convert to list of tuples for bulk insert
            rows = []
            for table in tables_list:
                if 'paper_arxiv_id' not in table:
                    print(f"Warning: Skipping table missing required field 'paper_arxiv_id': {table}")
                    continue
                    
                rows.append((
                    table['paper_arxiv_id'],
                    table.get('path', None),
                    table.get('caption', None),
                    table.get('label', None),
                    table.get('table_text', None)
                ))

            if not rows:
                print("No valid table records to import")
                return False

            conn = self._get_connection()
            try:
                cur = conn.cursor()
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO arxiv_tables (paper_arxiv_id, path, caption, label, table_text)
                    VALUES %s
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} tables from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing tables from JSON: {e}")
            return False

    def construct_tables_table_from_api(self, arxiv_ids, dest_dir):
        # Check if papers already exists in the directory
        md = MultiDownload()
        downloaded_paper_ids = []
        for arxiv_id in arxiv_ids:
            paper_dir = f"{dest_dir}/{arxiv_id}/{arxiv_id}_metadata.json"

            if not os.path.exists(paper_dir):
                downloaded_paper_ids.append(arxiv_id)

        for arxiv_id in downloaded_paper_ids:
            try:
                md.download_arxiv(input=arxiv_id, input_type = "id", output_type="latex", dest_dir=dest_dir)
                print(f"paper with id {arxiv_id} downloaded")
                downloaded_paper_ids.append(arxiv_id)
            except RuntimeError as e:
                print(f"[ERROR] Failed to download {arxiv_id}: {e}")
                continue
        
        for arxiv_id in arxiv_ids:
            # Search if the corresponding paper graph exists

            json_path = f"{dest_dir}/output/{arxiv_id}.json"
            if not os.path.exists(json_path):
                # arxiv_id_graph.append(arxiv_id)
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

            try:
                with open(json_path, 'r') as file:
                    file_json = json.load(file)
                    table_jsons = file_json['table']
                    for table_json in table_jsons:

                        caption = table_json['caption']
                        label = table_json['label']
                        table = table_json['tabular']
                        self.insert_table(paper_arxiv_id=arxiv_id, path=None, caption=caption, label=label, table_text=table)

            except FileNotFoundError:
                print(f"Error: The file '{json_path}' was not found.")
                continue
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from '{json_path}'. Check if the file contains valid JSON.")
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                continue