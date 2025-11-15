import psycopg2
import psycopg2.extras
import pandas as pd
import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..arxiv_utils.multi_input.multi_download import MultiDownload
from ..arxiv_utils.graph_constructor.node_processor import NodeConstructor
from ..arxiv_utils.utils import arxiv_id_processor
from .sql_arxiv_tables import SQLArxivTable


class SQLArxivPaperTable:
    def __init__(self, host: str, dbname: str, user: str, password: str, port: str):
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.port = port
        self.autocommit = True
        self.create_paper_tables_table()
        self.sqlat = SQLArxivTable(host=host, dbanme=dbname, user=user, password=password, port=port)

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
    def create_paper_tables_table(self):
        """
        Create the arxiv_paper_tables table with a composite uniqueness constraint.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_paper_tables (
                    paper_arxiv_id VARCHAR(100) NOT NULL,
                    table_id INTEGER NOT NULL
                )
            """)
            # Composite unique index mirrors CSV conflict check
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS ux_arxiv_paper_tables_unique
                ON arxiv_paper_tables (paper_arxiv_id, table_id)
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # Insert
    # -------------------------
    def insert_paper_table(self, paper_arxiv_id, table_id) -> bool:
        """
        Insert a (paper_arxiv_id, table_id) edge.
        Returns True if inserted, False if the pair already exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_paper_tables (paper_arxiv_id, table_id)
                VALUES (%s, %s)
                ON CONFLICT ON CONSTRAINT ux_arxiv_paper_tables_unique DO NOTHING
                """,
                (paper_arxiv_id, table_id)
            )
            inserted = cur.rowcount > 0  # rowcount==1 if inserted, 0 if conflict
            cur.close()
            return inserted
        finally:
            conn.close()

    # -------------------------
    # Query
    # -------------------------
    def get_all_paper_tables(self):
        """Get all paper-table relationships from the database."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT paper_arxiv_id, table_id FROM arxiv_paper_tables"
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_paper_neighboring_tables(self, paper_arxiv_id: str):
        """
        Get all tables for a specific paper.
        Returns list of tuples: (paper_arxiv_id, table_id)
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT paper_arxiv_id, table_id FROM arxiv_paper_tables WHERE paper_arxiv_id = %s",
                (paper_arxiv_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_table_neighboring_papers(self, table_id: int):
        """
        Get all papers that contain a specific table.
        Returns list of tuples: (paper_arxiv_id, table_id)
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT paper_arxiv_id, table_id FROM arxiv_paper_tables WHERE table_id = %s",
                (table_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    # -------------------------
    # Delete
    # -------------------------
    def delete_paper_table_by_id(self, paper_arxiv_id: str, table_id: int) -> bool:
        """
        Delete a specific paper-table relationship.
        Returns True if deleted, False if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paper_tables WHERE paper_arxiv_id = %s AND table_id = %s RETURNING paper_arxiv_id",
                (paper_arxiv_id, table_id)
            )
            deleted = cur.fetchone() is not None
            cur.close()
            return deleted
        finally:
            conn.close()

    def delete_paper_table_by_paper_id(self, paper_arxiv_id: str) -> int:
        """
        Delete all table relationships for a specific paper.
        Returns count of deleted rows.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paper_tables WHERE paper_arxiv_id = %s",
                (paper_arxiv_id,)
            )
            count = cur.rowcount
            cur.close()
            return count
        finally:
            conn.close()

    def delete_paper_table_by_table_id(self, table_id: int) -> int:
        """
        Delete all paper relationships for a specific table.
        Returns count of deleted rows.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paper_tables WHERE table_id = %s",
                (table_id,)
            )
            count = cur.rowcount
            cur.close()
            return count
        finally:
            conn.close()

    # -------------------------
    # Bulk import from CSV
    # -------------------------
    def construct_table_from_csv(self, csv_file: str) -> bool:
        """
        Construct the paper-table relationships from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file
            
        Expected CSV format:
            - Required columns: paper_arxiv_id, table_id
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        try:
            df = pd.read_csv(csv_file)

            required_cols = ['paper_arxiv_id', 'table_id']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"Error: External CSV is missing required columns: {missing_cols}")
                return False

            rows = list(df[required_cols].itertuples(index=False, name=None))
            if not rows:
                print("No rows to import.")
                return True

            conn = self._get_connection()
            try:
                cur = conn.cursor()
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO arxiv_paper_tables (paper_arxiv_id, table_id)
                    VALUES %s
                    ON CONFLICT ON CONSTRAINT ux_arxiv_paper_tables_unique DO NOTHING
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} paper-table relationships from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing paper-table relationships from CSV: {e}")
            return False

    # -------------------------
    # Bulk import from JSON
    # -------------------------
    def construct_table_from_json(self, json_file: str) -> bool:
        """
        Construct the paper-table relationships from an external JSON file.
        
        Args:
            json_file: Path to the JSON file
            
        Expected JSON format:
            [
                {"paper_arxiv_id": "1706.03762v7", "table_id": 1},
                {"paper_arxiv_id": "1706.03762v7", "table_id": 2},
                ...
            ]
            
        Or:
            {
                "paper_tables": [
                    {"paper_arxiv_id": "1706.03762v7", "table_id": 1},
                    ...
                ]
            }
            
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
                if 'paper_tables' in json_data:
                    relations_list = json_data['paper_tables']
                else:
                    relations_list = [json_data]
            elif isinstance(json_data, list):
                relations_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not relations_list:
                print("Error: No paper-table data found in JSON file")
                return False
            
            # Convert to list of tuples for bulk insert
            rows = []
            for relation in relations_list:
                if 'paper_arxiv_id' not in relation or 'table_id' not in relation:
                    print(f"Warning: Skipping relation missing required fields: {relation}")
                    continue
                    
                rows.append((
                    relation['paper_arxiv_id'],
                    relation['table_id']
                ))

            if not rows:
                print("No valid paper-table records to import")
                return False

            conn = self._get_connection()
            try:
                cur = conn.cursor()
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO arxiv_paper_tables (paper_arxiv_id, table_id)
                    VALUES %s
                    ON CONFLICT ON CONSTRAINT ux_arxiv_paper_tables_unique DO NOTHING
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} paper-table relationships from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing paper-table relationships from JSON: {e}")
            return False
        


    def construct_paper_tables_table_from_api(self, arxiv_ids, dest_dir):
        """
        Construct paper-table relationships from ArXiv API.
        
        Args:
            arxiv_ids: List of arxiv IDs to process
            dest_dir: Destination directory for downloads
            tables_table: SQLArxivTable instance to query table IDs
        """
        md = MultiDownload()
        
        # Check if papers already exist in the directory
        downloaded_paper_ids = []
        for arxiv_id in arxiv_ids:
            paper_dir = f"{dest_dir}/{arxiv_id}/{arxiv_id}_metadata.json"
            if not os.path.exists(paper_dir):
                downloaded_paper_ids.append(arxiv_id)

        # Download papers that don't exist
        for arxiv_id in downloaded_paper_ids:
            try:
                md.download_arxiv(input=arxiv_id, input_type="id", output_type="latex", dest_dir=dest_dir)
                print(f"paper with id {arxiv_id} downloaded")
            except RuntimeError as e:
                print(f"[ERROR] Failed to download {arxiv_id}: {e}")
                continue
        
        # Process each arxiv_id to extract paper-table relationships
        for arxiv_id in arxiv_ids:
            json_path = f"{dest_dir}/output/{arxiv_id}.json"
            
            # Build paper graph if it doesn't exist
            if not os.path.exists(json_path):
                try:
                    md.build_paper_graph(
                        input=arxiv_id,
                        input_type="id",
                        dest_dir=dest_dir
                    )
                except Exception as e:
                    print(f"[Warning] Failed to process papers: {e}")
                    continue

            # Extract tables and insert relationships
            try:
                with open(json_path, 'r') as file:
                    file_json = json.load(file)
                    table_jsons = file_json.get('table', [])
                    
                    for table_json in table_jsons:
                        # Get the label from the table
                        label = table_json.get('label')
                        if not label:
                            continue
                        
                        # Get table_id from the tables table using arxiv_id and label
                        table_id = self.sqlat.get_table_id_by_arxiv_id_label(
                            arxiv_id=arxiv_id, 
                            label=label
                        )
                        
                        if table_id is not None:
                            self.insert_paper_table(
                                paper_arxiv_id=arxiv_id, 
                                table_id=table_id
                            )

            except FileNotFoundError:
                print(f"Error: The file at path '{json_path}' was not found.")
                continue
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from '{json_path}'. Check if the file contains valid JSON.")
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                continue