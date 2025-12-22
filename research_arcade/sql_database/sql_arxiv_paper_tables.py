import psycopg2
import psycopg2.extras
import pandas as pd
import json

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Optional


class SQLArxivPaperTable:
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
                    table_id INTEGER NOT NULL,
                    CONSTRAINT ux_arxiv_paper_tables_unique UNIQUE (paper_arxiv_id, table_id)
                )
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
        Get all tables for a given paper.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT paper_arxiv_id, table_id 
                FROM arxiv_paper_tables 
                WHERE paper_arxiv_id = %s
                """,
                (paper_arxiv_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_table_neighboring_papers(self, table_id: int):
        """
        Get all papers that contain a given table.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT paper_arxiv_id, table_id 
                FROM arxiv_paper_tables 
                WHERE table_id = %s
                """,
                (table_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def delete_paper_table_by_id(self, paper_arxiv_id: str, table_id: int) -> bool:
        """
        Delete a specific paper-table relationship.
        Returns True if deleted, False if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM arxiv_paper_tables 
                WHERE paper_arxiv_id = %s AND table_id = %s
                """,
                (paper_arxiv_id, table_id)
            )
            deleted = cur.rowcount > 0
            cur.close()
            return deleted
        finally:
            conn.close()

    def delete_paper_table_by_paper_id(self, paper_arxiv_id: str) -> int:
        """
        Delete all table relationships for a given paper.
        Returns the count of deleted rows.
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
        Delete all paper relationships for a given table.
        Returns the count of deleted rows.
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

    def match_table_id(self, paper_arxiv_id, label) -> Optional[int]:
        """
        Look up table ID by paper_arxiv_id and label from the arxiv_tables table.
        Returns the table id if found, None otherwise.
        
        Args:
            paper_arxiv_id: The arxiv ID of the paper
            label: The label of the table (e.g., 'tab:results')
            
        Returns:
            The table id if found, None otherwise
        """
        if label is None:
            return None
            
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id 
                FROM arxiv_tables 
                WHERE paper_arxiv_id = %s AND label = %s
                LIMIT 1
                """,
                (str(paper_arxiv_id), str(label))
            )
            result = cur.fetchone()
            cur.close()
            return result[0] if result else None
        finally:
            conn.close()

    def construct_table_from_csv(self, csv_file):
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

    def construct_table_from_json(self, json_file):
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
        Construct paper-table relationships from paper graph JSON files.
        
        Args:
            arxiv_ids: List of arxiv IDs to process
            dest_dir: Directory containing paper output files
        """
        for arxiv_id in arxiv_ids:
            json_path = f"{dest_dir}/output/{arxiv_id}.json"

            try:
                with open(json_path, 'r') as file:
                    file_json = json.load(file)
            except FileNotFoundError:
                print(f"Error: The file '{json_path}' was not found.")
                continue
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from '{json_path}'. Check if the file contains valid JSON.")
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                continue

            table_jsons = file_json.get('table', [])
            for table_json in table_jsons:
                caption = table_json.get('caption')
                label = table_json.get('label')
                table = table_json.get('tabular')
                # We don't currently store the table anywhere as a file so the table path is empty
                path = None

                table_id = self.match_table_id(paper_arxiv_id=arxiv_id, label=label)
                
                if table_id is not None:
                    self.insert_paper_table(paper_arxiv_id=arxiv_id, table_id=table_id)