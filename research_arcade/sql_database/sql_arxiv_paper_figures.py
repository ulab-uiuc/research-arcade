import psycopg2
import psycopg2.extras
import pandas as pd
import json

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
        self.port = port
        self.autocommit = True
        self.create_paper_figures_table()

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
                    figure_id INTEGER NOT NULL,
                    CONSTRAINT ux_arxiv_paper_figures_unique UNIQUE (paper_arxiv_id, figure_id)
                )
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

    def get_all_paper_figures(self):
        """Get all paper-figure relationships from the database."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT paper_arxiv_id, figure_id FROM arxiv_paper_figures"
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_paper_neighboring_figures(self, paper_arxiv_id: str):
        """
        Get all figures for a given paper.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT paper_arxiv_id, figure_id 
                FROM arxiv_paper_figures 
                WHERE paper_arxiv_id = %s
                """,
                (paper_arxiv_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_figure_neighboring_papers(self, figure_id: int):
        """
        Get all papers that contain a given figure.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT paper_arxiv_id, figure_id 
                FROM arxiv_paper_figures 
                WHERE figure_id = %s
                """,
                (figure_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def delete_paper_figure_by_id(self, paper_arxiv_id: str, figure_id: int) -> bool:
        """
        Delete a specific paper-figure relationship.
        Returns True if deleted, False if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM arxiv_paper_figures 
                WHERE paper_arxiv_id = %s AND figure_id = %s
                """,
                (paper_arxiv_id, figure_id)
            )
            deleted = cur.rowcount > 0
            cur.close()
            return deleted
        finally:
            conn.close()

    def delete_paper_figure_by_paper_id(self, paper_arxiv_id: str) -> int:
        """
        Delete all figure relationships for a given paper.
        Returns the count of deleted rows.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paper_figures WHERE paper_arxiv_id = %s",
                (paper_arxiv_id,)
            )
            count = cur.rowcount
            cur.close()
            return count
        finally:
            conn.close()

    def delete_paper_figure_by_figure_id(self, figure_id: int) -> int:
        """
        Delete all paper relationships for a given figure.
        Returns the count of deleted rows.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paper_figures WHERE figure_id = %s",
                (figure_id,)
            )
            count = cur.rowcount
            cur.close()
            return count
        finally:
            conn.close()

    def construct_table_from_csv(self, csv_file):
        """
        Construct the paper-figure relationships from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file
            
        Expected CSV format:
            - Required columns: paper_arxiv_id, figure_id
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        try:
            df = pd.read_csv(csv_file)

            required_cols = ['paper_arxiv_id', 'figure_id']
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
                    INSERT INTO arxiv_paper_figures (paper_arxiv_id, figure_id)
                    VALUES %s
                    ON CONFLICT ON CONSTRAINT ux_arxiv_paper_figures_unique DO NOTHING
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} paper-figure relationships from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing paper-figure relationships from CSV: {e}")
            return False

    def construct_table_from_json(self, json_file):
        """
        Construct the paper-figure relationships from an external JSON file.
        
        Args:
            json_file: Path to the JSON file
            
        Expected JSON format:
            [
                {"paper_arxiv_id": "1706.03762v7", "figure_id": 1},
                {"paper_arxiv_id": "1706.03762v7", "figure_id": 2},
                ...
            ]
            
        Or:
            {
                "paper_figures": [
                    {"paper_arxiv_id": "1706.03762v7", "figure_id": 1},
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
                if 'paper_figures' in json_data:
                    relations_list = json_data['paper_figures']
                else:
                    relations_list = [json_data]
            elif isinstance(json_data, list):
                relations_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not relations_list:
                print("Error: No paper-figure data found in JSON file")
                return False
            
            # Convert to list of tuples for bulk insert
            rows = []
            for relation in relations_list:
                if 'paper_arxiv_id' not in relation or 'figure_id' not in relation:
                    print(f"Warning: Skipping relation missing required fields: {relation}")
                    continue
                    
                rows.append((
                    relation['paper_arxiv_id'],
                    relation['figure_id']
                ))

            if not rows:
                print("No valid paper-figure records to import")
                return False

            conn = self._get_connection()
            try:
                cur = conn.cursor()
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO arxiv_paper_figures (paper_arxiv_id, figure_id)
                    VALUES %s
                    ON CONFLICT ON CONSTRAINT ux_arxiv_paper_figures_unique DO NOTHING
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} paper-figure relationships from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing paper-figure relationships from JSON: {e}")
            return False