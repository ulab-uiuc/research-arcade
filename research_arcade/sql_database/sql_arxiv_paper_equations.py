import psycopg2
import psycopg2.extras
import pandas as pd
import json
from typing import Optional

import os
import sys

from sympy import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Optional
from research_arcade.research_arcade.arxiv_utils.get_paragraph_global_id import get_paragraph_global_id_sql

class SQLArxivPaperEquation:
    def __init__(self, host: str, dbname: str, user: str, password: str, port: str):
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.port = port
        self.autocommit = True
        self.create_paper_equations_table()

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
    def create_paper_equations_table(self):
        """
        Create the arxiv_paper_equations table with a composite uniqueness constraint.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_paper_equations (
                    paper_arxiv_id VARCHAR(100) NOT NULL,
                    equation_id INTEGER NOT NULL,
                    CONSTRAINT ux_arxiv_paper_equations_unique UNIQUE (paper_arxiv_id, equation_id)
                )
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # Insert
    # -------------------------
    def insert_paper_equation(self, paper_arxiv_id, equation_id) -> bool:
        """
        Insert a (paper_arxiv_id, equation_id) edge.
        Returns True if inserted, False if the pair already exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_paper_equations (paper_arxiv_id, equation_id)
                VALUES (%s, %s)
                ON CONFLICT ON CONSTRAINT ux_arxiv_paper_equations_unique DO NOTHING
                """,
                (paper_arxiv_id, equation_id)
            )
            inserted = cur.rowcount > 0  # rowcount==1 if inserted, 0 if conflict
            cur.close()
            return inserted
        finally:
            conn.close()

    def get_all_paper_equations(self):
        """Get all paper-equation relationships from the database."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT paper_arxiv_id, equation_id FROM arxiv_paper_equations"
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_paper_neighboring_equations(self, paper_arxiv_id: str) -> Optional[pd.DataFrame]:
        """
        Get all equations for a given paper.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT paper_arxiv_id, equation_id 
                FROM arxiv_paper_equations 
                WHERE paper_arxiv_id = %s
                """,
                (paper_arxiv_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return pd.DataFrame(rows, columns=['paper_arxiv_id', 'equation_id']) if rows else None
        finally:
            conn.close()

    def get_equation_neighboring_papers(self, equation_id: int):
        """
        Get all papers that contain a given equation.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT paper_arxiv_id, equation_id 
                FROM arxiv_paper_equations 
                WHERE equation_id = %s
                """,
                (equation_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def delete_paper_equation_by_id(self, paper_arxiv_id: str, equation_id: int) -> bool:
        """
        Delete a specific paper-equation relationship.
        Returns True if deleted, False if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM arxiv_paper_equations 
                WHERE paper_arxiv_id = %s AND equation_id = %s
                """,
                (paper_arxiv_id, equation_id)
            )
            deleted = cur.rowcount > 0
            cur.close()
            return deleted
        finally:
            conn.close()

    def delete_paper_equation_by_paper_id(self, paper_arxiv_id: str) -> int:
        """
        Delete all equation relationships for a given paper.
        Returns the count of deleted rows.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paper_equations WHERE paper_arxiv_id = %s",
                (paper_arxiv_id,)
            )
            count = cur.rowcount
            cur.close()
            return count
        finally:
            conn.close()

    def delete_paper_equation_by_equation_id(self, equation_id: int) -> int:
        """
        Delete all paper relationships for a given equation.
        Returns the count of deleted rows.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paper_equations WHERE equation_id = %s",
                (equation_id,)
            )
            count = cur.rowcount
            cur.close()
            return count
        finally:
            conn.close()

    def match_equation_id(self, paper_arxiv_id, label) -> Optional[int]:
        """
        Look up equation ID by paper_arxiv_id and label from the arxiv_equations table.
        Returns the equation id if found, None otherwise.
        
        Args:
            paper_arxiv_id: The arxiv ID of the paper
            label: The label of the equation (e.g., 'eq:loss')
            
        Returns:
            The equation id if found, None otherwise
        """
        if label is None:
            return None
            
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id 
                FROM arxiv_equations 
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

    def construct_equation_from_csv(self, csv_file, dest_dir=None):
        """
        Construct the paper-equation relationships from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file
            
        Expected CSV format:
            - Required columns: paper_arxiv_id, equation_id
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        try:
            df = pd.read_csv(csv_file)

            required_cols = ['paper_arxiv_id', 'equation_id']
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
                    INSERT INTO arxiv_paper_equations (paper_arxiv_id, equation_id)
                    VALUES %s
                    ON CONFLICT ON CONSTRAINT ux_arxiv_paper_equations_unique DO NOTHING
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} paper-equation relationships from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing paper-equation relationships from CSV: {e}")
            return False

    def construct_equation_from_json(self, json_file):
        """
        Construct the paper-equation relationships from an external JSON file.
        
        Args:
            json_file: Path to the JSON file
            
        Expected JSON format:
            [
                {"paper_arxiv_id": "1706.03762v7", "equation_id": 1},
                {"paper_arxiv_id": "1706.03762v7", "equation_id": 2},
                ...
            ]
            
        Or:
            {
                "paper_equations": [
                    {"paper_arxiv_id": "1706.03762v7", "equation_id": 1},
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
                if 'paper_equations' in json_data:
                    relations_list = json_data['paper_equations']
                else:
                    relations_list = [json_data]
            elif isinstance(json_data, list):
                relations_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not relations_list:
                print("Error: No paper-equation data found in JSON file")
                return False
            
            # Convert to list of tuples for bulk insert
            rows = []
            for relation in relations_list:
                if 'paper_arxiv_id' not in relation or 'equation_id' not in relation:
                    print(f"Warning: Skipping relation missing required fields: {relation}")
                    continue
                    
                rows.append((
                    relation['paper_arxiv_id'],
                    relation['equation_id']
                ))

            if not rows:
                print("No valid paper-equation records to import")
                return False

            conn = self._get_connection()
            try:
                cur = conn.cursor()
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO arxiv_paper_equations (paper_arxiv_id, equation_id)
                    VALUES %s
                    ON CONFLICT ON CONSTRAINT ux_arxiv_paper_equations_unique DO NOTHING
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} paper-equation relationships from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing paper-equation relationships from JSON: {e}")
            return False

    def construct_paper_equations_table_from_api(self, arxiv_ids, dest_dir):
        """
        Construct paper-equation relationships from paper graph JSON files.
        
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

            equation_jsons = file_json.get('equations', [])
            for equation_text in equation_jsons:
                
                label = ""
                label_match = re.search(r'\\label\{([^}]+)\}', equation_text)
                if label_match:
                    label = f"\\label{{{label_match.group(1)}}}"
                else:
                    print(f"Warning: No label found in equation for paper {arxiv_id}. Skipping.")
                    continue
                
                equation_id = self.match_equation_id(paper_arxiv_id=arxiv_id, label=label)
                
                if equation_id is not None:
                    self.insert_paper_equation(paper_arxiv_id=arxiv_id, equation_id=equation_id)
                else:
                    print(f"Warning: No matching equation found for paper {arxiv_id} with label {label}. Skipping.")
