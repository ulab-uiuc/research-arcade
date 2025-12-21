import psycopg2
import psycopg2.extras
import pandas as pd
import json

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class SQLArxivPaperCategory:
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
    def create_paper_category_table(self):
        """
        Create the arxiv_paper_category table with a composite uniqueness constraint.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_paper_category (
                    paper_arxiv_id VARCHAR(100) NOT NULL,
                    category_id VARCHAR(100) NOT NULL
                )
            """)
            # Composite unique index for preventing duplicates
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS ux_arxiv_paper_category_unique
                ON arxiv_paper_category (paper_arxiv_id, category_id)
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # Insert
    # -------------------------
    def insert_paper_category(self, paper_arxiv_id, category_id) -> bool:
        """
        Insert a (paper_arxiv_id, category_id) mapping.
        Returns True if inserted, False if it already exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_paper_category (paper_arxiv_id, category_id)
                VALUES (%s, %s)
                ON CONFLICT ON CONSTRAINT ux_arxiv_paper_category_unique DO NOTHING
                """,
                (paper_arxiv_id, category_id)
            )
            inserted = cur.rowcount > 0  # rowcount == 1 if inserted, 0 if skipped due to conflict
            cur.close()
            return inserted
        finally:
            conn.close()

    def get_all_paper_categories(self):
        """Get all paper-category relationships from the database."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT paper_arxiv_id, category_id FROM arxiv_paper_category"
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_paper_neighboring_categories(self, paper_arxiv_id: str):
        """
        Get all categories for a given paper.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT paper_arxiv_id, category_id 
                FROM arxiv_paper_category 
                WHERE paper_arxiv_id = %s
                """,
                (paper_arxiv_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_category_neighboring_papers(self, category_id: str):
        """
        Get all papers for a given category.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT paper_arxiv_id, category_id 
                FROM arxiv_paper_category 
                WHERE category_id = %s
                """,
                (category_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def delete_paper_category_by_id(self, paper_arxiv_id: str, category_id: str) -> bool:
        """
        Delete a specific paper-category relationship.
        Returns True if deleted, False if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM arxiv_paper_category 
                WHERE paper_arxiv_id = %s AND category_id = %s
                """,
                (paper_arxiv_id, category_id)
            )
            deleted = cur.rowcount > 0
            cur.close()
            return deleted
        finally:
            conn.close()

    def delete_paper_category_by_paper_id(self, paper_arxiv_id: str) -> int:
        """
        Delete all category relationships for a given paper.
        Returns the count of deleted rows.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paper_category WHERE paper_arxiv_id = %s",
                (paper_arxiv_id,)
            )
            count = cur.rowcount
            cur.close()
            return count
        finally:
            conn.close()

    def delete_paper_category_by_category_id(self, category_id: str) -> int:
        """
        Delete all paper relationships for a given category.
        Returns the count of deleted rows.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paper_category WHERE category_id = %s",
                (category_id,)
            )
            count = cur.rowcount
            cur.close()
            return count
        finally:
            conn.close()

    def construct_table_from_csv(self, csv_file):
        """
        Construct the paper-category relationships from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file
            
        Expected CSV format:
            - Required columns: paper_arxiv_id, category_id
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        try:
            df = pd.read_csv(csv_file)

            required_cols = ['paper_arxiv_id', 'category_id']
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
                    INSERT INTO arxiv_paper_category (paper_arxiv_id, category_id)
                    VALUES %s
                    ON CONFLICT ON CONSTRAINT ux_arxiv_paper_category_unique DO NOTHING
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} paper-category relationships from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing paper-category relationships from CSV: {e}")
            return False

    def construct_table_from_json(self, json_file):
        """
        Construct the paper-category relationships from an external JSON file.
        
        Args:
            json_file: Path to the JSON file
            
        Expected JSON format:
            [
                {"paper_arxiv_id": "1706.03762v7", "category_id": "cs.AI"},
                {"paper_arxiv_id": "1706.03762v7", "category_id": "cs.LG"},
                ...
            ]
            
        Or:
            {
                "paper_categories": [
                    {"paper_arxiv_id": "1706.03762v7", "category_id": "cs.AI"},
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
                if 'paper_categories' in json_data:
                    relations_list = json_data['paper_categories']
                else:
                    relations_list = [json_data]
            elif isinstance(json_data, list):
                relations_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not relations_list:
                print("Error: No paper-category data found in JSON file")
                return False
            
            # Convert to list of tuples for bulk insert
            rows = []
            for relation in relations_list:
                if 'paper_arxiv_id' not in relation or 'category_id' not in relation:
                    print(f"Warning: Skipping relation missing required fields: {relation}")
                    continue
                    
                rows.append((
                    relation['paper_arxiv_id'],
                    relation['category_id']
                ))

            if not rows:
                print("No valid paper-category records to import")
                return False

            conn = self._get_connection()
            try:
                cur = conn.cursor()
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO arxiv_paper_category (paper_arxiv_id, category_id)
                    VALUES %s
                    ON CONFLICT ON CONSTRAINT ux_arxiv_paper_category_unique DO NOTHING
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} paper-category relationships from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing paper-category relationships from JSON: {e}")
            return False



    def construct_paper_category_table_from_api(self, arxiv_ids, dest_dir):
        # The same logic, that we first open the file, then create the corresponding stuff.

        # In fact, we only need to add paper category
        # Open the metadata

        for arxiv_id in arxiv_ids:
            metadata_path = f"{dest_dir}/{arxiv_id}/{arxiv_id}_metadata.json"

            with open(metadata_path, 'r') as file:
                metadata_json = json.load(file)
                categories = metadata_json['categories']
                for category in categories:
                    self.insert_paper_category_by_name(paper_arxiv_id=arxiv_id, category_name=category)
