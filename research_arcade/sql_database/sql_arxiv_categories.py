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


class SQLArxivCategory:
    def __init__(self, host: str, dbname: str, user: str, password: str, port: str):
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.port = port
        self.autocommit = True
        self.create_categories_table()

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
    def create_categories_table(self):
        """
        Creates arxiv_categories with UNIQUE(name) to mirror CSV conflict rule.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_categories (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    description TEXT
                )
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # CRUD
    # -------------------------
    def insert_category(self, name: str, description: Optional[str] = None) -> Optional[int]:
        """
        Insert a category; returns generated id or None if name conflicts.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_categories (name, description)
                VALUES (%s, %s)
                ON CONFLICT (name) DO NOTHING
                RETURNING id
                """,
                (name, description)
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def delete_category_by_id(self, id: int) -> bool:
        """
        Delete by id; returns True if a row was deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM arxiv_categories WHERE id = %s RETURNING id", (id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def update_category(self, id: int, name: Optional[str] = None, description: Optional[str] = None) -> bool:
        """
        Partial update by id. Only non-None fields are updated.
        Returns True if a row was updated.
        """
        sets: List[str] = []
        vals: List = []
        if name is not None:
            sets.append("name = %s")
            vals.append(name)
        if description is not None:
            sets.append("description = %s")
            vals.append(description)

        if not sets:
            return False

        sql = f"UPDATE arxiv_categories SET {', '.join(sets)} WHERE id = %s RETURNING id"
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

    def get_category_by_id(self, id: int, return_all: bool = False):
        """
        If return_all=False: returns a single tuple (id, name, description)
        If return_all=True: returns a list of such tuples.
        Returns None if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, name, description FROM arxiv_categories WHERE id = %s",
                (id,)
            )
            rows = cur.fetchall() if return_all else cur.fetchone()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def check_category_exists(self, id: int) -> bool:
        """
        True if a category with the given id exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM arxiv_categories WHERE id = %s LIMIT 1", (id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def get_all_categories(self, is_all_features=True):
        """Get all categories from the database."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, name, description FROM arxiv_categories")
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def construct_category_table_from_api(self, arxiv_ids, dest_dir):
        """
        Construct categories from arXiv API by downloading papers and extracting their categories.
        """
        downloaded_paper_ids = []
        for arxiv_id in arxiv_ids:
            paper_dir = f"{dest_dir}/{arxiv_id}/{arxiv_id}_metadata.json"

            if not os.path.exists(paper_dir):
                downloaded_paper_ids.append(arxiv_id)

        for arxiv_id in downloaded_paper_ids:
            md = MultiDownload()
            try:
                md.download_arxiv(input=arxiv_id, input_type="id", output_type="latex", dest_dir=dest_dir)
                print(f"paper with id {arxiv_id} downloaded")
            except RuntimeError as e:
                print(f"[ERROR] Failed to download {arxiv_id}: {e}")
                continue
        
        for arxiv_id in arxiv_ids:
            try:
                metadata_path = f"{dest_dir}/{arxiv_id}/{arxiv_id}_metadata.json"
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    print(metadata)
                
                required_fields = ['categories']
                if not all(field in metadata for field in required_fields):
                    raise ValueError(f"Missing category for {arxiv_id}")
                
                categories = metadata['categories']
                for category in categories:
                    self.insert_category(name=category)
            except Exception as e:
                print(e)
                print(f"Paper {arxiv_id} does not have category found")

    # -------------------------
    # Bulk import from CSV
    # -------------------------
    def construct_category_table_from_csv(self, csv_file: str) -> bool:
        """
        Imports rows from a CSV with required column: ['name']
        Optional column: ['description']
        Uses ON CONFLICT (name) DO NOTHING to skip duplicates.
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        df = pd.read_csv(csv_file)
        required_cols = ['name']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Error: External CSV is missing required columns: {missing}")
            return False

        if 'description' not in df.columns:
            df['description'] = None

        rows: List[Tuple] = list(df[['name', 'description']].itertuples(index=False, name=None))
        if not rows:
            print("No rows to import.")
            return True

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO arxiv_categories (name, description)
                VALUES %s
                ON CONFLICT (name) DO NOTHING
                """,
                rows,
                page_size=1000
            )
            cur.close()
        finally:
            conn.close()

        print(f"Successfully imported {len(rows)} categories from {csv_file}")
        return True

    def construct_table_from_csv(self, csv_file: str) -> bool:
        """Alias for construct_category_table_from_csv for consistency."""
        return self.construct_category_table_from_csv(csv_file)

    def construct_table_from_json(self, json_file):
        """
        Construct the categories table from an external JSON file.
        
        Args:
            json_file: Path to the JSON file
            
        Expected JSON format:
            [
                {"name": "cs.AI", "description": "Artificial Intelligence"},
                {"name": "cs.LG", "description": "Machine Learning"},
                ...
            ]
            
        Or:
            {
                "categories": [
                    {"name": "cs.AI", "description": "Artificial Intelligence"},
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
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            if isinstance(json_data, dict):
                if 'categories' in json_data:
                    categories_list = json_data['categories']
                else:
                    categories_list = [json_data]
            elif isinstance(json_data, list):
                categories_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not categories_list:
                print("Error: No category data found in JSON file")
                return False
            
            # Convert to list of tuples for bulk insert
            rows = []
            for category in categories_list:
                if 'name' not in category:
                    print(f"Warning: Skipping category record missing required field 'name': {category}")
                    continue
                    
                rows.append((
                    category['name'],
                    category.get('description', None)
                ))

            if not rows:
                print("No valid category records to import")
                return False

            conn = self._get_connection()
            try:
                cur = conn.cursor()
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO arxiv_categories (name, description)
                    VALUES %s
                    ON CONFLICT (name) DO NOTHING
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} categories from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing categories from JSON: {e}")
            return False
        

    def get_category_by_name(self, name: str, return_all: bool = False):
        """
        Get a category by its name.
        - If return_all=False, returns a single tuple (id, name, description)
        - If return_all=True, returns a list of such tuples.
        Returns None if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, name, description FROM arxiv_categories WHERE name = %s",
                (name,)
            )
            rows = cur.fetchall() if return_all else cur.fetchone()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()


    def check_category_exists_by_name(self, name: str) -> bool:
        """
        True if a category with the given name exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM arxiv_categories WHERE name = %s LIMIT 1", (name,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()


    def get_category_id_by_name(self, name: str) -> int:
        """
        Get the internal database id for a category by name.
        Returns None if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id FROM arxiv_categories WHERE name = %s", (name,))
            result = cur.fetchone()
            cur.close()
            return result[0] if result else None
        finally:
            conn.close()


    def delete_category_by_name(self, name: str) -> bool:
        """
        Delete by name; returns True if a row was deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM arxiv_categories WHERE name = %s RETURNING id", (name,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()


    def update_category_by_name(self, name: str, new_name: str = None, description: str = None) -> bool:
        """
        Partial update by name. Only non-None fields are updated.
        Returns True if a row was updated.
        """
        sets = []
        vals = []
        
        if new_name is not None:
            sets.append("name = %s")
            vals.append(new_name)
        if description is not None:
            sets.append("description = %s")
            vals.append(description)

        if not sets:
            return False

        sql = f"UPDATE arxiv_categories SET {', '.join(sets)} WHERE name = %s RETURNING id"
        vals.append(name)

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(sql, tuple(vals))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()