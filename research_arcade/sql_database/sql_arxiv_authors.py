import psycopg2
import psycopg2.extras
from semanticscholar import SemanticScholar

import os
import json
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..arxiv_utils.multi_input.multi_download import MultiDownload
from ..arxiv_utils.graph_constructor.node_processor import NodeConstructor
from ..arxiv_utils.utils import arxiv_id_processor

class SQLArxivAuthors:
    def __init__(self, host: str, dbname: str, user: str, password: str, port: str):
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.port = port
        self.autocommit = True
        self.create_authors_table()

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

    def create_authors_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS authors (
                id SERIAL PRIMARY KEY,
                semantic_scholar_id VARCHAR(100) UNIQUE,
                name VARCHAR(255) NOT NULL,
                homepage VARCHAR(255)
            )
            """)
            cur.close()
        finally:
            conn.close()

    def insert_author(self, semantic_scholar_id, name, homepage=None):
        """Insert an author. Returns the generated author id (or None if conflict)."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO authors (semantic_scholar_id, name, homepage)
            VALUES (%s, %s, %s)
            ON CONFLICT (semantic_scholar_id) DO NOTHING
            RETURNING id
            """
            cur.execute(sql, (semantic_scholar_id, name, homepage))
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def delete_author_by_id(self, id: int) -> bool:
        """Delete an author by its id. Returns True if deleted, False if not found."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM authors WHERE id = %s RETURNING id", (id,))
            deleted = cur.fetchone() is not None
            cur.close()
            return deleted
        finally:
            conn.close()

    def update_author(self, id: int, semantic_scholar_id=None, name=None, homepage=None) -> bool:
        """
        Update an author by id. Returns True if updated, False if not found or no fields provided.
        """
        fields = []
        values = []
        if semantic_scholar_id is not None:
            fields.append("semantic_scholar_id = %s")
            values.append(semantic_scholar_id)
        if name is not None:
            fields.append("name = %s")
            values.append(name)
        if homepage is not None:
            fields.append("homepage = %s")
            values.append(homepage)

        if not fields:
            return False  # Nothing to update

        sql = f"UPDATE authors SET {', '.join(fields)} WHERE id = %s RETURNING id"
        values.append(id)

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(sql, tuple(values))
            updated = cur.fetchone() is not None
            cur.close()
            return updated
        finally:
            conn.close()

    def get_author_by_id(self, id: int, return_all=False):
        """
        Get an author by its id.
        - If return_all=False, returns a single row tuple (id, semantic_scholar_id, name, homepage).
        - If return_all=True, returns all matching rows as a list of tuples.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, semantic_scholar_id, name, homepage FROM authors WHERE id = %s",
                (id,),
            )
            res = cur.fetchall() if return_all else cur.fetchone()
            cur.close()
            return res if res else None
        finally:
            conn.close()

    def check_author_exists(self, id: int) -> bool:
        """Check if an author exists by its id."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM authors WHERE id = %s LIMIT 1", (id,))
            exists = cur.fetchone() is not None
            cur.close()
            return exists
        finally:
            conn.close()

    def get_all_authors(self, is_all_features=True):
        """Get all authors from the database."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, semantic_scholar_id, name, homepage FROM authors")
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def construct_authors_table_from_api(self, arxiv_ids, dest_dir):
        """
        Given arxiv ids, find the semantic scholar ids and pages of the authors
        """
        # Search for authors online
        sch = SemanticScholar()
        for arxiv_id in arxiv_ids:
            base_arxiv_id, version = arxiv_id_processor(arxiv_id=arxiv_id)
            print(f"base_arxiv_id: {base_arxiv_id}")
            try:
                paper_sch = sch.get_paper(f"ARXIV:{base_arxiv_id}")
                authors = paper_sch.authors
                for author in authors:
                    semantic_scholar_id = author.authorId
                    author_r = sch.get_author(semantic_scholar_id)
                    name = author_r.name
                    url = author_r.url
                
                    self.insert_author(semantic_scholar_id=semantic_scholar_id, name=name, homepage=url)
            except Exception as e:
                print(f"Paper with arxiv id {base_arxiv_id} not found on semantic scholar: {e}")
                continue

    def construct_table_from_csv(self, csv_file):
        """
        Construct the authors table from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file
            
        Expected CSV format:
            - Required columns: semantic_scholar_id, name
            - Optional columns: homepage
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False
    
        try:
            df = pd.read_csv(csv_file)

            required_cols = ['semantic_scholar_id', 'name']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"Error: External CSV is missing required columns: {missing_cols}")
                return False

            # Add optional columns if they don't exist
            if 'homepage' not in df.columns:
                df['homepage'] = None

            # Prepare rows for bulk insert
            rows = list(df[['semantic_scholar_id', 'name', 'homepage']].itertuples(index=False, name=None))

            if not rows:
                print("No rows to import.")
                return True

            conn = self._get_connection()
            try:
                cur = conn.cursor()
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO authors (semantic_scholar_id, name, homepage)
                    VALUES %s
                    ON CONFLICT (semantic_scholar_id) DO NOTHING
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} authors from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing authors from CSV: {e}")
            return False

    def construct_table_from_json(self, json_file):
        """
        Construct the authors table from an external JSON file.
        
        Args:
            json_file: Path to the JSON file containing author data
            
        Expected JSON format (list of objects):
            [
                {
                    "semantic_scholar_id": "123456",
                    "name": "John Doe",
                    "homepage": "https://example.com"  // optional
                },
                ...
            ]
            
        Or (single object with authors array):
            {
                "authors": [
                    {
                        "semantic_scholar_id": "123456",
                        "name": "John Doe",
                        "homepage": "https://example.com"
                    },
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
                # If it's a dict, look for an 'authors' key
                if 'authors' in json_data:
                    authors_list = json_data['authors']
                else:
                    # Treat the dict as a single author record
                    authors_list = [json_data]
            elif isinstance(json_data, list):
                authors_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not authors_list:
                print("Error: No author data found in JSON file")
                return False
            
            # Convert to list of tuples for bulk insert
            rows = []
            for author in authors_list:
                if 'semantic_scholar_id' not in author or 'name' not in author:
                    print(f"Warning: Skipping author record missing required fields: {author}")
                    continue
                    
                rows.append((
                    author['semantic_scholar_id'],
                    author['name'],
                    author.get('homepage', None)
                ))

            if not rows:
                print("No valid author records to import")
                return False

            conn = self._get_connection()
            try:
                cur = conn.cursor()
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO authors (semantic_scholar_id, name, homepage)
                    VALUES %s
                    ON CONFLICT (semantic_scholar_id) DO NOTHING
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} authors from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing authors from JSON: {e}")
            return False


    def get_author_by_semantic_scholar_id(self, semantic_scholar_id: str, return_all: bool = False):
        """
        Get an author by its semantic_scholar_id.
        - If return_all=False, returns a single row tuple (id, semantic_scholar_id, name, homepage).
        - If return_all=True, returns all matching rows as a list of tuples.
        Returns None if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, semantic_scholar_id, name, homepage FROM authors WHERE semantic_scholar_id = %s",
                (semantic_scholar_id,),
            )
            res = cur.fetchall() if return_all else cur.fetchone()
            cur.close()
            return res if res else None
        finally:
            conn.close()


    def check_author_exists_by_semantic_scholar_id(self, semantic_scholar_id: str) -> bool:
        """Check if an author exists by its semantic_scholar_id."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM authors WHERE semantic_scholar_id = %s LIMIT 1", (semantic_scholar_id,))
            exists = cur.fetchone() is not None
            cur.close()
            return exists
        finally:
            conn.close()


    def get_author_id_by_semantic_scholar_id(self, semantic_scholar_id: str) -> int:
        """
        Get the internal database id for an author by semantic_scholar_id.
        Returns None if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id FROM authors WHERE semantic_scholar_id = %s", (semantic_scholar_id,))
            result = cur.fetchone()
            cur.close()
            return result[0] if result else None
        finally:
            conn.close()


    def delete_author_by_semantic_scholar_id(self, semantic_scholar_id: str) -> bool:
        """Delete an author by its semantic_scholar_id. Returns True if deleted, False if not found."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM authors WHERE semantic_scholar_id = %s RETURNING id", (semantic_scholar_id,))
            deleted = cur.fetchone() is not None
            cur.close()
            return deleted
        finally:
            conn.close()


    def update_author_by_semantic_scholar_id(self, semantic_scholar_id: str, name=None, homepage=None) -> bool:
        """
        Update an author by semantic_scholar_id. Returns True if updated, False if not found or no fields provided.
        """
        fields = []
        values = []
        
        if name is not None:
            fields.append("name = %s")
            values.append(name)
        if homepage is not None:
            fields.append("homepage = %s")
            values.append(homepage)

        if not fields:
            return False  # Nothing to update

        sql = f"UPDATE authors SET {', '.join(fields)} WHERE semantic_scholar_id = %s RETURNING id"
        values.append(semantic_scholar_id)

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(sql, tuple(values))
            updated = cur.fetchone() is not None
            cur.close()
            return updated
        finally:
            conn.close()
