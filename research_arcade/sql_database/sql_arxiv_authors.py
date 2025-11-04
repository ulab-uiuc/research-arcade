import psycopg2
from semanticscholar import SemanticScholar

import os
import json

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
        self.autocommit = port
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

    def construct_category_table_from_api(self, arxiv_ids, dest_dir):
        
        # Check if papers already exists in the directory
        downloaded_paper_ids = []
        for arxiv_id in arxiv_ids:
            paper_dir = f"{dest_dir}/{arxiv_id}/{arxiv_id}_metadata.json"

            if not os.path.exists(paper_dir):
                downloaded_paper_ids.append(arxiv_id)

        for arxiv_id in downloaded_paper_ids:
            md = MultiDownload()
            try:
                md.download_arxiv(input=arxiv_id, input_type = "id", output_type="latex", dest_dir=self.dest_dir)
                print(f"paper with id {arxiv_id} downloaded")
            except RuntimeError as e:
                print(f"[ERROR] Failed to download {arxiv_id}: {e}")
                continue
        
        for arxiv_id in arxiv_ids:
            try:
                metadata_path = f"{dest_dir}/{arxiv_id}/{arxiv_id}_metadata.json"
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)  # Use json.load(), not json.loads()
                    print(metadata)
                
                # Validate required fields
                required_fields = ['categories']
                if not all(field in metadata for field in required_fields):
                    raise ValueError(f"Missing category for {arxiv_id}")
                
                categories = metadata['categories']
                for category in categories:
                    self.insert_category(name=category)
            except Exception as e:
                print(e)
                print(f"Paper {arxiv_id} does not have category found")