import psycopg2
import psycopg2.extras
import pandas as pd
import json

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class SQLArxivPaperAuthor:
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
    def create_paper_authors_table(self):
        """
        Create the arxiv_paper_authors table with composite uniqueness on (paper_arxiv_id, author_id).
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_paper_authors (
                    paper_arxiv_id VARCHAR(100) NOT NULL,
                    author_id VARCHAR(100),
                    author_sequence INTEGER,
                    author_name VARCHAR(255),
                    CONSTRAINT ux_arxiv_paper_authors_unique UNIQUE (paper_arxiv_id, author_id)
                )
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # Insert
    # -------------------------
    def insert_paper_author(self, paper_arxiv_id, author_name, author_id=None, author_sequence=None) -> bool:
        """
        Insert a (paper_arxiv_id, author_id, author_sequence, author_name) record.
        Returns True if inserted, False if already exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_paper_authors (paper_arxiv_id, author_id, author_sequence, author_name)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT ON CONSTRAINT ux_arxiv_paper_authors_unique DO NOTHING
                """,
                (paper_arxiv_id, author_id, author_sequence, author_name)
            )
            inserted = cur.rowcount > 0  # 1 if inserted, 0 if conflict
            cur.close()
            return inserted
        finally:
            conn.close()

    def get_all_paper_authors(self, is_all_features=True):
        """Get all paper-author relationships from the database."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT paper_arxiv_id, author_id, author_sequence, author_name FROM arxiv_paper_authors"
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_paper_neighboring_authors(self, paper_arxiv_id: str):
        """
        Get all authors for a given paper, ordered by author_sequence.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT paper_arxiv_id, author_id, author_sequence, author_name
                FROM arxiv_paper_authors 
                WHERE paper_arxiv_id = %s
                ORDER BY author_sequence ASC
                """,
                (paper_arxiv_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_author_neighboring_papers(self, author_id: str):
        """
        Get all papers for a given author.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT paper_arxiv_id, author_id, author_sequence, author_name
                FROM arxiv_paper_authors 
                WHERE author_id = %s
                """,
                (author_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def delete_paper_author_by_id(self, paper_arxiv_id: str, author_id: str) -> bool:
        """
        Delete a specific paper-author relationship.
        Returns True if deleted, False if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM arxiv_paper_authors 
                WHERE paper_arxiv_id = %s AND author_id = %s
                """,
                (paper_arxiv_id, author_id)
            )
            deleted = cur.rowcount > 0
            cur.close()
            return deleted
        finally:
            conn.close()

    def delete_paper_author_by_paper_id(self, paper_arxiv_id: str) -> int:
        """
        Delete all author relationships for a given paper.
        Returns the count of deleted rows.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paper_authors WHERE paper_arxiv_id = %s",
                (paper_arxiv_id,)
            )
            count = cur.rowcount
            cur.close()
            return count
        finally:
            conn.close()

    def delete_paper_author_by_author_id(self, author_id: str) -> int:
        """
        Delete all paper relationships for a given author.
        Returns the count of deleted rows.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_paper_authors WHERE author_id = %s",
                (author_id,)
            )
            count = cur.rowcount
            cur.close()
            return count
        finally:
            conn.close()

    def construct_table_from_csv(self, csv_file):
        """
        Construct the paper-author relationships from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file
            
        Expected CSV format:
            - Required columns: paper_arxiv_id, author_id, author_sequence
            - Optional columns: author_name
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        try:
            df = pd.read_csv(csv_file)

            required_cols = ['paper_arxiv_id', 'author_id', 'author_sequence']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"Error: External CSV is missing required columns: {missing_cols}")
                return False

            # Add optional column if missing
            if 'author_name' not in df.columns:
                df['author_name'] = None

            rows = list(df[['paper_arxiv_id', 'author_id', 'author_sequence', 'author_name']].itertuples(index=False, name=None))
            if not rows:
                print("No rows to import.")
                return True

            conn = self._get_connection()
            try:
                cur = conn.cursor()
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO arxiv_paper_authors (paper_arxiv_id, author_id, author_sequence, author_name)
                    VALUES %s
                    ON CONFLICT ON CONSTRAINT ux_arxiv_paper_authors_unique DO NOTHING
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} paper-author relationships from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing paper-author relationships from CSV: {e}")
            return False

    def construct_table_from_json(self, json_file):
        """
        Construct the paper-author relationships from an external JSON file.
        
        Args:
            json_file: Path to the JSON file
            
        Expected JSON format:
            [
                {
                    "paper_arxiv_id": "1706.03762v7",
                    "author_id": "12345",
                    "author_sequence": 1,
                    "author_name": "John Doe"
                },
                ...
            ]
            
        Or:
            {
                "paper_authors": [
                    {
                        "paper_arxiv_id": "1706.03762v7",
                        "author_id": "12345",
                        "author_sequence": 1,
                        "author_name": "John Doe"
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
                if 'paper_authors' in json_data:
                    relations_list = json_data['paper_authors']
                else:
                    relations_list = [json_data]
            elif isinstance(json_data, list):
                relations_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not relations_list:
                print("Error: No paper-author data found in JSON file")
                return False
            
            # Convert to list of tuples for bulk insert
            rows = []
            for relation in relations_list:
                if 'paper_arxiv_id' not in relation or 'author_id' not in relation or 'author_sequence' not in relation:
                    print(f"Warning: Skipping relation missing required fields: {relation}")
                    continue
                    
                rows.append((
                    relation['paper_arxiv_id'],
                    relation['author_id'],
                    relation['author_sequence'],
                    relation.get('author_name', None)
                ))

            if not rows:
                print("No valid paper-author records to import")
                return False

            conn = self._get_connection()
            try:
                cur = conn.cursor()
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO arxiv_paper_authors (paper_arxiv_id, author_id, author_sequence, author_name)
                    VALUES %s
                    ON CONFLICT ON CONSTRAINT ux_arxiv_paper_authors_unique DO NOTHING
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} paper-author relationships from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing paper-author relationships from JSON: {e}")
            return False

    def construct_paper_authors_table_from_api(self, arxiv_ids, dest_dir):
        """
        Build paper-author relationships, linking to existing author IDs.
        
        Args:
            arxiv_ids: List of arxiv IDs to process
            dest_dir: Directory containing paper metadata
        """
        for arxiv_id in arxiv_ids:
            metadata_path = f"{dest_dir}/{arxiv_id}/{arxiv_id}_metadata.json"

            try:
                with open(metadata_path, 'r') as file:
                    metadata_json = json.load(file)
                    authors = metadata_json['authors']
                    
                    for i, author_name in enumerate(authors, start=1):
                        # Look up author_id by name from the authors table
                        author_id = self._lookup_author_id_by_name(author_name)
                
                        self.insert_paper_author(
                            paper_arxiv_id=arxiv_id, 
                            author_name=author_name, 
                            author_id=author_id,
                            author_sequence=i
                        )
            except FileNotFoundError:
                print(f"Error: Metadata file not found for {arxiv_id}")
                continue
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from metadata for {arxiv_id}")
                continue
            except Exception as e:
                print(f"An unexpected error occurred for {arxiv_id}: {e}")
                continue

    def _lookup_author_id_by_name(self, author_name: str):
        """
        Look up author_id (semantic_scholar_id) by name from the authors table.
        Returns the semantic_scholar_id if found, None otherwise.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT semantic_scholar_id 
                FROM authors 
                WHERE name = %s
                LIMIT 1
                """,
                (author_name,)
            )
            result = cur.fetchone()
            cur.close()
            return result[0] if result else None
        finally:
            conn.close()