import os
from typing import Optional, List, Tuple
import psycopg2
import psycopg2.extras
import pandas as pd


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..arxiv_utils.multi_input.multi_download import MultiDownload
from ..arxiv_utils.graph_constructor.node_processor import NodeConstructor
from ..arxiv_utils.utils import arxiv_id_processor
from ..arxiv_utils.utils import arxiv_id_processor, figure_iteration_recursive
import json


class SQLArxivFigure:
    def __init__(self, host: str, dbname: str, user: str, password: str, port: str):
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.port = port
        self.autocommit = True
        self.create_figures_table()

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
    def create_figures_table(self):
        """
        Creates the arxiv_figures table.
        We add a partial UNIQUE index on name (WHERE name IS NOT NULL) to match the CSV
        behavior where we disallow duplicate non-null names.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_figures (
                    id SERIAL PRIMARY KEY,
                    paper_arxiv_id VARCHAR(100) NOT NULL,
                    path VARCHAR(1024),
                    caption TEXT,
                    label VARCHAR(255),
                    name VARCHAR(255)
                )
            """)
            # Partial unique: only enforce uniqueness when name IS NOT NULL
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS ux_arxiv_figures_name_notnull
                ON arxiv_figures (name)
                WHERE name IS NOT NULL
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # CRUD
    # -------------------------
    def insert_figure(self, paper_arxiv_id, path, caption=None, label=None, name=None) -> Optional[int]:
        """
        Insert a figure row; returns generated id or None on (name) conflict (when name is not NULL).
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_figures (paper_arxiv_id, path, caption, label, name)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (name) WHERE name IS NOT NULL DO NOTHING
                RETURNING id
                """,
                (paper_arxiv_id, path, caption, label, name)
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def delete_figure_by_id(self, id: int) -> bool:
        """
        Delete by id; returns True if a row was deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM arxiv_figures WHERE id = %s RETURNING id", (id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def update_figure(self, id: int, paper_arxiv_id=None, path=None, caption=None, label=None, name=None) -> bool:
        """
        Partial update by id. Only non-None fields are updated.
        Returns True if a row was updated.
        """
        sets: List[str] = []
        vals: List = []

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
        if name is not None:
            sets.append("name = %s")
            vals.append(name)

        if not sets:
            return False

        sql = f"UPDATE arxiv_figures SET {', '.join(sets)} WHERE id = %s RETURNING id"
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

    def get_figure_by_id(self, id: int, return_all: bool = False):
        """
        If return_all=False: returns a single tuple
           (id, paper_arxiv_id, path, caption, label, name)
        If return_all=True: returns a list of such tuples.
        Returns None if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, paper_arxiv_id, path, caption, label, name "
                "FROM arxiv_figures WHERE id = %s",
                (id,)
            )
            rows = cur.fetchall() if return_all else cur.fetchone()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def check_figure_exists(self, id: int) -> bool:
        """
        Returns True if a figure with the given id exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM arxiv_figures WHERE id = %s LIMIT 1", (id,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def get_all_figures(self, is_all_features=True):
        """Get all figures from the database."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, paper_arxiv_id, path, caption, label, name FROM arxiv_figures"
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    # -------------------------
    # Bulk import from CSV
    # -------------------------
    def construct_figure_table_from_csv(self, csv_file: str) -> bool:
        """
        Imports rows from a CSV with columns:
          ['paper_arxiv_id', 'path', 'caption', 'label', 'name']
        Ignores any 'id' column; DB assigns SERIAL ids.
        Skips conflicts on non-null 'name'.
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        df = pd.read_csv(csv_file)
        required_cols = ['paper_arxiv_id', 'path']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Error: External CSV is missing required columns: {missing}")
            return False

        # Add optional columns if missing
        for col in ['caption', 'label', 'name']:
            if col not in df.columns:
                df[col] = None

        rows: List[Tuple] = list(df[['paper_arxiv_id', 'path', 'caption', 'label', 'name']].itertuples(index=False, name=None))
        if not rows:
            print("No rows to import.")
            return True

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            # Use upsert-like bulk insert; conflict only triggers if name IS NOT NULL and duplicates.
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO arxiv_figures (paper_arxiv_id, path, caption, label, name)
                VALUES %s
                ON CONFLICT ON CONSTRAINT ux_arxiv_figures_name_notnull DO NOTHING
                """,
                rows,
                page_size=1000
            )
            cur.close()
        finally:
            conn.close()

        print(f"Successfully imported {len(rows)} figures from {csv_file}")
        return True

    def construct_table_from_csv(self, csv_file: str) -> bool:
        """Alias for construct_figure_table_from_csv for consistency."""
        return self.construct_figure_table_from_csv(csv_file)

    def construct_table_from_json(self, json_file):
        """
        Construct the figures table from an external JSON file.
        
        Args:
            json_file: Path to the JSON file containing figure data
            
        Expected JSON format:
            [
                {
                    "paper_arxiv_id": "1706.03762v7",
                    "path": "/path/to/figure1.png",
                    "caption": "Architecture diagram",
                    "label": "fig:architecture",
                    "name": "figure1"
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
                if 'figures' in json_data:
                    figures_list = json_data['figures']
                else:
                    figures_list = [json_data]
            elif isinstance(json_data, list):
                figures_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not figures_list:
                print("Error: No figure data found in JSON file")
                return False
            
            # Convert to list of tuples for bulk insert
            rows = []
            for figure in figures_list:
                if 'paper_arxiv_id' not in figure or 'path' not in figure:
                    print(f"Warning: Skipping figure missing required fields: {figure}")
                    continue
                    
                rows.append((
                    figure['paper_arxiv_id'],
                    figure['path'],
                    figure.get('caption', None),
                    figure.get('label', None),
                    figure.get('name', None)
                ))

            if not rows:
                print("No valid figure records to import")
                return False

            conn = self._get_connection()
            try:
                cur = conn.cursor()
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO arxiv_figures (paper_arxiv_id, path, caption, label, name)
                    VALUES %s
                    ON CONFLICT ON CONSTRAINT ux_arxiv_figures_name_notnull DO NOTHING
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} figures from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing figures from JSON: {e}")
            return False

    def construct_figures_table_from_api(self, arxiv_ids, dest_dir):

        # Check if papers already exists in the directory
        downloaded_paper_ids = []
        for arxiv_id in arxiv_ids:
            paper_dir = f"{dest_dir}/{arxiv_id}/{arxiv_id}_metadata.json"

            if not os.path.exists(paper_dir):
                downloaded_paper_ids.append(arxiv_id)

        for arxiv_id in downloaded_paper_ids:
            md = MultiDownload()
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
                    md = MultiDownload()
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
                figure_jsons = file_json['figure']
                for figure_json in figure_jsons:

                    figures = figure_iteration_recursive(figure_json=figure_json)
                    for figure in figures:
                        path, caption, label = figure
                        self.insert_figure(paper_arxiv_id=arxiv_id, path=path, caption=caption,label=label)

            except FileNotFoundError:
                print(f"Error: The file '{json_path}' was not found.")
                continue
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from '{json_path}'. Check if the file contains valid JSON.")
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                continue



    def get_figure_by_name(self, name: str, return_all: bool = False):
        """
        Get a figure by its name (when name is not NULL).
        - If return_all=False, returns a single tuple
        (id, paper_arxiv_id, path, caption, label, name)
        - If return_all=True, returns a list of such tuples.
        Returns None if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, paper_arxiv_id, path, caption, label, name "
                "FROM arxiv_figures WHERE name = %s",
                (name,)
            )
            rows = cur.fetchall() if return_all else cur.fetchone()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()


    def check_figure_exists_by_name(self, name: str) -> bool:
        """
        Returns True if a figure with the given name exists.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM arxiv_figures WHERE name = %s LIMIT 1", (name,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()


    def get_figure_id_by_arxiv_id_label(self, paper_arxiv_id, label):
        """
        Get the internal database id for a figure by name.
        Returns None if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id FROM arxiv_figures WHERE label = %s AND paper_arxiv_id = %s", (label, paper_arxiv_id))
            result = cur.fetchone()
            cur.close()
            return result[0] if result else None
        finally:
            conn.close()


    def delete_figure_by_name(self, name: str) -> bool:
        """
        Delete by name; returns True if a row was deleted.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM arxiv_figures WHERE name = %s RETURNING id", (name,))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()


    def get_figures_by_paper(self, paper_arxiv_id: str):
        """
        Get all figures for a specific paper.
        Returns a list of tuples (id, paper_arxiv_id, path, caption, label, name).
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, paper_arxiv_id, path, caption, label, name "
                "FROM arxiv_figures WHERE paper_arxiv_id = %s",
                (paper_arxiv_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()


    def delete_figures_by_paper(self, paper_arxiv_id: str) -> int:
        """
        Delete all figures for a specific paper.
        Returns the count of deleted rows.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM arxiv_figures WHERE paper_arxiv_id = %s",
                (paper_arxiv_id,)
            )
            count = cur.rowcount
            cur.close()
            return count
        finally:
            conn.close()