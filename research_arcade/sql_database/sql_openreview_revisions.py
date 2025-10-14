from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import psycopg2
from psycopg2.extras import Json

class SQLOpenReviewRevisions:
    def __init__(self, host: str = "localhost", dbname: str = "iclr_openreview_database", user: str = "jingjunx", password: str = "", port: str = "5432"):
        # Store connection and cursor for reuse
        self.conn = psycopg2.connect(
            host=host, dbname=dbname,
            user=user, password=password, port=port
        )
        # Enable autocommit
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        self.openreview_crawler = OpenReviewCrawler()
        self.create_revisions_table()
    
    def create_revisions_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS revisions (
            id SERIAL UNIQUE,
            venue TEXT,
            original_openreview_id VARCHAR(255),
            revision_openreview_id VARCHAR(255),
            content JSONB,
            time TEXT,
            PRIMARY KEY (venue, revision_openreview_id)
        );
        """
        self.cur.execute(create_table_sql)
        print("Table 'revisions' created successfully.")
    
    def insert_revision(self, venue: str, original_openreview_id: str, 
                        revision_openreview_id: str, content: dict, time: str) -> None | tuple:
        insert_sql = """
        INSERT INTO revisions (venue, original_openreview_id, revision_openreview_id, content, time)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (venue, revision_openreview_id) DO NOTHING
        RETURNING (venue, revision_openreview_id);
        """
        # clean revisions
        cleaned_revision_content = self._clean_json_content(content)

        # Execute the insertion query
        self.cur.execute(insert_sql, (venue, original_openreview_id, revision_openreview_id, Json(cleaned_revision_content), time))
        
        # Get the inserted paper's openreview id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
    
    def delete_revision_by_id(self, revision_openreview_id: str) -> None | pd.DataFrame:
        # search for the row based on primary key
        select_sql = """
        SELECT venue, original_openreview_id, revision_openreview_id, content, time
        FROM revisions WHERE revision_openreview_id = %s;
        """
        self.cur.execute(select_sql, (revision_openreview_id,))
        row = self.cur.fetchall()

        if row:
            columns = ['venue', 'original_openreview_id', 
                       'revision_openreview_id', 'content', 'time']
            # revision_dict = dict(zip(columns, row))
            revision_df = pd.DataFrame(row, columns=columns)

            delete_sql = """
            DELETE FROM revisions WHERE revision_openreview_id = %s;
            """
            self.cur.execute(delete_sql, (revision_openreview_id,))
            self.conn.commit()

            print(f"Revision with revision_openreview_id {revision_openreview_id} deleted successfully.")
            return revision_df
        else:
            print(f"No revision found with revision_openreview_id {revision_openreview_id}.")
            return None
        
    def delete_revisions_by_venue(self, venue: str) -> None | pd.DataFrame:
        # search for the row based on primary key
        select_sql = """
        SELECT venue, original_openreview_id, revision_openreview_id, content, time
        FROM revisions WHERE venue = %s;
        """
        self.cur.execute(select_sql, (venue,))
        row = self.cur.fetchall()

        if row:
            columns = ['venue', 'original_openreview_id', 
                       'revision_openreview_id', 'content', 'time']
            # revision_dict = dict(zip(columns, row))
            revision_df = pd.DataFrame(row, columns=columns)

            delete_sql = """
            DELETE FROM revisions WHERE venue = %s;
            """
            self.cur.execute(delete_sql, (venue,))
            self.conn.commit()

            print(f"All revisions in venue {venue} deleted successfully.")
            return revision_df
        else:
            print(f"No revisions found in venue {venue}.")
            return None
        
    def update_revision(self, venue: str, original_openreview_id: str, 
                        revision_openreview_id: str, content: dict, time: str) -> None | pd.DataFrame:
        # Query to select the current record using primary key
        select_sql = """
        SELECT venue, original_openreview_id, revision_openreview_id, content, time
        FROM revisions WHERE revision_openreview_id = %s AND venue = %s;
        """
        self.cur.execute(select_sql, (revision_openreview_id, venue,))
        row = self.cur.fetchone()

        if not row:
            print(f"No revision found with revision_openreview_id {revision_openreview_id}.")
            return None
        else:
            # If record exists, return the original record as a dictionary
            columns = ['venue', 'original_openreview_id', 
                       'revision_openreview_id', 'content', 'time']
            # original_record = dict(zip(columns, row))
            revision_df = pd.DataFrame(row, columns=columns)
            
            # SQL query to update the record
            update_sql = """
            UPDATE revisions
            SET venue = %s,
                original_openreview_id = %s,
                content = %s,
                time = %s
            WHERE revision_openreview_id = %s;
            """

            cleaned_content = self._clean_json_content(content)

            self.cur.execute(update_sql, (venue, original_openreview_id, 
                                        Json(cleaned_content), time, revision_openreview_id))
            self.conn.commit()

            print(f"Revision with revision_openreview_id {revision_openreview_id} updated successfully.")
            return revision_df
        
    def get_revision_by_id(self, revision_openreview_id: str) -> None | pd.DataFrame:
        # Query to select the current record using primary key
        select_sql = """
        SELECT * FROM revisions WHERE revision_openreview_id = %s;
        """
        self.cur.execute(select_sql, (revision_openreview_id,))
        row = self.cur.fetchone()
        if not row:
            print(f"No revision found with revision_openreview_id {revision_openreview_id}.")
            return None
        else:
            # If record exists, return the original record as a dictionary
            columns = ['venue', 'original_openreview_id', 
                       'revision_openreview_id', 'content', 'time']
            
            # 处理content字段，如果是dict则转换为JSON字符串
            row_list = list(row[1:])
            if isinstance(row_list[3], dict):
                row_list[3] = json.dumps(row_list[3])
            revision_df = pd.DataFrame([row_list], columns=columns)
            return revision_df
    
    def get_revisions_by_venue(self, venue: str) -> None | pd.DataFrame:
        # Query to select all records for a specific venue
        select_sql = """
        SELECT venue, original_openreview_id, revision_openreview_id, content, time
        FROM revisions WHERE venue = %s;
        """
        self.cur.execute(select_sql, (venue,))
        rows = self.cur.fetchall()
        
        if not rows:
            print(f"No revisions found in venue {venue}.")
            return None
        else:
            columns = ['venue', 'original_openreview_id', 
                       'revision_openreview_id', 'content', 'time']
            
            # 处理所有行的content字段，如果是dict则转换为JSON字符串
            processed_rows = []
            for row in rows:
                row_list = list(row[1:])
                if isinstance(row_list[3], dict):
                    row_list[3] = json.dumps(row_list[3])
                processed_rows.append(row_list)
            
            revisions_df = pd.DataFrame(processed_rows, columns=columns)
            return revisions_df
        
    def get_all_revisions(self, is_all_features: bool = False) -> None | pd.DataFrame:
        if is_all_features:
            select_query = """
            SELECT venue, original_openreview_id, revision_openreview_id, content, time
            FROM revisions ORDER BY id ASC
            """
            self.cur.execute(select_query)
            
            # Fetch all the results
            revisions = self.cur.fetchall()
            
            # 处理所有行的content字段，如果是dict则转换为JSON字符串
            processed_revisions = []
            for row in revisions:
                row_list = list(row[1:])
                if isinstance(row_list[3], dict):
                    row_list[3] = json.dumps(row_list[3])
                processed_revisions.append(row_list)
            
            # Return the result as a DataFrame
            revisions_df = pd.DataFrame(processed_revisions, columns=["venue", "original_openreview_id", "revision_openreview_id", "content", "time"])
            return revisions_df
        else:
            select_query = """
            SELECT venue, original_openreview_id, revision_openreview_id, time
            FROM revisions ORDER BY id ASC
            """
            self.cur.execute(select_query)
            
            # Fetch all the results
            revisions = self.cur.fetchall()
            
            # Return the result as a list of tuples (paper_openreview_id, title, author_full_names)
            revisions_df = pd.DataFrame(revisions, columns=["venue", "original_openreview_id", "revision_openreview_id", "time"])
            return revisions_df
        
    def check_revision_exists(self, revision_openreview_id: str) -> bool | None:
        self.cur.execute("SELECT 1 FROM revisions WHERE revision_openreview_id = %s LIMIT 1;", (revision_openreview_id,))
        result = self.cur.fetchone()

        return result is not None
    
    def construct_revisions_table_from_api(self, venue: str, filter_list: list, pdf_dir: str, log_file: str):
        # crawl revision data from openreview API
        print("Crawling revision data from OpenReview API...")
        revision_data = self.openreview_crawler.crawl_revision_data_from_api(venue, filter_list, pdf_dir, log_file)
        
        # insert data into revisions table
        if len(revision_data) > 0:
            print("Inserting data into 'papers' table...")
            for data in tqdm(revision_data):
                try:
                    self.insert_revision(**data)
                except Exception as e:
                    print(f"Error inserting revision data: {e}")
                    continue
        else:
            print("No new revision data to insert.")
    
    def construct_revisions_table_from_csv(self, csv_file: str):
        # read revision data from csv file
        print(f"Reading revisions data from {csv_file}...")
        revision_data = pd.read_csv(csv_file).to_dict(orient='records')
        
        # insert data into papers table
        if len(revision_data) > 0:
            print("Inserting data into 'papers' table...")
            for data in tqdm(revision_data):
                data["content"] = ast.literal_eval(data["content"])
                self.insert_revision(**data)
        else:
            print("No new revision data to insert.")
            
    def construct_revisions_table_from_json(self, json_file: str):
        # read revision data from json file
        print(f"Reading revisions data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            revision_data = json.load(f)
        
        # insert data into papers table
        if len(revision_data) > 0:
            print("Inserting data into 'papers' table...")
            for data in tqdm(revision_data):
                self.insert_revision(**data)
        else:
            print("No new revision data to insert.")
        
    def _clean_json_content(self, content: dict | list | str) -> dict | list | str:
        if isinstance(content, str):
            # Remove null character (\u0000) and any other non-printable characters
            return ''.join(char for char in content if char.isprintable())
        elif isinstance(content, dict):
            # Clean all string values in the dictionary
            return {key: self._clean_json_content(value) for key, value in content.items()}
        elif isinstance(content, list):
            # Clean all elements in the list
            return [self._clean_json_content(item) for item in content]
        else:
            return content