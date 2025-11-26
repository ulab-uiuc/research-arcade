from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import ast
import psycopg2
from psycopg2.extras import Json
import os
from typing import Optional, Union

class SQLOpenReviewReviews:
    def __init__(self, host: str, dbname: str, user: str, password: str, port: str) -> None:
        self.conn = psycopg2.connect(
            host=host,
            dbname=dbname,
            user=user,
            password=password,
            port=port
        )
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        self.crawler = OpenReviewCrawler()
        self.create_reviews_table()
        
    def create_reviews_table(self) -> None:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS openreview_reviews  (
            id SERIAL UNIQUE,
            venue TEXT,
            review_openreview_id VARCHAR(255),
            replyto_openreview_id VARCHAR(255),
            writer TEXT,
            title TEXT,
            content JSONB,
            time TEXT,
            PRIMARY KEY (venue, review_openreview_id)
        );
        """
        # Execute the SQL to create the table
        self.cur.execute(create_table_sql)
        
    def insert_review(self, venue: str, review_openreview_id: str, replyto_openreview_id: str, 
                      title: str, writer: str, content: dict, time: str) -> Optional[tuple]:
        """
        Insert a review into the openreview_reviews  table. Returns the inserted review id or None if it fails.
        - venue: str, the venue where the paper was submitted.
        - review_openreview_id: str, unique ID for the review (primary key).
        - replyto_openreview_id: str or None, ID for a reply (optional).
        - writer: str, the name or identity of the reviewer.
        - title: str, the title of the review.
        - content: JSON object, the content of the review.
        - time
        """
        insert_sql = """
        INSERT INTO openreview_reviews  (venue, review_openreview_id, replyto_openreview_id, writer, title, content, time)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (venue, review_openreview_id) DO NOTHING
        RETURNING (venue, review_openreview_id);
        """
        
        # clean content
        cleaned_content = self._clean_json_content(content)
        venue = self._clean_string(venue)
        review_openreview_id = self._clean_string(review_openreview_id)
        replyto_openreview_id = self._clean_string(replyto_openreview_id)
        writer = self._clean_string(writer)
        title = self._clean_string(title)
        time = self._clean_string(time)
        
        # Execute the insertion query
        self.cur.execute(insert_sql, (venue, review_openreview_id, replyto_openreview_id, writer, title, Json(cleaned_content), time))
        
        # Get the inserted review id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
    
    def delete_review_by_id(self, review_openreview_id: str) -> Optional[pd.DataFrame]:
        # search for the row based on primary key
        select_sql = """
        SELECT venue, review_openreview_id, replyto_openreview_id, writer, title, content, time 
        FROM openreview_reviews  WHERE review_openreview_id = %s;
        """
        self.cur.execute(select_sql, (review_openreview_id,))
        row = self.cur.fetchall()

        if row:
            columns = ['venue', 'review_openreview_id', 'replyto_openreview_id', 
                       'writer', 'title', 'content', 'time']
            # review_dict = dict(zip(columns, row))
            review_df = pd.DataFrame(row, columns=columns)

            delete_sql = """
            DELETE FROM openreview_reviews  WHERE review_openreview_id = %s;
            """
            self.cur.execute(delete_sql, (review_openreview_id,))
            self.conn.commit()

            print(f"Review with review_openreview_id {review_openreview_id} deleted successfully.")
            return review_df
        else:
            print(f"No review found with review_openreview_id {review_openreview_id}.")
            return None
        
    def delete_reviews_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        # search for the row based on primary key
        select_sql = """
        SELECT venue, review_openreview_id, replyto_openreview_id, writer, title, content, time 
        FROM openreview_reviews  WHERE venue = %s;
        """
        self.cur.execute(select_sql, (venue,))
        row = self.cur.fetchall()

        if row:
            columns = ['venue', 'review_openreview_id', 'replyto_openreview_id', 
                       'writer', 'title', 'content', 'time']
            # review_dict = dict(zip(columns, row))
            review_df = pd.DataFrame(row, columns=columns)

            delete_sql = """
            DELETE FROM openreview_reviews  WHERE venue = %s;
            """
            self.cur.execute(delete_sql, (venue,))
            self.conn.commit()

            print(f"All openreview_reviews  in venue {venue} deleted successfully.")
            return review_df
        else:
            print(f"No openreview_reviews  found in venue {venue}.")
            return None
        
    def update_review(self, venue: str, review_openreview_id: str, replyto_openreview_id: str, 
                    writer: str, title: str, content: dict, time: str) -> Optional[pd.DataFrame]:
        # Query to select the current record using primary key
        select_sql = """
        SELECT venue, review_openreview_id, replyto_openreview_id, writer, title, content, time 
        FROM openreview_reviews  WHERE review_openreview_id = %s AND venue = %s;
        """
        self.cur.execute(select_sql, (review_openreview_id, venue,))
        row = self.cur.fetchone()

        if not row:
            print(f"No review found with review_openreview_id {review_openreview_id}.")
            return None
        else:
            columns = ['venue', 'review_openreview_id', 'replyto_openreview_id', 
                    'writer', 'title', 'content', 'time']
            # original_record = dict(zip(columns, row))
            review_df = pd.DataFrame([row], columns=columns)
            
            # SQL query to update the record
            update_sql = """
            UPDATE openreview_reviews
            SET venue = %s,
                replyto_openreview_id = %s,
                writer = %s,
                title = %s,
                content = %s,
                time = %s
            WHERE review_openreview_id = %s;
            """
            
            venue = self._clean_string(venue)
            review_openreview_id = self._clean_string(review_openreview_id)
            replyto_openreview_id = self._clean_string(replyto_openreview_id)
            writer = self._clean_string(writer)
            title = self._clean_string(title)
            time = self._clean_string(time)
            
            cleaned_content = self._clean_json_content(content)

            self.cur.execute(update_sql, (venue, replyto_openreview_id, writer, title, 
                                        Json(cleaned_content), time, review_openreview_id))
            self.conn.commit()

            print(f"Review with review_openreview_id {review_openreview_id} updated successfully.")
            return review_df
        
    def get_review_by_id(self, review_openreview_id: str) -> Optional[pd.DataFrame]:
        # Query to select the current record using primary key
        select_sql = """
        SELECT venue, review_openreview_id, replyto_openreview_id, writer, title, content, time 
        FROM openreview_reviews  WHERE review_openreview_id = %s;
        """
        self.cur.execute(select_sql, (review_openreview_id,))
        row = self.cur.fetchone()

        if not row:
            print(f"No review found with review_openreview_id {review_openreview_id}.")
            return None
        else:
            columns = ['venue', 'review_openreview_id', 'replyto_openreview_id', 
                    'writer', 'title', 'content', 'time']

            row_list = list(row)
            if isinstance(row_list[5], dict):
                row_list[5] = json.dumps(row_list[5])
            # original_record = dict(zip(columns, row))
            review_df = pd.DataFrame([row_list], columns=columns)

            return review_df
        
    def get_reviews_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        # Query to select all records for a specific venue
        select_sql = """
        SELECT venue, review_openreview_id, replyto_openreview_id, writer, title, content, time 
        FROM openreview_reviews  WHERE venue = %s;
        """
        self.cur.execute(select_sql, (venue,))
        rows = self.cur.fetchall()
        
        if not rows:
            print(f"No openreview_reviews  found in venue {venue}.")
            return None
        else:
            columns = ['venue', 'review_openreview_id', 'replyto_openreview_id', 
                    'writer', 'title', 'content', 'time']
            # original_record = dict(zip(columns, row))
            processed_rows = []
            for row in rows:
                row_list = list(row)
                if isinstance(row_list[5], dict):
                    row_list[3] = json.dumps(row_list[5])
                processed_rows.append(row_list)
            reviews_df = pd.DataFrame(processed_rows, columns=columns)
            return reviews_df
        
    def get_all_reviews(self, is_all_features: bool = False) -> Optional[pd.DataFrame]:
        if is_all_features:
            select_query = """
            SELECT venue, review_openreview_id, replyto_openreview_id, writer, title, content, time 
            FROM openreview_reviews  ORDER BY id ASC
            """
            self.cur.execute(select_query)
            
            # Fetch all the results
            reviews= self.cur.fetchall()
            processed_reviews= []
            for row in reviews:
                row_list = list(row)
                if isinstance(row_list[5], dict):
                    row_list[5] = json.dumps(row_list[5])
                processed_reviews.append(row_list)
            
            # Return the result as a list of tuples (paper_openreview_id, title, author_full_names)
            reviews_df = pd.DataFrame(processed_reviews, columns=["venue", "review_openreview_id", "replyto_openreview_id", "writer", "title", "content", "time"])
            return reviews_df
        else:
            select_query = """
            SELECT venue, review_openreview_id, replyto_openreview_id, title, time 
            FROM openreview_reviews  ORDER BY id ASC
            """
            self.cur.execute(select_query)
            
            # Fetch all the results
            reviews= self.cur.fetchall()
            
            # Return the result as a list of tuples (paper_openreview_id, title, author_full_names)
            reviews_df = pd.DataFrame(reviews, columns=["venue", "review_openreview_id", "replyto_openreview_id", "title", "time"])
            return reviews_df
        
    def check_review_exists(self, review_openreview_id: str) -> bool:
        self.cur.execute("SELECT 1 FROM openreview_reviews  WHERE review_openreview_id = %s LIMIT 1;", (review_openreview_id,))
        result = self.cur.fetchone()

        return result is not None
    
    def construct_reviews_table_from_api(self, venue: str) -> None:
        # crawl review data from openreview API
        print("Crawling review data from OpenReview API...")
        review_data = self.crawler.crawl_review_data_from_api(venue)
        
        # insert data into openreview_reviews  table
        if len(review_data) > 0:
            print("Inserting data into 'openreview_reviews' table...")
            for data in tqdm(review_data):
                self.insert_review(**data)
        else:
            print("No new review data to insert.")
            
    def construct_reviews_table_from_csv(self, csv_file: str) -> bool:
        if not os.path.exists(csv_file):
            return False
        else:
            # read review data from csv
            print(f"Reading review data from {csv_file}...")
            review_data = pd.read_csv(csv_file).to_dict(orient='records')
            
            # insert data into openreview_reviews  table
            if len(review_data) > 0:
                print("Inserting data into 'openreview_reviews' table...")
                for data in tqdm(review_data):
                    data["content"] = ast.literal_eval(data["content"])
                    self.insert_review(**data)
                return True
            else:
                print("No new review data to insert.")
                return False
            
    def construct_reviews_table_from_json(self, json_file: str) -> bool:
        if not os.path.exists(json_file):
            return False
        else:
            # read review data from json
            print(f"Reading review data from {json_file}...")
            with open(json_file, 'r') as f:
                review_data = json.load(f)
            
            # insert data into openreview_reviews  table
            if len(review_data) > 0:
                print("Inserting data into 'openreview_reviews' table...")
                for data in tqdm(review_data):
                    self.insert_review(**data)
                return True
            else:
                print("No new review data to insert.")
                return False
    
    def _clean_json_content(self, content: Union[str, dict, list, int, float, bool, None]) -> Union[str, dict, list, int, float, bool, None]:
        if isinstance(content, str):
            return ''.join(char for char in content if char.isprintable())  # 返回 str
        elif isinstance(content, dict):
            return {key: self._clean_json_content(value) for key, value in content.items()}  # 返回 dict
        elif isinstance(content, list):
            return [self._clean_json_content(item) for item in content]  # 返回 list
        else:
            return content  # 返回原始类型
        
    def _clean_string(self, s: str) -> str:
        if isinstance(s, str):
            return s.replace('\x00', '')
        return s