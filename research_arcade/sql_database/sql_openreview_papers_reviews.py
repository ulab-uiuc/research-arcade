from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import psycopg2
from psycopg2.extras import Json

class SQLOpenReviewPapersReviews:
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
        self.create_papers_reviews_table()
        
    def create_papers_reviews_table(self) -> None:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS papers_reviews (
            venue TEXT,
            paper_openreview_id VARCHAR(255),
            review_openreview_id VARCHAR(255),
            title TEXT,
            time TEXT,
            PRIMARY KEY (venue, paper_openreview_id, review_openreview_id)
        );
        """
        self.cur.execute(create_table_sql)
        print("Table 'papers_reviews' created successfully.")
        
    def insert_paper_reviews(self, venue: str, paper_openreview_id: str, review_openreview_id: str, title: str, time: str) -> None | tuple:
        if self.check_paper_exists(paper_openreview_id) and self.check_review_exists(review_openreview_id):
            insert_sql = """
            INSERT INTO papers_reviews (venue, paper_openreview_id, review_openreview_id, title, time)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (venue, paper_openreview_id, review_openreview_id) DO NOTHING
            RETURNING (venue, paper_openreview_id, review_openreview_id);
            """
            
            venue = self._clean_string(venue)
            paper_openreview_id = self._clean_string(paper_openreview_id)
            review_openreview_id = self._clean_string(review_openreview_id)
            title = self._clean_string(title)
            time = self._clean_string(time)
            
            # Execute the insertion query
            self.cur.execute(insert_sql, (venue, paper_openreview_id, review_openreview_id, title, time))
            
            # Get the inserted paper's openreview id (if any)
            res = self.cur.fetchone()
            return res[0] if res else None
        elif self.check_paper_exists(paper_openreview_id) is False:
            print(f'''
                The paper {paper_openreview_id} is not exists in this database.
                ''')
            return None
        else:
            print(f'''
                The revision {review_openreview_id} is not exists in this database.
                ''')
            return None
    
    def delete_paper_review_by_id(self, paper_openreview_id: str, review_openreview_id: str) -> None | pd.DataFrame:
        original_record = self.get_paper_review_by_id(paper_openreview_id, review_openreview_id)
        
        if original_record is not None:
            self.cur.execute("""
            DELETE FROM papers_reviews
            WHERE paper_openreview_id = %s AND review_openreview_id = %s;
            """, (paper_openreview_id, review_openreview_id))
            
            self.conn.commit()
            print(f"The connection between paper {paper_openreview_id} and review {review_openreview_id} is deleted successfully.")
            return original_record
        else:
            return None
        
    def delete_papers_reviews_by_venue(self, venue: str) -> None | pd.DataFrame:
        # search the row based on primary key
        select_sql = """
        SELECT venue, paper_openreview_id, review_openreview_id FROM papers_reviews WHERE venue = %s;
        """
        self.cur.execute(select_sql, (venue,))
        rows = self.cur.fetchall()

        if rows:
            columns = ['venue', 'paper_openreview_id', 'review_openreview_id']
            # papers_reviews_dict = dict(zip(columns, rows))
            papers_reviews_df = pd.DataFrame(rows, columns=columns)

            delete_sql = """
            DELETE FROM papers_reviews WHERE venue = %s;
            """
            self.cur.execute(delete_sql, (venue,))
            self.conn.commit()

            print(f"All connections in venue {venue} deleted successfully.")
            return papers_reviews_df
        else:
            print(f"No connections found in venue {venue}.")
            return None
    
    def get_papers_reviews_by_venue(self, venue: str) -> None | pd.DataFrame:
        self.cur.execute("""
        SELECT venue, paper_openreview_id, review_openreview_id, title, time FROM papers_reviews
        WHERE venue = %s;
        """, (venue,))
        
        papers_reviews = self.cur.fetchall()
        
        if papers_reviews is not None:
            papers_reviews_df = pd.DataFrame(papers_reviews, columns=["venue", "paper_openreview_id", "review_openreview_id", "title", "time"])
            return papers_reviews_df
        else:
            return None
    
    def get_all_papers_reviews(self) -> None | pd.DataFrame:
        select_query = """
        SELECT venue, paper_openreview_id, review_openreview_id, title, time
        FROM papers_reviews ORDER BY paper_openreview_id ASC
        """
        self.cur.execute(select_query)
            
        # Fetch all the results
        papers_reviews = self.cur.fetchall()
        
        # Return the result as a list of tuples (venue, author_openreview_id, author_full_name)
        papers_reviews_df = pd.DataFrame(papers_reviews, columns=["venue", "paper_openreview_id", "review_openreview_id", "title", "time"])
        return papers_reviews_df
    
    def check_paper_review_exists(self, paper_openreview_id: str, review_openreview_id: str) -> bool | None:
        self.cur.execute("""
        SELECT 1 FROM papers_reviews
        WHERE paper_openreview_id = %s AND review_openreview_id = %s 
        LIMIT 1;
        """, (paper_openreview_id, review_openreview_id))
        
        result = self.cur.fetchone()

        return result is not None
    
    def get_paper_neighboring_reviews(self, paper_openreview_id: str) -> None | pd.DataFrame:
        self.cur.execute("""
        SELECT venue, paper_openreview_id, review_openreview_id, title, time FROM papers_reviews
        WHERE paper_openreview_id = %s;
        """, (paper_openreview_id,))
        
        paper_neighboring_reviews = self.cur.fetchall()
        
        if paper_neighboring_reviews is not None:
            paper_neighboring_reviews_df = pd.DataFrame(paper_neighboring_reviews, columns=["venue", "paper_openreview_id", "review_openreview_id", "title", "time"])
            return paper_neighboring_reviews_df
        else:
            return None
        
    def get_review_neighboring_papers(self, review_openreview_id: str) -> None | pd.DataFrame:
        self.cur.execute("""
        SELECT venue, paper_openreview_id, review_openreview_id, title, time FROM papers_reviews
        WHERE review_openreview_id = %s;
        """, (review_openreview_id,))
        
        review_neighboring_papers = self.cur.fetchall()
        
        if review_neighboring_papers is not None:
            review_neighboring_papers_df = pd.DataFrame(review_neighboring_papers, columns=["venue", "paper_openreview_id", "review_openreview_id", "title", "time"])
            return review_neighboring_papers_df
        else:
            return None
        
    def construct_papers_reviews_table_from_api(self, venue: str) -> None:
        # crawl the data from openreview API
        print(f"Crawling paper-review connections for venue: {venue}...")
        papers_reviews_data = self.openreview_crawler.crawl_papers_reviews_from_api(venue)
        
        if len(papers_reviews_data) > 0:
            print(f"Inserting paper-review connections into the database for venue: {venue}...")
            for data in tqdm(papers_reviews_data):
                self.insert_paper_reviews(**data)
        else:
            print(f"No paper-review connections found for venue: {venue}.")
            
    def construct_papers_reviews_table_from_csv(self, csv_file: str) -> None:
        # read paper-review connection data from csv file
        print(f"Reading paper-review connection data from {csv_file}...")
        papers_reviews_data = pd.read_csv(csv_file).to_dict(orient='records')
        
        # insert data into papers_reviews table
        if len(papers_reviews_data) > 0:
            print("Inserting data into 'papers_reviews' table...")
            for data in tqdm(papers_reviews_data):
                self.insert_paper_reviews(**data)
        else:
            print("No new paper-review connection data to insert.")
            
    def construct_papers_reviews_table_from_json(self, json_file: str) -> None:
        # read paper-review connection data from json file
        print(f"Reading paper-review connection data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            papers_reviews_data = json.load(f)
        
        # insert data into papers_reviews table
        if len(papers_reviews_data) > 0:
            print("Inserting data into 'papers_reviews' table...")
            for data in tqdm(papers_reviews_data):
                self.insert_paper_reviews(**data)
        else:
            print("No new paper-review connection data to insert.")
            
    def _clean_string(self, s: str) -> str:
        if isinstance(s, str):
            return s.replace('\x00', '')
        return s