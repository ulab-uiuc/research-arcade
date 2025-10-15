from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import psycopg2
from psycopg2.extras import Json

class SQLOpenReviewRevisionsReviews:
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
        self.create_revisions_reviews_table()
        
    def create_revisions_reviews_table(self) -> None:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS openreview_revisions_reviews (
            venue TEXT,
            revision_openreview_id VARCHAR(255),
            review_openreview_id VARCHAR(255),
            PRIMARY KEY (venue, revision_openreview_id, review_openreview_id)
        );
        """
        # Execute the SQL to create the table
        self.cur.execute(create_table_sql)
        print("Table 'openreview_revisions_reviews' created successfully.")
        
    def insert_revision_reviews(self, venue: str, revision_openreview_id: str, review_openreview_id: str) -> None | tuple:
        insert_sql = """
        INSERT INTO openreview_revisions_reviews (venue, revision_openreview_id, review_openreview_id)
        VALUES (%s, %s, %s)
        ON CONFLICT (venue, revision_openreview_id, review_openreview_id) DO NOTHING
        RETURNING (venue, revision_openreview_id, review_openreview_id);
        """
        
        venue = self._clean_string(venue)
        revision_openreview_id = self._clean_string(revision_openreview_id)
        review_openreview_id = self._clean_string(review_openreview_id)
        
        # Execute the insertion query
        self.cur.execute(insert_sql, (venue, revision_openreview_id, review_openreview_id))
        
        # Get the inserted paper's openreview id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
        
    def get_revision_review_by_id(self, revision_openreview_id: str, review_openreview_id: str) -> None | pd.DataFrame:
        self.cur.execute("""
        SELECT venue, revision_openreview_id, review_openreview_id FROM openreview_revisions_reviews
        WHERE revision_openreview_id = %s AND review_openreview_id = %s;
        """, (revision_openreview_id, review_openreview_id))
        
        result = self.cur.fetchone()
        
        if result is None:
            print(f'''
                The revision {revision_openreview_id} and the review {review_openreview_id} are not connect in this database.
                ''')
            return None
        else:
            columns = ['venue', 'revision_openreview_id', 'review_openreview_id']
            # result = dict(zip(columns, result))
            result_df = pd.DataFrame(result, columns=columns)
            return result_df
        
    def get_revisions_reviews_by_venue(self, venue: str) -> None | pd.DataFrame:
        self.cur.execute("""
        SELECT venue, revision_openreview_id, review_openreview_id FROM openreview_revisions_reviews
        WHERE venue = %s;
        """, (venue,))
        
        openreview_revisions_reviews = self.cur.fetchall()
        
        if openreview_revisions_reviews is not None:
            revisions_reviews_df = pd.DataFrame(openreview_revisions_reviews, columns=["venue", "revision_openreview_id", "review_openreview_id"])
            return revisions_reviews_df
        else:
            return None
        
    def get_all_revisions_reviews(self) -> None | pd.DataFrame:
        select_query = """
        SELECT venue, revision_openreview_id, review_openreview_id
        FROM openreview_revisions_reviews;
        """
        self.cur.execute(select_query)
            
        # Fetch all the results
        openreview_revisions_reviews = self.cur.fetchall()
        
        # Return the result as a list of tuples (venue, author_openreview_id, author_full_name)
        revisions_reviews_df = pd.DataFrame(openreview_revisions_reviews, columns=["venue", "revision_openreview_id", "review_openreview_id"])
        return revisions_reviews_df
    
    def delete_revision_review_by_id(self, revision_openreview_id: str, review_openreview_id: str) -> None | pd.DataFrame:
        # search for records with the given arxiv_id and delete them
        select_sql = """
        SELECT * FROM openreview_revisions_reviews WHERE revision_openreview_id = %s AND review_openreview_id = %s;
        """
        self.cur.execute(select_sql, (revision_openreview_id, review_openreview_id))
        records = self.cur.fetchall()
        
        if records:
            columns=["venue", "revision_openreview_id", "review_openreview_id"]
            records_df = pd.DataFrame(records, columns=columns)
            
            delete_sql = """
            DELETE FROM openreview_revisions_reviews WHERE revision_openreview_id = %s AND review_openreview_id = %s;
            """
            self.cur.execute(delete_sql, (revision_openreview_id, review_openreview_id))
            self.conn.commit()
            
            print(f"Deleted {len(records)} records from 'openreview_revisions_reviews' with revision_openreview_id = {revision_openreview_id} and review_openreview_id = {review_openreview_id}.")
            return records_df
        else:
            print(f"No records found in 'openreview_revisions_reviews' with revision_openreview_id = {revision_openreview_id} and review_openreview_id = {review_openreview_id}.")
            return None
    
    def delete_revision_review_by_revision_id(self, revision_openreview_id: str):
        # search for records with the given arxiv_id and delete them
        select_sql = """
        SELECT * FROM openreview_revisions_reviews WHERE revision_openreview_id = %s;
        """
        self.cur.execute(select_sql, (revision_openreview_id,))
        records = self.cur.fetchall()
        
        if records:
            columns=["venue", "revision_openreview_id", "review_openreview_id"]
            records_df = pd.DataFrame(records, columns=columns)
            
            delete_sql = """
            DELETE FROM openreview_revisions_reviews WHERE revision_openreview_id = %s;
            """
            self.cur.execute(delete_sql, (revision_openreview_id,))
            self.conn.commit()
            
            print(f"Deleted {len(records)} records from 'openreview_revisions_reviews' with revision_openreview_id = {revision_openreview_id}.")
            return records_df
        else:
            print(f"No records found in 'openreview_revisions_reviews' with revision_openreview_id = {revision_openreview_id}.")
            return None
        
    def delete_revision_review_by_review_id(self, review_openreview_id: str):
        # search for records with the given arxiv_id and delete them
        select_sql = """
        SELECT * FROM openreview_revisions_reviews WHERE review_openreview_id = %s;
        """
        self.cur.execute(select_sql, (review_openreview_id,))
        records = self.cur.fetchall()
        
        if records:
            columns=["venue", "revision_openreview_id", "review_openreview_id"]
            records_df = pd.DataFrame(records, columns=columns)
            
            delete_sql = """
            DELETE FROM openreview_revisions_reviews WHERE review_openreview_id = %s;
            """
            self.cur.execute(delete_sql, (review_openreview_id,))
            self.conn.commit()
            
            print(f"Deleted {len(records)} records from 'openreview_revisions_reviews' with review_openreview_id = {review_openreview_id}.")
            return records_df
        else:
            print(f"No records found in 'openreview_revisions_reviews' with review_openreview_id = {review_openreview_id}.")
            return None
    
    def delete_revisions_reviews_by_venue(self, venue: str) -> None | pd.DataFrame:
        # search the row based on primary key
        select_sql = """
        SELECT venue, revision_openreview_id, review_openreview_id FROM openreview_revisions_reviews WHERE venue = %s;
        """
        self.cur.execute(select_sql, (venue,))
        rows = self.cur.fetchall()
        
        if rows:
            columns = ['venue', 'revision_openreview_id', 'review_openreview_id']
            # papers_reviews_dict = dict(zip(columns, rows))
            revisions_reviews_df = pd.DataFrame(rows, columns=columns)

            delete_sql = """
            DELETE FROM openreview_revisions_reviews WHERE venue = %s;
            """
            self.cur.execute(delete_sql, (venue,))
            self.conn.commit()

            print(f"All connections in venue {venue} deleted successfully.")
            return revisions_reviews_df
        else:
            print(f"No connections found in venue {venue}.")
            return None
        
    def check_revision_review_exists(self, revision_openreview_id: str, review_openreview_id: str) -> bool | None:
        self.cur.execute("""
        SELECT 1 FROM openreview_revisions_reviews
        WHERE revision_openreview_id = %s AND review_openreview_id = %s 
        LIMIT 1;
        """, (revision_openreview_id, review_openreview_id))
        
        result = self.cur.fetchone()

        return result is not None
    
    def get_revision_neighboring_reviews(self, revision_openreview_id: str) -> None | pd.DataFrame:
        self.cur.execute("""
        SELECT venue, revision_openreview_id, review_openreview_id FROM openreview_revisions_reviews
        WHERE revision_openreview_id = %s;
        """, (revision_openreview_id,))
        
        revision_neighboring_reviews = self.cur.fetchall()
        
        if revision_neighboring_reviews is not None:
            revision_neighboring_reviews_df = pd.DataFrame(revision_neighboring_reviews, columns=["venue", "revision_openreview_id", "review_openreview_id"])
            return revision_neighboring_reviews_df
        else:
            return None
        
    def get_review_neighboring_revisions(self, review_openreview_id: str) -> None | pd.DataFrame:
        self.cur.execute("""
        SELECT venue, revision_openreview_id, review_openreview_id FROM openreview_revisions_reviews
        WHERE review_openreview_id = %s;
        """, (review_openreview_id,))
        
        review_neighboring_revisions = self.cur.fetchall()
        
        if review_neighboring_revisions is not None:
            review_neighboring_revisions_df = pd.DataFrame(review_neighboring_revisions, columns=["venue", "revision_openreview_id", "review_openreview_id"])
            return review_neighboring_revisions_df
        else:
            return None
        
    def construct_revisions_reviews_table(self, papers_reviews_df: pd.DataFrame, papers_revisions_df: pd.DataFrame) -> None:
        # get unique paper ids
        unique_paper_ids = papers_revisions_df['paper_openreview_id'].unique()
        
        for paper_id in tqdm(unique_paper_ids):
            # get the revision ids
            paper_revision_edges = papers_revisions_df[papers_revisions_df['paper_openreview_id'] == paper_id].sort_values(by='time', ascending=True)
            
            # get the review ids
            paper_review_edges = papers_reviews_df[papers_reviews_df['paper_openreview_id'] == paper_id].sort_values(by='time', ascending=True)
            
            start_idx = 0
            for revision in paper_revision_edges.itertuples():
                # get the revision time
                revision_time = revision.time
                # get the revision id
                revision_id = revision.revision_openreview_id
                
                # get the review ids
                for review in paper_review_edges.iloc[start_idx:].itertuples():
                    # get the review time
                    review_time = review.time
                    if review_time > revision_time:
                        break
                    
                    # get the review id
                    review_id = review.review_openreview_id
                    
                    # insert the edge
                    self.db.insert_revision_reviews(venue, revision_id, review_id)
                    
                    start_idx += 1
                    
    def construct_revisions_reviews_table_from_csv(self, csv_file: str) -> None:
        # read data from csv file
        print(f"Reading revisions-reviews data from {csv_file}...")
        revisions_reviews_data = pd.read_csv(csv_file).to_dict(orient='records')
        
        if len(revisions_reviews_data) > 0:
            print(f"Inserting revisions-reviews data from {csv_file}...")
            for data in tqdm(revisions_reviews_data):
                self.insert_revision_reviews(venue, revision_openreview_id, review_openreview_id)
        else:
            print(f"No revisions-reviews data found in {csv_file}.")
            
    def construct_revisions_reviews_table_from_json(self, json_file: str) -> None:
        # read revision data from json file
        print(f"Reading revisions-reviews data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            revisions_reviews_data = json.load(f)
        
        if len(revisions_reviews_data) > 0:
            print(f"Inserting revisions-reviews data from {json_file}...")
            for data in tqdm(revisions_reviews_data):
                self.insert_revision_reviews(**data)
        else:
            print(f"No revisions-reviews data found in {json_file}.")
            
    def _clean_string(self, s: str) -> str:
        if isinstance(s, str):
            return s.replace('\x00', '')
        return s