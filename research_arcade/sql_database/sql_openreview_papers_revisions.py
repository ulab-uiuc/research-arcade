from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import psycopg2
from psycopg2.extras import Json

class SQLOpenReviewPapersRevisions:
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
        self.create_papers_revisions_table()
        
    def create_papers_revisions_table(self) -> None:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS openreview_papers_revisions (
            venue TEXT,
            paper_openreview_id VARCHAR(255),
            revision_openreview_id VARCHAR(255),
            title TEXT,
            time TEXT,
            PRIMARY KEY (venue, paper_openreview_id, revision_openreview_id)
        );
        """
        self.cur.execute(create_table_sql)
        
    def insert_paper_revisions(self, venue: str, paper_openreview_id: str, revision_openreview_id: str, title: str, time: str) -> None | tuple:
        insert_sql = """
        INSERT INTO openreview_papers_revisions (venue, paper_openreview_id, revision_openreview_id, title, time)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (venue, paper_openreview_id, revision_openreview_id) DO NOTHING
        RETURNING (venue, paper_openreview_id, revision_openreview_id);
        """
        
        venue = self._clean_string(venue)
        paper_openreview_id = self._clean_string(paper_openreview_id)
        revision_openreview_id = self._clean_string(revision_openreview_id)
        title = self._clean_string(title)
        time = self._clean_string(time)
        
        # Execute the insertion query
        self.cur.execute(insert_sql, (venue, paper_openreview_id, revision_openreview_id, title, time))
        
        # Get the inserted paper's openreview id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
    
    def delete_paper_revision_by_id(self, paper_openreview_id: str, revision_openreview_id: str) -> None | pd.DataFrame:
        # search for records with the given arxiv_id and delete them
        select_sql = """
        SELECT * FROM openreview_papers_revisions WHERE paper_openreview_id = %s AND revision_openreview_id = %s;
        """
        self.cur.execute(select_sql, (paper_openreview_id, revision_openreview_id))
        records = self.cur.fetchall()
        
        if records:
            columns=["venue", "paper_openreview_id", "revision_openreview_id", 'title', 'time']
            records_df = pd.DataFrame(records, columns=columns)
            
            delete_sql = """
            DELETE FROM openreview_papers_revisions WHERE paper_openreview_id = %s AND revision_openreview_id = %s;
            """
            self.cur.execute(delete_sql, (paper_openreview_id, revision_openreview_id))
            self.conn.commit()
            
            print(f"Deleted {len(records)} records from 'openreview_papers_revisions' with paper_openreview_id = {paper_openreview_id} and revision_openreview_id = {revision_openreview_id}.")
            return records_df
        else:
            print(f"No records found in 'openreview_papers_revisions' with paper_openreview_id = {paper_openreview_id} and revision_openreview_id = {revision_openreview_id}.")
            return None
    
    def delete_paper_revision_by_paper_id(self, paper_openreview_id: str):
        # search for records with the given arxiv_id and delete them
        select_sql = """
        SELECT * FROM openreview_papers_revisions WHERE paper_openreview_id = %s;
        """
        self.cur.execute(select_sql, (paper_openreview_id,))
        records = self.cur.fetchall()
        
        if records:
            columns=["venue", "paper_openreview_id", "revision_openreview_id", 'title', 'time']
            records_df = pd.DataFrame(records, columns=columns)
            
            delete_sql = """
            DELETE FROM openreview_papers_revisions WHERE paper_openreview_id = %s;
            """
            self.cur.execute(delete_sql, (paper_openreview_id,))
            self.conn.commit()
            
            print(f"Deleted {len(records)} records from 'openreview_papers_revisions' with paper_openreview_id = {paper_openreview_id}.")
            return records_df
        else:
            print(f"No records found in 'openreview_papers_revisions' with paper_openreview_id = {paper_openreview_id}.")
            return None
        
    def delete_paper_revision_by_revision_id(self, revision_openreview_id: str):
        # search for records with the given arxiv_id and delete them
        select_sql = """
        SELECT * FROM openreview_papers_revisions WHERE revision_openreview_id = %s;
        """
        self.cur.execute(select_sql, (revision_openreview_id,))
        records = self.cur.fetchall()
        
        if records:
            columns=["venue", "paper_openreview_id", "revision_openreview_id", 'title', 'time']
            records_df = pd.DataFrame(records, columns=columns)
            
            delete_sql = """
            DELETE FROM openreview_papers_revisions WHERE revision_openreview_id = %s;
            """
            self.cur.execute(delete_sql, (revision_openreview_id,))
            self.conn.commit()
            
            print(f"Deleted {len(records)} records from 'openreview_papers_revisions' with revision_openreview_id = {revision_openreview_id}.")
            return records_df
        else:
            print(f"No records found in 'openreview_papers_revisions' with revision_openreview_id = {revision_openreview_id}.")
            return None
        
    def delete_papers_revisions_by_venue(self, venue: str) -> None | pd.DataFrame:
        # search the row based on primary key
        select_sql = """
        SELECT venue, paper_openreview_id, revision_openreview_id, title, time FROM openreview_papers_revisions WHERE venue = %s;
        """
        self.cur.execute(select_sql, (venue,))
        rows = self.cur.fetchall()

        if rows:
            columns = ['venue', 'paper_openreview_id', 'revision_openreview_id', 'title', 'time']
            # papers_revisions_dict = dict(zip(columns, rows))
            papers_revisions_df = pd.DataFrame(rows, columns=columns)

            delete_sql = """
            DELETE FROM openreview_papers_revisions WHERE venue = %s;
            """
            self.cur.execute(delete_sql, (venue,))
            self.conn.commit()

            print(f"All connections in venue {venue} deleted successfully.")
            return papers_revisions_df
        else:
            print(f"No connections found in venue {venue}.")
            return None
    
    def get_papers_revisions_by_venue(self, venue: str) -> None | pd.DataFrame:
        self.cur.execute("""
        SELECT venue, paper_openreview_id, revision_openreview_id, title, time FROM openreview_papers_revisions
        WHERE venue = %s;
        """, (venue,))
        
        openreview_papers_revisions = self.cur.fetchall()
        
        if openreview_papers_revisions is not None:
            papers_revisions_df = pd.DataFrame(papers_revisions, columns=["venue", "paper_openreview_id", "revision_openreview_id", "title", "time"])
            return papers_revisions_df
        else:
            return None
    
    def get_all_papers_revisions(self) -> None | pd.DataFrame:
        select_query = """
        SELECT venue, paper_openreview_id, revision_openreview_id, title, time
        FROM openreview_papers_revisions ORDER BY paper_openreview_id ASC
        """
        self.cur.execute(select_query)
            
        # Fetch all the results
        papers_authors = self.cur.fetchall()
        
        # Return the result as a list of tuples (venue, author_openreview_id, author_full_name)
        authors_df = pd.DataFrame(papers_authors, columns=["venue", "paper_openreview_id", "revision_openreview_id", "title", "time"])
        return authors_df
    
    def check_paper_revision_exists(self, paper_openreview_id: str, revision_openreview_id: str) -> bool | None:
        self.cur.execute("""
        SELECT 1 FROM openreview_papers_revisions
        WHERE paper_openreview_id = %s AND revision_openreview_id = %s 
        LIMIT 1;
        """, (paper_openreview_id, revision_openreview_id))
        
        result = self.cur.fetchone()

        return result is not None
    
    def get_paper_neighboring_revisions(self, paper_openreview_id: str) -> None | pd.DataFrame:
        self.cur.execute("""
        SELECT venue, paper_openreview_id, revision_openreview_id, title, time FROM openreview_papers_revisions
        WHERE paper_openreview_id = %s;
        """, (paper_openreview_id,))
        
        paper_neighboring_revisions = self.cur.fetchall()
        
        if paper_neighboring_revisions is not None:
            paper_neighboring_revisions_df = pd.DataFrame(paper_neighboring_revisions, columns=["venue", "paper_openreview_id", "revision_openreview_id", "title", "time"])
            return paper_neighboring_revisions_df
        else:
            return None
        
    def get_revision_neighboring_papers(self, revision_openreview_id: str) -> None | pd.DataFrame:
        self.cur.execute("""
        SELECT venue, paper_openreview_id, revision_openreview_id, title, time FROM openreview_papers_revisions
        WHERE revision_openreview_id = %s;
        """, (revision_openreview_id,))
        
        revision_neighboring_papers = self.cur.fetchall()
        
        if revision_neighboring_papers is not None:
            revision_neighboring_papers_df = pd.DataFrame(revision_neighboring_papers, columns=["venue", "paper_openreview_id", "revision_openreview_id", "title", "time"])
            return revision_neighboring_papers_df
        else:
            return None
        
    def construct_papers_revisions_table_from_api(self, venue: str) -> None:
        # fetch paper-revision data from openreview API
        print(f"Crawling paper-revision data from OpenReview API for venue: {venue}...")
        paper_revision_data = self.openreview_crawler.crawl_papers_revisions_data_from_api(venue)
        
        # insert data into openreview_papers_revisions table
        if len(paper_revision_data) > 0:
            print(f"Inserting paper-revision data into the database for venue: {venue}...")
            for data in tqdm(paper_revision_data):
                self.insert_paper_revisions(**data)
        else:
            print(f"No paper-revision data found for venue: {venue}.")
        
    def construct_papers_revisions_table_from_csv(self, csv_file: str) -> None:
        # read revision data from csv file
        revision_data = pd.read_csv(csv_file).to_dict(orient='records')
        
        # insert data into papers table
        if len(revision_data) > 0:
            print(f"Inserting paper-revision data from {csv_file} into the database...")
            for data in tqdm(revision_data):
                self.insert_paper_revisions(**data)
        else:
            print(f"No paper-revision data found in {csv_file}.")
            
    def construct_papers_revisions_table_from_json(self, json_file: str) -> None:
        # read revision data from json file
        print(f"Reading revisions data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            revision_data = json.load(f)
        
        # insert data into papers table
        if len(revision_data) > 0:
            print(f"Inserting paper-revision data from {json_file} into the database...")
            for data in tqdm(revision_data):
                self.insert_paper_revisions(**data)
        else:
            print(f"No paper-revision data found in {json_file}.")
            
    def _clean_string(self, s: str) -> str:
        if isinstance(s, str):
            return s.replace('\x00', '')
        return s