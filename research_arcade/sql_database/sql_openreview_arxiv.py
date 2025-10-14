from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import psycopg2
from psycopg2.extras import Json

class SQLOpenReviewArxiv:
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
        self.create_openreview_arxiv_table()
        
    def create_openreview_arxiv_table(self) -> None:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS openreview_arxiv (
            venue TEXT,
            paper_openreview_id VARCHAR(255),
            arxiv_id VARCHAR(255),
            title TEXT,
            PRIMARY KEY (venue, paper_openreview_id)
        );
        """
        # Execute the SQL to create the table
        self.cur.execute(create_table_sql)
        print("Table 'openreview_arxiv' created successfully.")
    
    def insert_openreview_arxiv(self, venue, openreview_id, arxiv_id, title) -> None | tuple:
        insert_sql = """
        INSERT INTO openreview_arxiv (venue, paper_openreview_id, arxiv_id, title)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (venue, paper_openreview_id) DO NOTHING
        RETURNING (venue, paper_openreview_id);
        """
        
        # Execute the insertion query
        self.cur.execute(insert_sql, (venue, openreview_id, arxiv_id, title))
        
        # Get the inserted paper's openreview id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
    
    def delete_openreview_arxiv_by_id(self, openreview_id: str) -> None:
        # search for records with the given openreview_id and delete them
        select_sql = """
        SELECT * FROM openreview_arxiv WHERE paper_openreview_id = %s;
        """
        self.cur.execute(select_sql, (openreview_id,))
        records = self.cur.fetchall()
        
        if records:
            columns=["venue", "paper_openreview_id", "arxiv_id", "title"]
            records_df = pd.DataFrame(records, columns=columns)
            
            delete_sql = """
            DELETE FROM openreview_arxiv WHERE paper_openreview_id = %s;
            """
            self.cur.execute(delete_sql, (openreview_id,))
            self.conn.commit()
            
            print(f"Deleted {len(records)} records from 'openreview_arxiv' with paper_openreview_id = {openreview_id}.")
            return records_df
        else:
            print(f"No records found in 'openreview_arxiv' with paper_openreview_id = {openreview_id}.")
            return None
    
    def delete_openreview_arxiv_by_venue(self, venue: str) -> None:
        # search for records with the given venue and delete them
        select_sql = """
        SELECT * FROM openreview_arxiv WHERE venue = %s;
        """
        self.cur.execute(select_sql, (venue,))
        records = self.cur.fetchall()
        
        if records:
            columns=["venue", "paper_openreview_id", "arxiv_id", "title"]
            records_df = pd.DataFrame(records, columns=columns)
            
            delete_sql = """
            DELETE FROM openreview_arxiv WHERE venue = %s;
            """
            self.cur.execute(delete_sql, (venue,))
            self.conn.commit()
            
            print(f"Deleted {len(records)} records from 'openreview_arxiv' where venue = {venue}.")
            return records_df
        else:
            print(f"No records found in 'openreview_arxiv' for venue = {venue}.")
            return None
    
    def get_openreview_neighboring_arxivs(self, openreview_id: str) -> None | pd.DataFrame:
        self.cur.execute("""
        SELECT venue, paper_openreview_id, arxiv_id, title FROM openreview_arxiv
        WHERE paper_openreview_id = %s;
        """, (openreview_id,))
        
        openreview_arxiv = self.cur.fetchall()
        
        if openreview_arxiv is not None:
            openreview_arxiv_df = pd.DataFrame(openreview_arxiv, columns=["venue", "paper_openreview_id", "arxiv_id", "title"])
            return openreview_arxiv_df
        else:
            return None
        
    def get_arxiv_neighboring_openreviews(self, arxiv_id: str) -> None | pd.DataFrame:
        self.cur.execute("""
        SELECT venue, paper_openreview_id, arxiv_id, title FROM openreview_arxiv
        WHERE arxiv_id = %s;
        """, (arxiv_id,))
        
        openreview_arxiv = self.cur.fetchall()
        
        if openreview_arxiv is not None:
            openreview_arxiv_df = pd.DataFrame(openreview_arxiv, columns=["venue", "paper_openreview_id", "arxiv_id", "title"])
            return openreview_arxiv_df
        else:
            return None
    
    def get_openreview_arxiv_by_venue(self, venue: str) -> None | pd.DataFrame:
        self.cur.execute("""
        SELECT venue, paper_openreview_id, arxiv_id, title FROM openreview_arxiv
        WHERE venue = %s;
        """, (venue,))
        
        openreview_arxiv = self.cur.fetchall()
        
        if openreview_arxiv is not None:
            openreview_arxiv_df = pd.DataFrame(openreview_arxiv, columns=["venue", "paper_openreview_id", "arxiv_id", "title"])
            return openreview_arxiv_df
        else:
            return None
    
    def get_all_openreview_arxiv(self) -> None | pd.DataFrame:
        self.cur.execute("""
        SELECT venue, paper_openreview_id, arxiv_id, title FROM openreview_arxiv;
        """)
        
        openreview_arxiv = self.cur.fetchall()
        
        if openreview_arxiv is not None:
            openreview_arxiv_df = pd.DataFrame(openreview_arxiv, columns=["venue", "paper_openreview_id", "arxiv_id", "title"])
            return openreview_arxiv_df
        else:
            return None
        
    def check_openreview_arxiv_exists(self, venue: str, openreview_id: str) -> bool:
        self.cur.execute("""
        SELECT 1 FROM openreview_arxiv
        WHERE venue = %s AND paper_openreview_id = %s;
        """, (venue, openreview_id))
        
        return self.cur.fetchone() is not None
    
    def construct_openreview_arxiv_table_from_api(self, venue: str) -> None:
        # crawl openreview arxiv data from openreview.net
        print(f"Crawling openreview arxiv data for venue: {venue}...")
        openreview_arxiv_data = self.openreview_crawler.crawl_openreview_arxiv_data_from_api(venue)
        
        # insert data into openreview_arxiv table
        if len(openreview_arxiv_data) > 0:
            print("Inserting data into 'openreview_arxiv' table...")
            for data in tqdm(openreview_arxiv_data):
                self.insert_openreview_arxiv(**data)
        else:
            print("No new openreview arxiv data to insert.")
    
    def construct_openreview_arxiv_table_from_csv(self, venue: str, csv_file: str) -> None:
        # read openreview arxiv data from csv file
        print(f"Reading openreview arxiv data from {csv_file}...")
        openreview_arxiv_data = pd.read_csv(csv_file).to_dict(orient='records')
        
        # insert data into openreview_arxiv table
        if len(openreview_arxiv_data) > 0:
            print("Inserting data into 'openreview_arxiv' table...")
            for data in tqdm(openreview_arxiv_data):
                self.insert_openreview_arxiv(**data)
        else:
            print("No new openreview arxiv data to insert.")
    
    def construct_openreview_arxiv_table_from_json(self, venue: str, json_file: str) -> None:
        # read openreview arxiv data from json file
        print(f"Reading openreview arxiv data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            openreview_arxiv_data = json.load(f)
        
        # insert data into openreview_arxiv table
        if len(openreview_arxiv_data) > 0:
            print("Inserting data into 'openreview_arxiv' table...")
            for data in tqdm(openreview_arxiv_data):
                self.insert_openreview_arxiv(**data)
        else:
            print("No new openreview arxiv data to insert.")