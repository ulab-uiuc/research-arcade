from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import psycopg2
import os
from typing import Optional

class SQLOpenReviewPapersAuthors:
    def __init__(self, host: str, dbname: str, user: str, password: str, port: str) -> None:
        # Store connection and cursor for reuse
        self.conn = psycopg2.connect(
            host=host, dbname=dbname,
            user=user, password=password, port=port
        )
        # Enable autocommit
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        self.openreview_crawler = OpenReviewCrawler()
        self.create_papers_authors_table()
        
    def create_papers_authors_table(self) -> None:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS openreview_papers_authors (
            venue TEXT,
            paper_openreview_id VARCHAR(255),
            author_openreview_id VARCHAR(255),
            PRIMARY KEY (venue, paper_openreview_id, author_openreview_id)
        );
        """
        self.cur.execute(create_table_sql)
        
    def insert_paper_authors(self, venue: str, paper_openreview_id: str, author_openreview_id: str) -> Optional[tuple]:
        insert_sql = """
        INSERT INTO openreview_papers_authors (venue, paper_openreview_id, author_openreview_id)
        VALUES (%s, %s, %s)
        ON CONFLICT (venue, paper_openreview_id, author_openreview_id) DO NOTHING
        RETURNING (venue, paper_openreview_id, author_openreview_id);
        """
        
        venue = self._clean_string(venue)
        paper_openreview_id = self._clean_string(paper_openreview_id)
        author_openreview_id = self._clean_string(author_openreview_id)
        
        # Execute the insertion query
        self.cur.execute(insert_sql, (venue, paper_openreview_id, author_openreview_id))
        
        # Get the inserted paper's openreview id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
    
    def delete_paper_author_by_id(self, paper_openreview_id: str, author_openreview_id: str) -> Optional[pd.DataFrame]:
        # search for records with the given arxiv_id and delete them
        select_sql = """
        SELECT * FROM openreview_papers_authors WHERE paper_openreview_id = %s AND author_openreview_id = %s;
        """
        self.cur.execute(select_sql, (paper_openreview_id, author_openreview_id))
        records = self.cur.fetchall()
        
        if records:
            columns=["venue", "paper_openreview_id", "author_openreview_id"]
            records_df = pd.DataFrame(records, columns=columns)
            
            delete_sql = """
            DELETE FROM openreview_papers_authors WHERE paper_openreview_id = %s AND author_openreview_id = %s;
            """
            self.cur.execute(delete_sql, (paper_openreview_id, author_openreview_id))
            self.conn.commit()
            
            print(f"Deleted {len(records)} records from 'openreview_papers_authors' with paper_openreview_id = {paper_openreview_id} and author_openreview_id = {author_openreview_id}.")
            return records_df
        else:
            print(f"No records found in 'openreview_papers_authors' with paper_openreview_id = {paper_openreview_id} and author_openreview_id = {author_openreview_id}.")
            return None
    
    def delete_paper_author_by_paper_id(self, paper_openreview_id: str) -> Optional[pd.DataFrame]:
        # search for records with the given arxiv_id and delete them
        select_sql = """
        SELECT * FROM openreview_papers_authors WHERE paper_openreview_id = %s;
        """
        self.cur.execute(select_sql, (paper_openreview_id,))
        records = self.cur.fetchall()
        
        if records:
            columns=["venue", "paper_openreview_id", "author_openreview_id"]
            records_df = pd.DataFrame(records, columns=columns)
            
            delete_sql = """
            DELETE FROM openreview_papers_authors WHERE paper_openreview_id = %s;
            """
            self.cur.execute(delete_sql, (paper_openreview_id,))
            self.conn.commit()
            
            print(f"Deleted {len(records)} records from 'openreview_papers_authors' with paper_openreview_id = {paper_openreview_id}.")
            return records_df
        else:
            print(f"No records found in 'openreview_papers_authors' with paper_openreview_id = {paper_openreview_id}.")
            return None
        
    def delete_paper_author_by_author_id(self, author_openreview_id: str) -> Optional[pd.DataFrame]:
        # search for records with the given arxiv_id and delete them
        select_sql = """
        SELECT * FROM openreview_papers_authors WHERE author_openreview_id = %s;
        """
        self.cur.execute(select_sql, (author_openreview_id,))
        records = self.cur.fetchall()
        
        if records:
            columns=["venue", "paper_openreview_id", "author_openreview_id"]
            records_df = pd.DataFrame(records, columns=columns)
            
            delete_sql = """
            DELETE FROM openreview_papers_authors WHERE author_openreview_id = %s;
            """
            self.cur.execute(delete_sql, (author_openreview_id,))
            self.conn.commit()
            
            print(f"Deleted {len(records)} records from 'openreview_papers_authors' with author_openreview_id = {author_openreview_id}.")
            return records_df
        else:
            print(f"No records found in 'openreview_papers_authors' with author_openreview_id = {author_openreview_id}.")
            return None
    
    def delete_papers_authors_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        # search the row based on primary key
        select_sql = """
        SELECT venue, paper_openreview_id, author_openreview_id FROM openreview_papers_authors WHERE venue = %s;
        """
        self.cur.execute(select_sql, (venue,))
        rows = self.cur.fetchall()

        if rows:
            columns = ['venue', 'paper_openreview_id', 'author_openreview_id']
            # papers_authors_dict = dict(zip(columns, rows))
            papers_authors_df = pd.DataFrame(rows, columns=columns)

            delete_sql = """
            DELETE FROM openreview_papers_authors WHERE venue = %s;
            """
            self.cur.execute(delete_sql, (venue,))
            self.conn.commit()

            print(f"All connections in venue {venue} deleted successfully.")
            return papers_authors_df
        else:
            print(f"No connections found in venue {venue}.")
            return None
    
    def get_paper_neighboring_authors(self, paper_openreview_id: str) -> Optional[pd.DataFrame]:
        self.cur.execute("""
        SELECT venue, paper_openreview_id, author_openreview_id FROM openreview_papers_authors
        WHERE paper_openreview_id = %s;
        """, (paper_openreview_id,))
        
        paper_neighboring_authors = self.cur.fetchall()
        
        if paper_neighboring_authors is not None:
            paper_neighboring_authors_df = pd.DataFrame(paper_neighboring_authors, columns=["venue", "paper_openreview_id", "author_openreview_id"])
            return paper_neighboring_authors_df
        else:
            return None
        
    def get_author_neighboring_papers(self, author_openreview_id: str) -> Optional[pd.DataFrame]:
        self.cur.execute("""
        SELECT venue, paper_openreview_id, author_openreview_id FROM openreview_papers_authors
        WHERE author_openreview_id = %s;
        """, (author_openreview_id,))
        
        author_neighboring_papers = self.cur.fetchall()
        
        if author_neighboring_papers is not None:
            author_neighboring_papers_df = pd.DataFrame(author_neighboring_papers, columns=["venue", "paper_openreview_id", "author_openreview_id"])
            return author_neighboring_papers_df
        else:
            return None
    
    def get_papers_authors_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        self.cur.execute("""
        SELECT venue, paper_openreview_id, author_openreview_id FROM openreview_papers_authors
        WHERE venue = %s;
        """, (venue,))
        
        openreview_papers_authors = self.cur.fetchall()
        
        if openreview_papers_authors is not None:
            papers_authors_df = pd.DataFrame(openreview_papers_authors, columns=["venue", "paper_openreview_id", "author_openreview_id"])
            return papers_authors_df
        else:
            return None
    
    def get_all_papers_authors(self) -> Optional[pd.DataFrame]:
        select_query = """
        SELECT venue, paper_openreview_id, author_openreview_id
        FROM openreview_papers_authors ORDER BY paper_openreview_id ASC
        """
        self.cur.execute(select_query)
            
        # Fetch all the results
        openreview_papers_authors = self.cur.fetchall()
        
        # Return the result as a list of tuples (venue, author_openreview_id, author_full_name)
        authors_df = pd.DataFrame(openreview_papers_authors, columns=["venue", "paper_openreview_id", "author_openreview_id"])
        return authors_df
    
    def check_paper_author_exists(self, paper_openreview_id: str, author_openreview_id: str) -> bool:
        self.cur.execute("""
        SELECT 1 FROM openreview_papers_authors
        WHERE paper_openreview_id = %s AND author_openreview_id = %s 
        LIMIT 1;
        """, (paper_openreview_id, author_openreview_id))
        
        result = self.cur.fetchone()

        return result is not None
    
    def construct_papers_authors_table_from_api(self, venue: str) -> bool:
        # crawl the data from openreview API
        print(f"Crawling papers-authors data for venue {venue} from OpenReview API...")
        papers_authors_data = self.openreview_crawler.crawl_papers_authors_data_from_api(venue)
        
        if len(papers_authors_data) > 0:
            print(f"Inserting papers-authors data for venue {venue}...")
            for data in tqdm(papers_authors_data, desc=f"Inserting papers-authors data for venue {venue}"):
                self.insert_paper_authors(**data)
            return True
        else:
            print(f"No papers-authors data found for venue {venue}.")
            return False
            
    def construct_papers_authors_table_from_csv(self, csv_file: str) -> bool:
        # read the data from csv file
        papers_authors_data = pd.read_csv(csv_file).to_dict(orient='records')
        
        if len(papers_authors_data) > 0:
            print(f"Inserting papers-authors data from {csv_file}...")
            for data in tqdm(papers_authors_data, desc=f"Inserting papers-authors data from {csv_file}"):
                self.insert_paper_authors(**data)
            return True
        else:
            print(f"No papers-authors data found in {csv_file}.")
            return False
            
    def construct_papers_authors_table_from_json(self, json_file: str) -> bool:
        if not os.path.exists(json_file):
            return False
        else:
            # read the data from json file
            with open(json_file, 'r', encoding='utf-8') as f:
                papers_authors_data = json.load(f)
            
            if len(papers_authors_data) > 0:
                print(f"Inserting papers-authors data from {json_file}...")
                for data in tqdm(papers_authors_data, desc=f"Inserting papers-authors data from {json_file}"):
                    self.insert_paper_authors(**data)
                return True
            else:
                print(f"No papers-authors data found in {json_file}.")
                return False
            
    def _clean_string(self, s: str) -> str:
        if isinstance(s, str):
            return s.replace('\x00', '')
        return s