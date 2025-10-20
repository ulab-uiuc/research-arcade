from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import os
import psycopg2

class SQLOpenReviewPapers:
    def __init__(self, host: str = "localhost", dbname: str = "iclr_openreview_database", user: str = "jingjunx", password: str = "", port: str = "5432"):
        self.conn = psycopg2.connect(
            host=host,
            dbname=dbname,
            user=user,
            password=password,
            port=port
        )
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        self.openreview_crawler = OpenReviewCrawler()
        self.create_papers_table()
        
    def create_papers_table(self) -> None:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS openreview_papers (
            venue TEXT,
            paper_openreview_id VARCHAR(255),
            title TEXT,
            abstract TEXT,
            paper_decision TEXT,
            paper_pdf_link TEXT,
            PRIMARY KEY (venue, paper_openreview_id)
        );
        """
        self.cur.execute(create_table_sql)
        
    def insert_paper(self, venue: str, paper_openreview_id: str, title: str, abstract: str,
                    paper_decision: str, paper_pdf_link: str) -> None | tuple:
        insert_sql = """
        INSERT INTO openreview_papers (venue, paper_openreview_id, title, abstract, paper_decision, paper_pdf_link)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (venue, paper_openreview_id) DO NOTHING
        RETURNING (venue, paper_openreview_id);
        """
        
        venue = self._clean_string(venue)
        paper_openreview_id = self._clean_string(paper_openreview_id)
        title = self._clean_string(title)
        abstract = self._clean_string(abstract)
        paper_decision = self._clean_string(paper_decision)
        paper_pdf_link = self._clean_string(paper_pdf_link)

        # Execute the insertion query
        self.cur.execute(insert_sql, (venue, paper_openreview_id, title, abstract, paper_decision, paper_pdf_link))
        
        # Get the inserted paper id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
    
    def delete_paper_by_id(self, paper_openreview_id: str) -> None | pd.DataFrame:
        # search for the row based on primary key
        select_sql = """
        SELECT * FROM openreview_papers WHERE paper_openreview_id = %s;
        """
        self.cur.execute(select_sql, (paper_openreview_id,))
        row = self.cur.fetchall()

        if row:
            columns = ['venue', 'paper_openreview_id', 'title', 'abstract', 
                       'paper_decision', 'paper_pdf_link']
            # paper_dict = dict(zip(columns, row))
            paper_df = pd.DataFrame(row, columns=columns)

            delete_sql = """
            DELETE FROM openreview_papers WHERE paper_openreview_id = %s;
            """
            self.cur.execute(delete_sql, (paper_openreview_id,))
            self.conn.commit()

            print(f"Paper with paper_openreview_id {paper_openreview_id} deleted successfully.")
            return paper_df
        else:
            print(f"No paper found with paper_openreview_id {paper_openreview_id}.")
            return None
        
    def delete_papers_by_venue(self, venue: str) -> None | pd.DataFrame:
        # search for the row based on primary key
        select_sql = """
        SELECT * FROM openreview_papers WHERE venue = %s;
        """
        self.cur.execute(select_sql, (venue,))
        row = self.cur.fetchall()

        if row:
            columns = ['venue', 'paper_openreview_id', 'title', 'abstract', 
                       'paper_decision', 'paper_pdf_link']
            # paper_dict = dict(zip(columns, row))
            paper_df = pd.DataFrame(row, columns=columns)

            delete_sql = """
            DELETE FROM openreview_papers WHERE venue = %s;
            """
            self.cur.execute(delete_sql, (venue,))
            self.conn.commit()

            print(f"All openreview_papers in venue {venue} deleted successfully.")
            return paper_df
        else:
            print(f"No openreview_papers found in venue {venue}.")
            return None
        
    def update_paper(self, venue: str, paper_openreview_id: str, title: str, abstract: str,
                    paper_decision: str, paper_pdf_link: str) -> None | pd.DataFrame:
        # Query to select the current record using primary key
        select_sql = """
        SELECT * FROM openreview_papers WHERE paper_openreview_id = %s AND venue = %s;
        """
        # find the row in the table
        self.cur.execute(select_sql, (paper_openreview_id, venue,))
        row = self.cur.fetchone()

        if not row:
            print(f"No paper found with paper_openreview_id {paper_openreview_id}.")
            return None
        else:
            columns = ['venue', 'paper_openreview_id', 'title', 'abstract',
                    'paper_decision', 'paper_pdf_link']
            # original_record = dict(zip(columns, row))
            paper_df = pd.DataFrame([row], columns=columns)
            
            update_sql = """
            UPDATE openreview_papers
            SET venue = %s,
                title = %s,
                abstract = %s,
                paper_decision = %s,
                paper_pdf_link = %s
            WHERE paper_openreview_id = %s;
            """
            
            venue = self._clean_string(venue)
            paper_openreview_id = self._clean_string(paper_openreview_id)
            title = self._clean_string(title)
            abstract = self._clean_string(abstract)
            paper_decision = self._clean_string(paper_decision)
            paper_pdf_link = self._clean_string(paper_pdf_link)
            
            self.cur.execute(update_sql, (venue, title, abstract, paper_decision,
                                        paper_pdf_link, paper_openreview_id))

            self.conn.commit()
            
            print(f"Paper with paper_openreview_id {paper_openreview_id} updated successfully.")
            return paper_df
    
    def get_paper_by_id(self, paper_openreview_id: str) -> None | pd.DataFrame:
        # Query to select the current record using primary key
        select_sql = """
        SELECT * FROM openreview_papers WHERE paper_openreview_id = %s;
        """
        # find the row in the table
        self.cur.execute(select_sql, (paper_openreview_id,))
        row = self.cur.fetchall()
        
        if not row:
            print(f"No paper found with paper_openreview_id {paper_openreview_id}.")
            return None
        else:
            columns = ['venue', 'paper_openreview_id', 'title', 'abstract',
                    'paper_decision', 'paper_pdf_link']
            # original_record = dict(zip(columns, row))
            paper_df = pd.DataFrame(row, columns=columns)
            return paper_df
        
    def get_papers_by_venue(self, venue: str) -> None | pd.DataFrame:
        # Query to select all records for a specific venue
        select_sql = """
        SELECT * FROM openreview_papers WHERE venue = %s;
        """
        self.cur.execute(select_sql, (venue,))
        rows = self.cur.fetchall()
        
        if not rows:
            print(f"No openreview_papers found in venue {venue}.")
            return None
        else:
            columns = ['venue', 'paper_openreview_id', 'title', 'abstract',
                    'paper_decision', 'paper_pdf_link']
            # original_record = dict(zip(columns, row))
            papers_df = pd.DataFrame(rows, columns=columns)
            return papers_df
    
    def get_all_papers(self, is_all_features: bool = False) -> None | pd.DataFrame:
        if is_all_features:
            # Select query to get paper_openreview_id, title, and author_full_names
            select_query = """
            SELECT venue, paper_openreview_id, title, abstract, paper_decision, paper_pdf_link
            FROM openreview_papers;
            """
            self.cur.execute(select_query)
            
            # Fetch all the results
            papers = self.cur.fetchall()
            
            # Return the result as a list of tuples (paper_openreview_id, title, author_full_names)
            papers_df = pd.DataFrame(papers, columns=["venue", "paper_openreview_id", "title", "abstract", "paper_decision", "paper_pdf_link"])
            return papers_df
        else:
            # Select query to get paper_openreview_id, title, and author_full_names
            select_query = """
            SELECT venue, paper_openreview_id, title
            FROM openreview_papers;
            """
            self.cur.execute(select_query)
            
            # Fetch all the results
            papers = self.cur.fetchall()
            
            # Return the result as a list of tuples (paper_openreview_id, title, author_full_names)
            papers_df = pd.DataFrame(papers, columns=["venue", "paper_openreview_id", "title"])
            return papers_df
    
    def check_paper_exists(self, paper_openreview_id: str) -> bool | None:
        self.cur.execute("SELECT 1 FROM openreview_papers WHERE paper_openreview_id = %s LIMIT 1;", (paper_openreview_id,))
        result = self.cur.fetchone()

        return result is not None
    
    def construct_papers_table_from_api(self, venue: str):
        # crawl paper data from openreview API
        print("Crawling paper data from OpenReview API...")
        paper_data = self.openreview_crawler.crawl_paper_data_from_api(venue)
        
        # insert data into openreview_papers table
        if len(paper_data) > 0:
            print("Inserting data into 'openreview_papers' table...")
            for data in tqdm(paper_data):
                self.insert_paper(**data)
        else:
            print("No new paper data to insert.")
            
    def construct_papers_table_from_csv(self, csv_file: str):
        if not os.path.exists(csv_file):
            print(f"File {csv_file} not exists")
            return False
        # read paper data from csv file
        print(f"Reading paper data from {csv_file}...")
        paper_data = pd.read_csv(csv_file).to_dict(orient='records')
        
        # insert data into openreview_papers table
        if len(paper_data) > 0:
            print("Inserting data into 'openreview_papers' table...")
            for data in tqdm(paper_data):
                self.insert_paper(**data)
            return True
        else:
            print("No new paper data to insert.")
            return False
            
    def construct_papers_table_from_json(self, json_file: str):
        if not os.path.exists(json_file):
            print(f"File {json_file} not exists")
            return False
        # read paper data from json file
        print(f"Reading paper data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        
        # insert data into openreview_papers table
        if len(paper_data) > 0:
            print("Inserting data into 'openreview_papers' table...")
            for data in tqdm(paper_data):
                self.insert_paper(**data)
            return True
        else:
            print("No new paper data to insert.")
            return False
            
    def _clean_string(self, s: str) -> str:
        if isinstance(s, str):
            return s.replace('\x00', '')
        return s