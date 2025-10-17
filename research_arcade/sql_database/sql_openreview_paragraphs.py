from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import os
import psycopg2
from psycopg2.extras import Json

class SQLOpenReviewParagraphs:
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
        self.create_paragraphs_table()
        
    def create_paragraphs_table(self) -> None:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS openreview_paragraphs (
            venue TEXT,
            paper_openreview_id VARCHAR(255),
            paragraph_idx INTEGER CHECK (paragraph_idx >= 0),
            section TEXT,
            content TEXT,
            PRIMARY KEY (venue, paper_openreview_id, paragraph_idx)
        );
        """
        # Execute the SQL to create the table
        self.cur.execute(create_table_sql)
        
    def insert_paragraph(self, venue: str, paper_openreview_id: str, paragraph_idx: int, section: str, content: str) -> None | tuple:
        insert_sql = """
        INSERT INTO openreview_paragraphs (venue, paper_openreview_id, paragraph_idx, section, content)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (venue, paper_openreview_id, paragraph_idx) DO NOTHING
        RETURNING (venue, paper_openreview_id, paragraph_idx);
        """
        
        # Execute the insertion query
        self.cur.execute(insert_sql, (self._clean_string(venue), self._clean_string(paper_openreview_id), paragraph_idx, self._clean_string(section), self._clean_string(content)))
        
        # Get the inserted review id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
    
    def get_all_paragraphs(self, is_all_features: bool = False):
        if is_all_features:
            select_sql = """
            SELECT * FROM openreview_paragraphs;
            """
            self.cur.execute(select_sql)
            row = self.cur.fetchall()

            if not row:
                print(f"No paragraph found in openreview_paragraphs table.")
                return None
            else:
                columns = ['venue', 'paper_openreview_id', 'paragraph_idx', 'section', 'content']
                # original_record = dict(zip(columns, row))
                paragraph_df = pd.DataFrame(row, columns=columns)
                return paragraph_df
        else:
            select_sql = """
            SELECT venue, paper_openreview_id, paragraph_idx, section FROM openreview_paragraphs;
            """
            self.cur.execute(select_sql)
            row = self.cur.fetchall()

            if not row:
                print(f"No paragraph found in openreview_paragraphs table.")
                return None
            else:
                columns = ['venue', 'paper_openreview_id', 'paragraph_idx', 'section']
                # original_record = dict(zip(columns, row))
                paragraph_df = pd.DataFrame(row, columns=columns)
                return paragraph_df
    
    def get_paragraphs_by_paper_id(self, paper_openreview_id: str) -> None | pd.DataFrame:
        # Query to select the current record using primary key
        select_sql = """
        SELECT * FROM openreview_paragraphs WHERE paper_openreview_id = %s;
        """
        self.cur.execute(select_sql, (paper_openreview_id,))
        row = self.cur.fetchall()

        if not row:
            print(f"No paragraph found with paper_openreview_id {paper_openreview_id}.")
            return None
        else:
            columns = ['venue', 'paper_openreview_id', 'paragraph_idx', 'section', 'content']
            # original_record = dict(zip(columns, row))
            paragraph_df = pd.DataFrame(row, columns=columns)
            return paragraph_df
    
    def get_paragraphs_by_venue(self, venue: str) -> None | pd.DataFrame:
        # Query to select all records for a specific venue
        select_sql = """
        SELECT * FROM openreview_paragraphs WHERE venue = %s;
        """
        self.cur.execute(select_sql, (venue,))
        rows = self.cur.fetchall()
        
        if not rows:
            print(f"No paragraphs found in venue {venue}.")
            return None
        else:
            columns = ['venue', 'paper_openreview_id', 'paragraph_idx', 'section', 'content']
            # original_record = dict(zip(columns, row))
            paragraphs_df = pd.DataFrame(rows, columns=columns)
            return paragraphs_df
        
    def delete_paragraphs_by_venue(self, venue: str) -> None | pd.DataFrame:
        select_sql = """
        SELECT * FROM openreview_paragraphs WHERE venue = %s;
        """
        self.cur.execute(select_sql, (venue,))
        rows = self.cur.fetchall()

        if rows:
            columns = ['venue', 'paper_openreview_id', 'paragraph_idx', 'section', 'content']
            # original_record = dict(zip(columns, row))
            paragraphs_df = pd.DataFrame(rows, columns=columns)

            delete_sql = """
            DELETE FROM openreview_paragraphs WHERE venue = %s;
            """
            self.cur.execute(delete_sql, (venue,))
            self.conn.commit()

            print(f"All paragraphs in venue {venue} deleted successfully.")
            return paragraphs_df
        else:
            print(f"No paragraphs found in venue {venue}.")
            return None
        
    def delete_paragraphs_by_paper_id(self, paper_openreview_id: str) -> None | pd.DataFrame:
        select_sql = """
        SELECT * FROM openreview_paragraphs WHERE paper_openreview_id = %s;
        """
        self.cur.execute(select_sql, (paper_openreview_id,))
        rows = self.cur.fetchall()
        
        if rows:
            columns = ['venue', 'paper_openreview_id', 'paragraph_idx', 'section', 'content']
            # original_record = dict(zip(columns, row))
            paragraphs_df = pd.DataFrame(rows, columns=columns)

            delete_sql = """
            DELETE FROM openreview_paragraphs WHERE paper_openreview_id = %s;
            """
            self.cur.execute(delete_sql, (paper_openreview_id,))
            self.conn.commit()

            print(f"All paragraphs in paper {paper_openreview_id} deleted successfully.")
            return paragraphs_df
        else:
            print(f"No paragraphs found in paper {paper_openreview_id}.")
            return None
    
    def construct_paragraphs_table_from_api(self, venue: str, pdf_dir: str, filter_list: list, log_file: str, 
                                            is_paper = True, is_revision = True, is_pdf_delete: bool = True):
        # crawl paragraph data from openreview API
        print("Crawling paragraph data from OpenReview API...")
        paragraph_data = self.openreview_crawler.crawl_paragraph_data_from_api(venue, pdf_dir, filter_list, log_file,
                                                                           is_paper, is_revision, is_pdf_delete)
        # insert data into openreview_paragraphs table
        if len(paragraph_data) > 0:
            print("Inserting data into 'openreview_paragraphs' table...")
            for data in tqdm(paragraph_data):
                self.insert_paragraph(**data)
        else:
            print("No new paragraph data to insert.")
            
    def construct_paragraphs_table_from_csv(self, csv_file: str):
        if not os.path.exists(csv_file):
            print(f"File {csv_file} not exists")
            return False
        # read paragraph data from csv file
        print(f"Reading paragraph data from {csv_file}...")
        paragraph_data = pd.read_csv(csv_file).to_dict(orient='records')
        
        # insert data into openreview_paragraphs table
        if len(paragraph_data) > 0:
            print("Inserting data into 'openreview_paragraphs' table...")
            for data in tqdm(paragraph_data):
                self.insert_paragraph(**data)
            return True
        else:
            print("No new paragraph data to insert.")
            return False
            
    def construct_paragraphs_table_from_json(self, json_file: str):
        if not os.path.exists(json_file):
            print(f"File {json_file} not exists")
            return False
        # read insert_paragraph data from json file
        print(f"Reading paragraph data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            paragraph_data = json.load(f)
        
        # insert data into openreview_paragraphs table
        if len(paragraph_data) > 0:
            print("Inserting data into 'openreview_paragraphs' table...")
            for data in tqdm(paragraph_data):
                self.insert_paragraph(**data)
            return True
        else:
            print("No new insert_paragraph data to insert.")
            return False
            
    def _clean_string(self, s: str) -> str:
        if isinstance(s, str):
            return s.replace('\x00', '')
        return s