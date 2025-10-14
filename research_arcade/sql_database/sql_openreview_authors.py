from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import psycopg2
from psycopg2.extras import Json

class SQLOpenReviewAuthors:
    def __init__(self, host: str = "localhost", dbname: str = "iclr_openreview_database", user: str = "jingjunx", password: str = "", port: str = "5432"):
        self.conn = psycopg2.connect(
            host=host,
            dbname=dbname,
            user=user,
            password=password,
            port=port
        )
        self.cursor = self.conn.cursor()
        self.openreview_crawler = OpenReviewCrawler()
        self.create_author_table()
        
    def create_author_table(self) -> None:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS authors (
            venue TEXT,
            author_openreview_id VARCHAR(255),
            author_full_name TEXT,
            email TEXT,
            affiliation TEXT,
            homepage TEXT,
            dblp TEXT,
            PRIMARY KEY (venue, author_openreview_id)
        );
        """
        # Execute the SQL to create the table
        self.cur.execute(create_table_sql)
        print("Table 'authors' created successfully.")
        
    def insert_author(self, venue: str, author_openreview_id: str, author_full_name: str, email: str, 
                      affiliation: str, homepage: str, dblp: str) -> None | tuple:
        insert_sql = """
        INSERT INTO authors (venue, author_openreview_id, author_full_name, email, affiliation, homepage, dblp)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (venue, author_openreview_id) DO NOTHING
        RETURNING (venue, author_openreview_id);
        """
        
        # Execute the insertion query
        self.cur.execute(insert_sql, (venue, author_openreview_id, author_full_name, email, affiliation, homepage, dblp))
        
        # Get the inserted author id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
    
    def delete_author_by_id(self, author_openreview_id: str) -> None | pd.DataFrame:
        # search the row based on primary key
        select_sql = """
        SELECT * FROM authors WHERE author_openreview_id = %s;
        """
        self.cur.execute(select_sql, (author_openreview_id,))
        row = self.cur.fetchall()

        if row:
            columns = ['venue', 'author_openreview_id', 'author_full_name', 'email', 
                       'affiliation', 'homepage', 'dblp']
            # author_dict = dict(zip(columns, row))
            author_df = pd.DataFrame(row, columns=columns)

            delete_sql = """
            DELETE FROM authors WHERE author_openreview_id = %s;
            """
            self.cur.execute(delete_sql, (author_openreview_id,))
            self.conn.commit()

            print(f"Author with author_openreview_id {author_openreview_id} deleted successfully.")
            return author_df
        else:
            print(f"No author found with author_openreview_id {author_openreview_id}.")
            return None
        
    def delete_authors_by_venue(self, venue: str) -> None | pd.DataFrame:
        # search the row based on primary key
        select_sql = """
        SELECT * FROM authors WHERE venue = %s;
        """
        self.cur.execute(select_sql, (venue,))
        row = self.cur.fetchall()

        if row:
            columns = ['venue', 'author_openreview_id', 'author_full_name', 'email', 
                       'affiliation', 'homepage', 'dblp']
            # author_dict = dict(zip(columns, row))
            author_df = pd.DataFrame(row, columns=columns)

            delete_sql = """
            DELETE FROM authors WHERE venue = %s;
            """
            self.cur.execute(delete_sql, (venue,))
            self.conn.commit()

            print(f"All authors in venue {venue} deleted successfully.")
            return author_df
        else:
            print(f"No authors found in venue {venue}.")
            return None
        
    def update_author(self, venue: str, author_openreview_id: str, author_full_name: str, email: str, 
                      affiliation: str, homepage: str, dblp: str) -> None | pd.DataFrame:
        # Query to select the current record using primary key
        select_sql = """
        SELECT * FROM authors WHERE author_openreview_id = %s AND venue = %s;
        """
        self.cur.execute(select_sql, (author_openreview_id, venue,))
        row = self.cur.fetchone()

        if not row:
            print(f"No author found with author_openreview_id {author_openreview_id}.")
            return None
        else:
            columns = ['venue', 'author_openreview_id', 'author_full_name', 
                       'email', 'affiliation', 'homepage', 'dblp']
            # original_record = dict(zip(columns, row))
            author_df = pd.DataFrame(row, columns=columns)
            
            update_sql = """
            UPDATE authors
            SET venue = %s,
                author_full_name = %s,
                email = %s,
                affiliation = %s,
                homepage = %s,
                dblp = %s
            WHERE author_openreview_id = %s;
            """

            self.cur.execute(update_sql, (venue, author_full_name, email, affiliation, homepage, dblp, author_openreview_id))
            self.conn.commit()

            print(f"Author with author_openreview_id {author_openreview_id} updated successfully.")
            return author_df
    
    def get_author_by_id(self, author_openreview_id: str) -> None | pd.DataFrame:
        # Query to select the current record using primary key
        select_sql = """
        SELECT * FROM authors WHERE author_openreview_id = %s;
        """
        self.cur.execute(select_sql, (author_openreview_id,))
        row = self.cur.fetchall()

        if not row:
            print(f"No author found with author_openreview_id {author_openreview_id}.")
            return None
        else:
            columns = ['venue', 'author_openreview_id', 'author_full_name', 
                       'email', 'affiliation', 'homepage', 'dblp']
            # original_record = dict(zip(columns, row))
            author_df = pd.DataFrame(row, columns=columns)
            return author_df
        
    def get_authors_by_venue(self, venue: str) -> None | pd.DataFrame:
        # Query to select all records for a specific venue
        select_sql = """
        SELECT * FROM authors WHERE venue = %s;
        """
        self.cur.execute(select_sql, (venue,))
        rows = self.cur.fetchall()
        
        if not rows:
            print(f"No authors found in venue {venue}.")
            return None
        else:
            columns = ['venue', 'author_openreview_id', 'author_full_name', 
                       'email', 'affiliation', 'homepage', 'dblp']
            # original_record = dict(zip(columns, row))
            authors_df = pd.DataFrame(rows, columns=columns)
            return authors_df
    
    def get_all_authors(self, is_all_features: bool = False) -> None | pd.DataFrame:
        if is_all_features:
            select_query = """
            SELECT venue, author_openreview_id, author_full_name, email, affiliation, homepage, dblp
            FROM authors ORDER BY author_openreview_id ASC
            """
            self.cur.execute(select_query)
            
            # Fetch all the results
            authors = self.cur.fetchall()
            
            # Return the result as a list of tuples (venue, author_openreview_id, author_full_name)
            authors_df = pd.DataFrame(authors, columns=["venue", "author_openreview_id", "author_full_name", "email", "affiliation", "homepage", "dblp"])
            return authors_df
        else:
            select_query = """
            SELECT venue, author_openreview_id, author_full_name
            FROM authors ORDER BY author_openreview_id ASC
            """
            self.cur.execute(select_query)
            
            # Fetch all the results
            authors = self.cur.fetchall()
            
            # Return the result as a list of tuples (venue, author_openreview_id, author_full_name)
            authors_df = pd.DataFrame(authors, columns=["venue", "author_openreview_id", "author_full_name"])
            return authors_df
    
    def check_author_exists(self, author_openreview_id: str) -> bool | None:
        self.cur.execute("SELECT 1 FROM authors WHERE author_openreview_id = %s LIMIT 1;", (author_openreview_id,))
        result = self.cur.fetchone()

        return result is not None
    
    def construct_authors_table_from_api(self, venue: str):
        # crawl author data from openreview API
        print("Crawling author data from OpenReview API...")
        author_data = self.openreview_crawler.crawl_author_data_from_api(venue)
        
        # insert data into authors table
        if len(author_data) > 0:
            print("Inserting data into 'authors' table...")
            for data in tqdm(author_data):
                self.insert_author(**data)
        else:
            print("No new author data to insert.")
            
    def construct_authors_table_from_json(self, json_file: str):
        # read author data from json file
        print(f"Reading authors data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            author_data = json.load(f)
        
        # insert data into authors table
        if len(author_data) > 0:
            print("Inserting data into 'authors' table...")
            for data in tqdm(author_data):
                self.insert_author(**data)
        else:
            print("No new author data to insert.")
            
    def construct_authors_table_from_csv(self, csv_file: str):
        # read author data from csv file
        print(f"Reading authors data from {csv_file}...")
        author_data = pd.read_csv(csv_file).to_dict(orient='records')
        
        # insert data into authors table
        if len(author_data) > 0:
            print("Inserting data into 'authors' table...")
            for data in tqdm(author_data):
                self.insert_author(**data)
        else:
            print("No new author data to insert.")