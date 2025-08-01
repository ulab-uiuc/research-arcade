import pandas as pd
import psycopg2
from psycopg2.extras import Json

PASSWORD = "Tina20041128"

class Database:
    def __init__(self):
        # Store connection and cursor for reuse
        self.conn = psycopg2.connect(
            host="localhost", dbname="iclr2025_openreview_database",
            user="postgres", password=PASSWORD, port="5432"
        )
        # Enable autocommit
        self.conn.autocommit = True
        self.cur = self.conn.cursor()

    # papers table
    def create_papers_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS papers (
            venue TEXT,
            paper_openreview_id VARCHAR(255) PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            paper_decision TEXT,
            paper_pdf_link TEXT
        );
        """
        self.cur.execute(create_table_sql)
        print("Table 'papers' created successfully.")
        
    def insert_paper(self, venue: str, paper_openreview_id: str, title: str, abstract: str,
                    paper_decision: str, paper_pdf_link: str):
        """
        Insert a paper into the papers table. Returns the inserted paper id or None if it fails.
        - venue: str, the venue where the paper is submitted.
        - paper_openreview_id: str, unique identifier for the paper (primary key).
        - title: str, the title of the paper.
        - abstract: str or None, the abstract of the paper (optional).
        - paper_decision: str or None, the decision for the paper (optional).
        - paper_pdf_link: str or None, a link to the paper's PDF (optional).
        """
        insert_sql = """
        INSERT INTO papers (venue, paper_openreview_id, title, abstract, paper_decision, paper_pdf_link)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (paper_openreview_id) DO NOTHING
        RETURNING paper_openreview_id;
        """

        # Execute the insertion query
        self.cur.execute(insert_sql, (venue, paper_openreview_id, title, abstract, paper_decision, paper_pdf_link))
        
        # Get the inserted paper id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
    
    def delete_paper(self, paper_openreview_id: str):
        # search for the row based on primary key
        select_sql = """
        SELECT * FROM papers WHERE paper_openreview_id = %s;
        """
        self.cur.execute(select_sql, (paper_openreview_id,))
        row = self.cur.fetchone()

        if row:
            columns = ['venue', 'paper_openreview_id', 'title', 'abstract', 
                       'paper_decision', 'paper_pdf_link']
            # paper_dict = dict(zip(columns, row))
            paper_df = pd.DataFrame(row, columns=columns)

            delete_sql = """
            DELETE FROM papers WHERE paper_openreview_id = %s;
            """
            self.cur.execute(delete_sql, (paper_openreview_id,))
            self.conn.commit()

            print(f"Paper with paper_openreview_id {paper_openreview_id} deleted successfully.")
            return paper_df
        else:
            print(f"No paper found with paper_openreview_id {paper_openreview_id}.")
            return None
        
    def update_paper(self, venue: str, paper_openreview_id: str, title: str, abstract: str,
                    paper_decision: str, paper_pdf_link: str):
        # Query to select the current record using primary key
        select_sql = """
        SELECT * FROM papers WHERE paper_openreview_id = %s;
        """
        # find the row in the table
        self.cur.execute(select_sql, (paper_openreview_id,))
        row = self.cur.fetchone()

        if not row:
            print(f"No paper found with paper_openreview_id {paper_openreview_id}.")
            return None
        else:
            columns = ['venue', 'paper_openreview_id', 'title', 'abstract',
                    'paper_decision', 'paper_pdf_link']
            # original_record = dict(zip(columns, row))
            paper_df = pd.DataFrame(row, columns=columns)
            
            update_sql = """
            UPDATE papers
            SET venue = %s,
                title = %s,
                abstract = %s,
                paper_decision = %s,
                paper_pdf_link = %s,
            WHERE paper_openreview_id = %s;
            """
            
            self.cur.execute(update_sql, (venue, title, abstract, paper_decision,
                                        paper_pdf_link, paper_openreview_id))

            self.conn.commit()
            
            print(f"Paper with paper_openreview_id {paper_openreview_id} updated successfully.")
            return paper_df
    
    def get_paper(self, paper_openreview_id: str):
        # Query to select the current record using primary key
        select_sql = """
        SELECT * FROM papers WHERE paper_openreview_id = %s;
        """
        # find the row in the table
        self.cur.execute(select_sql, (paper_openreview_id,))
        row = self.cur.fetchone()
        
        if not row:
            print(f"No paper found with paper_openreview_id {paper_openreview_id}.")
            return None
        else:
            columns = ['venue', 'paper_openreview_id', 'title', 'abstract',
                    'paper_decision', 'paper_pdf_link']
            # original_record = dict(zip(columns, row))
            paper_df = pd.DataFrame(row, columns=columns)
            return paper_df
    
    def get_all_papers(self, is_all_features: bool = False):
        if is_all_features:
            # Select query to get paper_openreview_id, title, and author_full_names
            select_query = """
            SELECT venue, paper_openreview_id, title, abstract, paper_decision, paper_pdf_link
            FROM papers;
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
            FROM papers;
            """
            self.cur.execute(select_query)
            
            # Fetch all the results
            papers = self.cur.fetchall()
            
            # Return the result as a list of tuples (paper_openreview_id, title, author_full_names)
            papers_df = pd.DataFrame(papers, columns=["venue", "paper_openreview_id", "title"])
            return papers_df
    
    def check_paper_exists(self, paper_openreview_id: str) -> bool:
        self.cur.execute("SELECT 1 FROM papers WHERE paper_openreview_id = %s LIMIT 1;", (paper_openreview_id,))
        result = self.cur.fetchone()

        return result is not None
    
    # reviews table
    def create_review_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS reviews (
            id SERIAL UNIQUE,
            venue TEXT,
            review_openreview_id VARCHAR(255) PRIMARY KEY,
            replyto_openreview_id VARCHAR(255),
            writer TEXT,
            title TEXT,
            content JSONB,
            time TEXT
        );
        """
        # Execute the SQL to create the table
        self.cur.execute(create_table_sql)
        print("Table 'reviews' created successfully.")
        
    def insert_review(self, venue: str, review_openreview_id: str, replyto_openreview_id: str, 
                      writer: str, title: str, content: dict, time: str):
        """
        Insert a review into the reviews table. Returns the inserted review id or None if it fails.
        - venue: str, the venue where the paper was submitted.
        - review_openreview_id: str, unique ID for the review (primary key).
        - replyto_openreview_id: str or None, ID for a reply (optional).
        - writer: str, the name or identity of the reviewer.
        - title: str, the title of the review.
        - content: JSON object, the content of the review.
        - time
        """
        insert_sql = """
        INSERT INTO reviews (venue, review_openreview_id, replyto_openreview_id, writer, title, content, time)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (review_openreview_id) DO NOTHING
        RETURNING review_openreview_id;
        """
        
        # clean content
        cleaned_content = self._clean_json_content(content)
        
        # Execute the insertion query
        self.cur.execute(insert_sql, (venue, review_openreview_id, replyto_openreview_id, writer, title, Json(cleaned_content), time))
        
        # Get the inserted review id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
    
    def delete_review(self, review_openreview_id: str):
        # search for the row based on primary key
        select_sql = """
        SELECT venue, review_openreview_id, replyto_openreview_id, writer, title, content, time 
        FROM reviews WHERE review_openreview_id = %s;
        """
        self.cur.execute(select_sql, (review_openreview_id,))
        row = self.cur.fetchone()

        if row:
            columns = ['venue', 'review_openreview_id', 'replyto_openreview_id', 
                       'writer', 'title', 'content', 'time']
            # review_dict = dict(zip(columns, row))
            review_df = pd.DataFrame(row, columns=columns)

            delete_sql = """
            DELETE FROM reviews WHERE review_openreview_id = %s;
            """
            self.cur.execute(delete_sql, (review_openreview_id,))
            self.conn.commit()

            print(f"Review with review_openreview_id {review_openreview_id} deleted successfully.")
            return review_df
        else:
            print(f"No review found with review_openreview_id {review_openreview_id}.")
            return None
    
    def update_review(self, venue: str, review_openreview_id: str, replyto_openreview_id: str, 
                    writer: str, title: str, content: dict, time: str):
        # Query to select the current record using primary key
        select_sql = """
        SELECT venue, review_openreview_id, replyto_openreview_id, writer, title, content, time 
        FROM reviews WHERE review_openreview_id = %s;
        """
        self.cur.execute(select_sql, (review_openreview_id,))
        row = self.cur.fetchone()

        if not row:
            print(f"No review found with review_openreview_id {review_openreview_id}.")
            return None
        else:
            columns = ['venue', 'review_openreview_id', 'replyto_openreview_id', 
                    'writer', 'title', 'content', 'time']
            # original_record = dict(zip(columns, row))
            review_df = pd.DataFrame(row, columns=columns)
            
            # SQL query to update the record
            update_sql = """
            UPDATE reviews
            SET venue = %s,
                replyto_openreview_id = %s,
                writer = %s,
                title = %s,
                content = %s,
                time = %s
            WHERE review_openreview_id = %s;
            """
            
            cleaned_content = self._clean_json_content(content)

            self.cur.execute(update_sql, (venue, replyto_openreview_id, writer, title, 
                                        Json(cleaned_content), time, review_openreview_id))
            self.conn.commit()

            print(f"Review with review_openreview_id {review_openreview_id} updated successfully.")
            return review_df
    
    def get_review(self, review_openreview_id: str):
        # Query to select the current record using primary key
        select_sql = """
        SELECT venue, review_openreview_id, replyto_openreview_id, writer, title, content, time 
        FROM reviews WHERE review_openreview_id = %s;
        """
        self.cur.execute(select_sql, (review_openreview_id,))
        row = self.cur.fetchone()

        if not row:
            print(f"No review found with review_openreview_id {review_openreview_id}.")
            return None
        else:
            columns = ['venue', 'review_openreview_id', 'replyto_openreview_id', 
                    'writer', 'title', 'content', 'time']
            # original_record = dict(zip(columns, row))
            review_df = pd.DataFrame(row, columns=columns)
            return review_df
    
    def get_all_reviews(self, is_all_features: bool = False):
        if is_all_features:
            select_query = """
            SELECT venue, review_openreview_id, replyto_openreview_id, writer, title, content, time 
            FROM reviews ORDER BY id ASC
            """
            self.cur.execute(select_query)
            
            # Fetch all the results
            reviews = self.cur.fetchall()
            
            # Return the result as a list of tuples (paper_openreview_id, title, author_full_names)
            reviews_df = pd.DataFrame(reviews, columns=["venue", "review_openreview_id", "replyto_openreview_id", "writer", "title", "content", "time"])
            return reviews_df
        else:
            select_query = """
            SELECT venue, review_openreview_id, replyto_openreview_id, title, time 
            FROM reviews ORDER BY id ASC
            """
            self.cur.execute(select_query)
            
            # Fetch all the results
            reviews = self.cur.fetchall()
            
            # Return the result as a list of tuples (paper_openreview_id, title, author_full_names)
            reviews_df = pd.DataFrame(reviews, columns=["venue", "review_openreview_id", "replyto_openreview_id", "title", "time"])
            return reviews_df
        
    def check_review_exists(self, review_openreview_id: str) -> bool:
        self.cur.execute("SELECT 1 FROM reviews WHERE review_openreview_id = %s LIMIT 1;", (review_openreview_id,))
        result = self.cur.fetchone()

        return result is not None
    
    # revisions table
    def create_revisions_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS revisions (
            id SERIAL UNIQUE,
            venue TEXT,
            original_openreview_id VARCHAR(255),
            revision_openreview_id VARCHAR(255) PRIMARY KEY,
            content JSONB,
            time TEXT
        );
        """
        self.cur.execute(create_table_sql)
        print("Table 'revisions' created successfully.")
    
    def insert_revision(self, venue: str, original_openreview_id: str, 
                        revision_openreview_id: str, content: dict, time: str):
        """
        Insert a revision into the revisions table. Returns the inserted revision id or None if it fails.
        - venue: str, the venue where the paper is submitted.
        - original_openreview_id: str, unique identifier for the revision's original paper.
        - revision_openreview_id: str, unique identifier for the revision's original paper (primary key).
        - content: json file.
        - time: text
        """
        insert_sql = """
        INSERT INTO revisions (venue, original_openreview_id, revision_openreview_id, content, time)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (revision_openreview_id) DO NOTHING
        RETURNING revision_openreview_id;
        """
        # clean revisions
        cleaned_revision_content = self._clean_json_content(content)

        # Execute the insertion query
        self.cur.execute(insert_sql, (venue, original_openreview_id, revision_openreview_id, Json(cleaned_revision_content), time))
        
        # Get the inserted paper's openreview id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
    
    def delete_revision(self, revision_openreview_id: str):
        # search for the row based on primary key
        select_sql = """
        SELECT venue, original_openreview_id, revision_openreview_id, content, time
        FROM revisions WHERE revision_openreview_id = %s;
        """
        self.cur.execute(select_sql, (revision_openreview_id,))
        row = self.cur.fetchone()

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
        
    def update_revision(self, venue: str, original_openreview_id: str, 
                        revision_openreview_id: str, content: dict, time: str):
        # Query to select the current record using primary key
        select_sql = """
        SELECT venue, original_openreview_id, revision_openreview_id, content, time
        FROM revisions WHERE revision_openreview_id = %s;
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
    
    def get_revision(self, revision_openreview_id: str):
        # Query to select the current record using primary key
        select_sql = """
        SELECT venue, original_openreview_id, revision_openreview_id, content, time
        FROM revisions WHERE revision_openreview_id = %s;
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
            # original_record = dict(zip(columns, row))
            revision_df = pd.DataFrame(row, columns=columns)
            return revision_df
    
    def get_all_revisions(self, is_all_features: bool = False):
        if is_all_features:
            select_query = """
            SELECT venue, original_openreview_id, revision_openreview_id, content, time
            FROM revisions ORDER BY id ASC
            """
            self.cur.execute(select_query)
            
            # Fetch all the results
            revisions = self.cur.fetchall()
            
            # Return the result as a list of tuples (paper_openreview_id, title, author_full_names)
            revisions_df = pd.DataFrame(revisions, columns=["venue", "original_openreview_id", "revision_openreview_id", "content", "time"])
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
    
    def check_revision_exists(self, revision_openreview_id: str) -> bool:
        self.cur.execute("SELECT 1 FROM revisions WHERE revision_openreview_id = %s LIMIT 1;", (revision_openreview_id,))
        result = self.cur.fetchone()

        return result is not None
    
    # author tables
    def create_author_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS authors (
            venue TEXT,
            author_openreview_id VARCHAR(255) PRIMARY KEY,
            author_full_name TEXT,
            email TEXT,
            affiliation TEXT,
            homepage TEXT,
            dblp TEXT
        );
        """
        # Execute the SQL to create the table
        self.cur.execute(create_table_sql)
        print("Table 'authors' created successfully.")
        
    def insert_author(self, venue: str, author_openreview_id: str, author_full_name: str, email: str, 
                      affiliation: str, homepage: str, dblp: str):
        """
        Insert an author into the authors table. Returns the inserted author id or None if it fails.
        - venue: str, the venue where the author submitted papers.
        - author_openreview_id: str, unique identifier for the author.
        - author_full_name: str, full name of the author.
        - email: str, author's email address.
        - affiliation: str or None, author's affiliation (optional).
        - homepage: str or None, author's homepage (optional).
        - dblp: str or None, author's DBLP URL (optional).
        """
        insert_sql = """
        INSERT INTO authors (venue, author_openreview_id, author_full_name, email, affiliation, homepage, dblp)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (author_openreview_id) DO NOTHING
        RETURNING author_openreview_id;
        """
        
        # Execute the insertion query
        self.cur.execute(insert_sql, (venue, author_openreview_id, author_full_name, email, affiliation, homepage, dblp))
        
        # Get the inserted author id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
    
    def delete_author(self, author_openreview_id: str):
        # search the row based on primary key
        select_sql = """
        SELECT * FROM authors WHERE author_openreview_id = %s;
        """
        self.cur.execute(select_sql, (author_openreview_id,))
        row = self.cur.fetchone()

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
        
    def update_author(self, venue: str, author_openreview_id: str, author_full_name: str, email: str, 
                      affiliation: str, homepage: str, dblp: str):
        # Query to select the current record using primary key
        select_sql = """
        SELECT * FROM authors WHERE author_openreview_id = %s;
        """
        self.cur.execute(select_sql, (author_openreview_id,))
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
    
    def get_author(self, author_openreview_id: str):
        # Query to select the current record using primary key
        select_sql = """
        SELECT * FROM authors WHERE author_openreview_id = %s;
        """
        self.cur.execute(select_sql, (author_openreview_id,))
        row = self.cur.fetchone()

        if not row:
            print(f"No author found with author_openreview_id {author_openreview_id}.")
            return None
        else:
            columns = ['venue', 'author_openreview_id', 'author_full_name', 
                       'email', 'affiliation', 'homepage', 'dblp']
            # original_record = dict(zip(columns, row))
            author_df = pd.DataFrame(row, columns=columns)
            return author_df
    
    def get_all_authors(self, is_all_features: bool = False):
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
    
    def check_author_exists(self, author_openreview_id: str) -> bool:
        self.cur.execute("SELECT 1 FROM authors WHERE author_openreview_id = %s LIMIT 1;", (author_openreview_id,))
        result = self.cur.fetchone()

        return result is not None
    
    # papers <-> authors
    def create_papers_authors_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS papers_authors (
            venue TEXT,
            paper_openreview_id VARCHAR(255),
            author_openreview_id VARCHAR(255),
            PRIMARY KEY (paper_openreview_id, author_openreview_id)
        );
        """
        self.cur.execute(create_table_sql)
        print("Table 'papers_authors' created successfully.")
        
    def insert_paper_authors(self, venue: str, paper_openreview_id: str, author_openreview_id: str):
        if self.check_paper_exists(paper_openreview_id) and self.check_author_exists(author_openreview_id):
            insert_sql = """
            INSERT INTO papers_authors (venue, paper_openreview_id, author_openreview_id)
            VALUES (%s, %s, %s)
            ON CONFLICT (paper_openreview_id, author_openreview_id) DO NOTHING
            RETURNING (paper_openreview_id, author_openreview_id);
            """
            
            # Execute the insertion query
            self.cur.execute(insert_sql, (venue, paper_openreview_id, author_openreview_id))
            
            # Get the inserted paper's openreview id (if any)
            res = self.cur.fetchone()
            return res[0] if res else None
        else:
            print(f'''
                The paper {paper_openreview_id} or the author {author_openreview_id} 
                is(are) not exists in this database. Please add them before connect them.
                ''')
            return None
    
    def delete_paper_author(self, paper_openreview_id: str, author_openreview_id: str):
        original_record = self.get_paper_author(paper_openreview_id, author_openreview_id)
        
        if original_record is not None:
            self.cur.execute("""
            DELETE FROM papers_authors
            WHERE paper_openreview_id = %s AND author_openreview_id = %s;
            """, (paper_openreview_id, author_openreview_id))
            
            self.conn.commit()
            print(f"The connection between paper {paper_openreview_id} and author {author_openreview_id} is deleted successfully.")
            return original_record
        else:
            return None
        
    def get_paper_author(self, paper_openreview_id: str, author_openreview_id: str):
        self.cur.execute("""
        SELECT venue, paper_openreview_id, author_openreview_id FROM papers_authors
        WHERE paper_openreview_id = %s AND author_openreview_id = %s;
        """, (paper_openreview_id, author_openreview_id))
        
        result = self.cur.fetchone()
        
        if result is None:
            print(f'''
                The paper {paper_openreview_id} and the author {author_openreview_id} are not connect in this database.
                ''')
            return None
        else:
            columns = ['venue', 'paper_openreview_id', 'author_openreview_id']
            # result = dict(zip(columns, result))
            result_df = pd.DataFrame(result, columns=columns)
        return result_df
    
    def get_all_papers_authors(self):
        select_query = """
        SELECT venue, paper_openreview_id, author_openreview_id
        FROM papers_authors ORDER BY paper_openreview_id ASC
        """
        self.cur.execute(select_query)
            
        # Fetch all the results
        papers_authors = self.cur.fetchall()
        
        # Return the result as a list of tuples (venue, author_openreview_id, author_full_name)
        authors_df = pd.DataFrame(papers_authors, columns=["venue", "paper_openreview_id", "author_openreview_id"])
        return authors_df
    
    def check_paper_author_exists(self, paper_openreview_id: str, author_openreview_id: str) -> bool:
        self.cur.execute("""
        SELECT 1 FROM papers_authors
        WHERE paper_openreview_id = %s AND author_openreview_id = %s 
        LIMIT 1;
        """, (paper_openreview_id, author_openreview_id))
        
        result = self.cur.fetchone()

        return result is not None
    
    def get_paper_neighboring_authors(self, paper_openreview_id: str):
        self.cur.execute("""
        SELECT venue, paper_openreview_id, author_openreview_id FROM papers_authors
        WHERE paper_openreview_id = %s;
        """, (paper_openreview_id,))
        
        paper_neighboring_authors = self.cur.fetchall()
        
        if paper_neighboring_authors is not None:
            paper_neighboring_authors_df = pd.DataFrame(paper_neighboring_authors, columns=["venue", "paper_openreview_id", "author_openreview_id"])
            return paper_neighboring_authors_df
        else:
            return None
        
    def get_author_neighboring_papers(self, author_openreview_id: str):
        self.cur.execute("""
        SELECT venue, paper_openreview_id, author_openreview_id FROM papers_authors
        WHERE author_openreview_id = %s;
        """, (author_openreview_id,))
        
        author_neighboring_papers = self.cur.fetchall()
        
        if author_neighboring_papers is not None:
            author_neighboring_papers_df = pd.DataFrame(author_neighboring_papers, columns=["venue", "paper_openreview_id", "author_openreview_id"])
            return author_neighboring_papers_df
        else:
            return None
    
    # papers <-> revisions
    def create_papers_revisions_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS papers_revisions (
            venue TEXT,
            paper_openreview_id VARCHAR(255),
            revision_openreview_id VARCHAR(255),
            title TEXT,
            time TEXT,
            PRIMARY KEY (paper_openreview_id, revision_openreview_id)
        );
        """
        self.cur.execute(create_table_sql)
        print("Table 'papers_revisions' created successfully.")
        
    def insert_paper_revisions(self, venue: str, paper_openreview_id: str, revision_openreview_id: str, title: str, time: str):
        if self.check_paper_exists(paper_openreview_id) and self.check_revision_exists(revision_openreview_id):
            insert_sql = """
            INSERT INTO papers_revisions (venue, paper_openreview_id, revision_openreview_id, title, time)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (paper_openreview_id, revision_openreview_id) DO NOTHING
            RETURNING (paper_openreview_id, revision_openreview_id);
            """
            
            # Execute the insertion query
            self.cur.execute(insert_sql, (venue, paper_openreview_id, revision_openreview_id, title, time))
            
            # Get the inserted paper's openreview id (if any)
            res = self.cur.fetchone()
            return res[0] if res else None
        else:
            print(f'''
                The paper {paper_openreview_id} or the revision {revision_openreview_id} 
                is(are) not exists in this database. Please add them before connect them.
                ''')
            return None
    
    def delete_paper_revision(self, paper_openreview_id: str, revision_openreview_id: str):
        original_record = self.get_paper_revision(paper_openreview_id, revision_openreview_id)
        
        if original_record is not None:
            self.cur.execute("""
            DELETE FROM papers_revisions
            WHERE paper_openreview_id = %s AND revision_openreview_id = %s;
            """, (paper_openreview_id, revision_openreview_id))
            
            self.conn.commit()
            print(f"The connection between paper {paper_openreview_id} and revision {revision_openreview_id} is deleted successfully.")
            return original_record
        else:
            return None
    
    def get_paper_revision(self, paper_openreview_id: str, revision_openreview_id: str):
        self.cur.execute("""
        SELECT venue, paper_openreview_id, revision_openreview_id, title, time FROM papers_revisions
        WHERE paper_openreview_id = %s AND revision_openreview_id = %s;
        """, (paper_openreview_id, revision_openreview_id))
        
        result = self.cur.fetchone()
        
        if result is None:
            print(f'''
                The paper {paper_openreview_id} and the revision {revision_openreview_id} are not connect in this database.
                ''')
            return None
        else:
            columns = ['venue', 'paper_openreview_id', 
                       'revision_openreview_id', "title", "time"]
            # result = dict(zip(columns, result))
            result_df = pd.DataFrame(result, columns=columns)
        return result_df
    
    def get_all_papers_revisions(self):
        select_query = """
        SELECT venue, paper_openreview_id, revision_openreview_id, title, time
        FROM papers_revisions ORDER BY paper_openreview_id ASC
        """
        self.cur.execute(select_query)
            
        # Fetch all the results
        papers_authors = self.cur.fetchall()
        
        # Return the result as a list of tuples (venue, author_openreview_id, author_full_name)
        authors_df = pd.DataFrame(papers_authors, columns=["venue", "paper_openreview_id", "revision_openreview_id", "title", "time"])
        return authors_df
    
    def check_paper_revision_exists(self, paper_openreview_id: str, revision_openreview_id: str) -> bool:
        self.cur.execute("""
        SELECT 1 FROM papers_revisions
        WHERE paper_openreview_id = %s AND revision_openreview_id = %s 
        LIMIT 1;
        """, (paper_openreview_id, revision_openreview_id))
        
        result = self.cur.fetchone()

        return result is not None
    
    def get_paper_neighboring_revisions(self, paper_openreview_id: str):
        self.cur.execute("""
        SELECT venue, paper_openreview_id, revision_openreview_id, title, time FROM papers_revisions
        WHERE paper_openreview_id = %s;
        """, (paper_openreview_id,))
        
        paper_neighboring_revisions = self.cur.fetchall()
        
        if paper_neighboring_revisions is not None:
            paper_neighboring_revisions_df = pd.DataFrame(paper_neighboring_revisions, columns=["venue", "paper_openreview_id", "revision_openreview_id", "title", "time"])
            return paper_neighboring_revisions_df
        else:
            return None
        
    def get_revision_neighboring_papers(self, revision_openreview_id: str):
        self.cur.execute("""
        SELECT venue, paper_openreview_id, revision_openreview_id, title, time FROM papers_revisions
        WHERE revision_openreview_id = %s;
        """, (revision_openreview_id,))
        
        revision_neighboring_papers = self.cur.fetchall()
        
        if revision_neighboring_papers is not None:
            revision_neighboring_papers_df = pd.DataFrame(revision_neighboring_papers, columns=["venue", "paper_openreview_id", "revision_openreview_id", "title", "time"])
            return revision_neighboring_papers_df
        else:
            return None
        
    # papers <-> reviews
    def create_papers_reviews_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS papers_reviews (
            venue TEXT,
            paper_openreview_id VARCHAR(255),
            review_openreview_id VARCHAR(255),
            title TEXT,
            time TEXT,
            PRIMARY KEY (paper_openreview_id, review_openreview_id)
        );
        """
        self.cur.execute(create_table_sql)
        print("Table 'papers_reviews' created successfully.")
        
    def insert_paper_reviews(self, venue: str, paper_openreview_id: str, review_openreview_id: str, title: str, time: str):
        if self.check_paper_exists(paper_openreview_id) and self.check_review_exists(review_openreview_id):
            insert_sql = """
            INSERT INTO papers_reviews (venue, paper_openreview_id, review_openreview_id, title, time)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (paper_openreview_id, review_openreview_id) DO NOTHING
            RETURNING (paper_openreview_id, review_openreview_id);
            """
            
            # Execute the insertion query
            self.cur.execute(insert_sql, (venue, paper_openreview_id, review_openreview_id, title, time))
            
            # Get the inserted paper's openreview id (if any)
            res = self.cur.fetchone()
            return res[0] if res else None
        else:
            print(f'''
                The paper {paper_openreview_id} or the revision {review_openreview_id} 
                is(are) not exists in this database. Please add them before connect them.
                ''')
            return None
    
    def delete_paper_review(self, paper_openreview_id: str, review_openreview_id: str):
        original_record = self.get_paper_review(paper_openreview_id, review_openreview_id)
        
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
        
    def get_paper_review(self, paper_openreview_id: str, review_openreview_id: str):
        self.cur.execute("""
        SELECT venue, paper_openreview_id, review_openreview_id, title, time FROM papers_reviews
        WHERE paper_openreview_id = %s AND review_openreview_id = %s;
        """, (paper_openreview_id, review_openreview_id))
        
        result = self.cur.fetchone()
        
        if result is None:
            print(f'''
                The paper {paper_openreview_id} and the review {review_openreview_id} are not connect in this database.
                ''')
            return None
        else:
            columns = ['venue', 'paper_openreview_id', 
                       'review_openreview_id', "title", "time"]
            # result = dict(zip(columns, result))
            result_df = pd.DataFrame(result, columns=columns)
        return result_df
    
    def get_all_papers_reviews(self):
        select_query = """
        SELECT venue, paper_openreview_id, review_openreview_id, title, time
        FROM papers_reviews ORDER BY paper_openreview_id ASC
        """
        self.cur.execute(select_query)
            
        # Fetch all the results
        papers_authors = self.cur.fetchall()
        
        # Return the result as a list of tuples (venue, author_openreview_id, author_full_name)
        authors_df = pd.DataFrame(papers_authors, columns=["venue", "paper_openreview_id", "review_openreview_id", "title", "time"])
        return authors_df
    
    def check_paper_review_exists(self, paper_openreview_id: str, review_openreview_id: str) -> bool:
        self.cur.execute("""
        SELECT 1 FROM papers_reviews
        WHERE paper_openreview_id = %s AND review_openreview_id = %s 
        LIMIT 1;
        """, (paper_openreview_id, review_openreview_id))
        
        result = self.cur.fetchone()

        return result is not None
    
    def get_paper_neighboring_reviews(self, paper_openreview_id: str):
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
        
    def get_review_neighboring_papers(self, review_openreview_id: str):
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
        
    # openreview -> arxiv
    def create_openreview_arxiv_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS openreview_arxiv (
            openreview_id VARCHAR(255) PRIMARY KEY,
            arxiv_id VARCHAR(255),
            title TEXT,
            author_names TEXT
        );
        """
        # Execute the SQL to create the table
        self.cur.execute(create_table_sql)
        print("Table 'openreview_arxiv' created successfully.")
        
    def insert_openreview_arxiv(self, openreview_id, arxiv_id, title, author_names):
        """
        Insert a connection between openreview id with arxiv id into the openreview_arxiv table. Returns the inserted openreview id or None if it fails.
        - openreview_id: str, unique identifier for the paper in openreview system(primary key).
        - arxiv_id: str, unique identifier for the paper in arxiv system.
        - title: str, the title of the paper.
        """
        insert_sql = """
        INSERT INTO openreview_arxiv (openreview_id, arxiv_id, title, author_names)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (openreview_id) DO NOTHING
        RETURNING openreview_id;
        """
        
        # Execute the insertion query
        self.cur.execute(insert_sql, (openreview_id, arxiv_id, title, author_names))
        
        # Get the inserted paper's openreview id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
    
    # revisions -> papers -> reviews
    def create_papers_revisions_reviews_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS papers_revisions_reviews (
            venue TEXT,
            paper_openreview_id VARCHAR(255),
            revision_openreview_id VARCHAR(255),
            review_openreview_id VARCHAR(255),
            revision_time TEXT,
            review_time TEXT,
            PRIMARY KEY (paper_openreview_id, revision_openreview_id, review_openreview_id)
        );
        """
        # Execute the SQL to create the table
        self.cur.execute(create_table_sql)
        print("Table 'papers_revisions_reviews' created successfully.")
        
    def insert_paper_revision_review(self, venue, paper_openreview_id, revision_openreview_id, review_openreview_id, revision_time, review_time):
        """
        Insert a revision into the revisions table. Returns the inserted revision id or None if it fails.
        - venue: str, the venue where the paper is submitted.
        - paper_openreview_id: str, unique identifier for the paper.
        - original_openreview_id: str, unique identifier for the revision's original paper.
        - revision_openreview_id: str, unique identifier for the revision's original paper (primary key).
        - reviews: json file.
        - time: text
        """
        insert_sql = """
        INSERT INTO papers_revisions_reviews (venue, paper_openreview_id, revision_openreview_id, review_openreview_id, revision_time, review_time)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (paper_openreview_id, revision_openreview_id, review_openreview_id) DO NOTHING
        RETURNING (paper_openreview_id, revision_openreview_id, review_openreview_id);
        """

        # Execute the insertion query
        self.cur.execute(insert_sql, (venue, paper_openreview_id, revision_openreview_id, review_openreview_id, revision_time, review_time))
        
        # Get the inserted paper's openreview id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
    
    def _clean_json_content(self, content):
        """
        Recursively remove invalid characters from the JSON content (e.g., \u0000).
        """
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