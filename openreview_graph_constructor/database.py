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

    def create_papers_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS papers (
            venue TEXT,
            paper_openreview_id VARCHAR(255) PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            author_openreview_ids TEXT,
            author_full_names TEXT,
            paper_decision TEXT,
            paper_pdf_link TEXT,
            revisions JSONB
        );
        """
        self.cur.execute(create_table_sql)
        print("Table 'papers' created successfully.")
        
    def insert_paper(self, venue, paper_openreview_id, title, abstract, author_openreview_ids, author_full_names, paper_decision, paper_pdf_link, revisions):
        """
        Insert a paper into the papers table. Returns the inserted paper id or None if it fails.
        - venue: str, the venue where the paper is submitted.
        - paper_openreview_id: str, unique identifier for the paper (primary key).
        - title: str, the title of the paper.
        - abstract: str or None, the abstract of the paper (optional).
        - author_openreview_ids: str or None, a list of author Openreview IDs (optional).
        - author_full_names: str or None, a list of author full names (optional).
        - paper_decision: str or None, the decision for the paper (optional).
        - paper_pdf_link: str or None, a link to the paper's PDF (optional).
        """
        insert_sql = """
        INSERT INTO papers (venue, paper_openreview_id, title, abstract, author_openreview_ids, author_full_names, paper_decision, paper_pdf_link, revisions)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (paper_openreview_id) DO NOTHING
        RETURNING paper_openreview_id;
        """
        # clean revisions
        cleaned_revisions = self._clean_json_content(revisions)
        
        # Execute the insertion query
        self.cur.execute(insert_sql, (venue, paper_openreview_id, title, abstract, author_openreview_ids, author_full_names, paper_decision, paper_pdf_link, Json(cleaned_revisions)))
        
        # Get the inserted paper id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
        
    def get_all_papers(self):
        # Select query to get paper_openreview_id, title, and author_full_names
        select_query = """
        SELECT venue, paper_openreview_id, title, author_full_names
        FROM papers;
        """
        self.cur.execute(select_query)
        
        # Fetch all the results
        papers = self.cur.fetchall()
        
        # Return the result as a list of tuples (paper_openreview_id, title, author_full_names)
        return papers
    
    def create_review_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS reviews (
            id SERIAL UNIQUE,
            venue TEXT,
            paper_openreview_id VARCHAR(255),
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
        
    def insert_review(self, venue, paper_openreview_id, review_openreview_id, replyto_openreview_id, writer, title, content, time):
        """
        Insert a review into the reviews table. Returns the inserted review id or None if it fails.
        - venue: str, the venue where the paper was submitted.
        - paper_openreview_id: str, the paper's Openreview ID.
        - review_openreview_id: str, unique ID for the review (primary key).
        - replyto_openreview_id: str or None, ID for a reply (optional).
        - writer: str, the name or identity of the reviewer.
        - title: str, the title of the review.
        - content: JSON object, the content of the review.
        """
        insert_sql = """
        INSERT INTO reviews (venue, paper_openreview_id, review_openreview_id, replyto_openreview_id, writer, title, content, time)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (review_openreview_id) DO NOTHING
        RETURNING review_openreview_id;
        """
        
        # clean content
        cleaned_content = self._clean_json_content(content)
        
        # Execute the insertion query
        self.cur.execute(insert_sql, (venue, paper_openreview_id, review_openreview_id, replyto_openreview_id, writer, title, Json(cleaned_content), time))
        
        # Get the inserted review id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
    
    def get_all_reviews(self):
        select_query = """
        SELECT id, venue, paper_openreview_id, review_openreview_id, replyto_openreview_id, time 
        FROM reviews ORDER BY id ASC
        """
        self.cur.execute(select_query)
        
        # Fetch all the results
        reviews = self.cur.fetchall()
        
        # Return the result as a list of tuples (paper_openreview_id, title, author_full_names)
        return reviews
        
    def create_revisions_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS revisions (
            id SERIAL UNIQUE,
            venue TEXT,
            paper_openreview_id VARCHAR(255),
            original_openreivew_id VARCHAR(255),
            modified_openreview_id VARCHAR(255) PRIMARY KEY,
            content JSONB,
            time TEXT
        );
        """
        self.cur.execute(create_table_sql)
        print("Table 'revisions' created successfully.")
    
    def insert_revision(self, venue_id, paper_id, original_id, modified_id, content, time):
        """
        Insert a revision into the revisions table. Returns the inserted revision id or None if it fails.
        - venue: str, the venue where the paper is submitted.
        - paper_openreview_id: str, unique identifier for the paper.
        - original_openreivew_id: str, unique identifier for the revision's original paper.
        - modified_openreview_id: str, unique identifier for the revision's original paper (primary key).
        - content: json file.
        - time: text
        """
        insert_sql = """
        INSERT INTO revisions (venue, paper_openreview_id, original_openreivew_id, modified_openreview_id, content, time)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (modified_openreview_id) DO NOTHING
        RETURNING modified_openreview_id;
        """
        # clean revisions
        cleaned_revision_content = self._clean_json_content(content)

        # Execute the insertion query
        self.cur.execute(insert_sql, (venue_id, paper_id, original_id, modified_id, Json(cleaned_revision_content), time))
        
        # Get the inserted paper's openreview id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
    
    def get_all_revisions(self):
        # Select query to get paper_openreview_id, original_openreivew_id, modified_openreview_id, time
        select_query = """
        SELECT id, venue, paper_openreview_id, original_openreivew_id, modified_openreview_id, time
        FROM revisions ORDER BY id ASC
        """
        self.cur.execute(select_query)
        
        # Fetch all the results
        revisions = self.cur.fetchall()
        
        # Return the result as a list of tuples (paper_openreview_id, title, author_full_names)
        return revisions
        
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
        
    def insert_author(self, venue, author_openreview_id, author_full_name, email, affiliation, homepage, dblp):
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
        
    def create_revisions_reviews_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS revisions_reviews (
            id SERIAL UNIQUE,
            venue TEXT,
            paper_openreview_id VARCHAR(255),
            original_openreivew_id VARCHAR(255),
            modified_openreview_id VARCHAR(255) PRIMARY KEY,
            reviews JSONB,
            time TEXT
        );
        """
        # Execute the SQL to create the table
        self.cur.execute(create_table_sql)
        print("Table 'revisions_reviews' created successfully.")
        
    def insert_revision_reviews(self, venue_id, paper_id, original_id, modified_id, reviews, time):
        """
        Insert a revision into the revisions table. Returns the inserted revision id or None if it fails.
        - venue: str, the venue where the paper is submitted.
        - paper_openreview_id: str, unique identifier for the paper.
        - original_openreivew_id: str, unique identifier for the revision's original paper.
        - modified_openreview_id: str, unique identifier for the revision's original paper (primary key).
        - reviews: json file.
        - time: text
        """
        insert_sql = """
        INSERT INTO revisions_reviews (venue, paper_openreview_id, original_openreivew_id, modified_openreview_id, reviews, time)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (modified_openreview_id) DO NOTHING
        RETURNING modified_openreview_id;
        """
        # clean reviews
        cleaned_reviews = self._clean_json_content(reviews)

        # Execute the insertion query
        self.cur.execute(insert_sql, (venue_id, paper_id, original_id, modified_id, Json(cleaned_reviews), time))
        
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