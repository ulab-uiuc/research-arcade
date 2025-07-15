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
            revisions JSONB,
            all_diffs JSONB
        );
        """
        self.cur.execute(create_table_sql)
        print("Table 'papers' created successfully.")
        
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
    
    def clean_json_content(self, content):
        """
        Recursively remove invalid characters from the JSON content (e.g., \u0000).
        """
        if isinstance(content, str):
            # Remove null character (\u0000) and any other non-printable characters
            return ''.join(char for char in content if char.isprintable())
        elif isinstance(content, dict):
            # Clean all string values in the dictionary
            return {key: self.clean_json_content(value) for key, value in content.items()}
        elif isinstance(content, list):
            # Clean all elements in the list
            return [self.clean_json_content(item) for item in content]
        else:
            return content
        
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
        cleaned_content = self.clean_json_content(content)
        
        # Execute the insertion query
        self.cur.execute(insert_sql, (venue, paper_openreview_id, review_openreview_id, replyto_openreview_id, writer, title, Json(cleaned_content), time))
        
        # Get the inserted review id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None
    
    def insert_paper(self, venue, paper_openreview_id, title, abstract, author_openreview_ids, author_full_names, paper_decision, paper_pdf_link, revisions, all_diffs):
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
        INSERT INTO papers (venue, paper_openreview_id, title, abstract, author_openreview_ids, author_full_names, paper_decision, paper_pdf_link, revisions, all_diffs)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (paper_openreview_id) DO NOTHING
        RETURNING paper_openreview_id;
        """
        # clean revisions
        cleaned_revisions = self.clean_json_content(revisions)
        cleaned_all_diffs = self.clean_json_content(all_diffs)
        
        # Execute the insertion query
        self.cur.execute(insert_sql, (venue, paper_openreview_id, title, abstract, author_openreview_ids, author_full_names, paper_decision, paper_pdf_link, Json(cleaned_revisions), Json(cleaned_all_diffs)))
        
        # Get the inserted paper id (if any)
        res = self.cur.fetchone()
        return res[0] if res else None