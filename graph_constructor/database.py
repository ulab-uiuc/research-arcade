import psycopg2
from psycopg2.extras import Json

# Store the pwd of db server in the env or here as a global variable
# PASSWORD = 
PASSWORD = "Lcs20031121!"

class Database:

    def __init__(self):
        # Store connection and cursor for reuse
        self.conn = psycopg2.connect(
            host="localhost", dbname="postgres",
            user="postgres", password=PASSWORD, port="5432"
        )
        # Enable autocommit
        self.conn.autocommit = True
        self.cur = self.conn.cursor()

    def create_papers_table(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id SERIAL PRIMARY KEY,
            arxiv_id VARCHAR(100) UNIQUE,
            title TEXT NOT NULL,
            abstract TEXT,
            submit_date DATE,
            metadata JSONB
        )
        """)

    def create_authors_table(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS authors (
            id SERIAL PRIMARY KEY,
            semantic_scholar_id VARCHAR(100) UNIQUE,
            name VARCHAR(255) NOT NULL,
            homepage VARCHAR(255)
        )
        """)

    def create_categories_table(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL,
            description TEXT
        )
        """)

    def create_institutions_table(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS institutions (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            location VARCHAR(255)
        )
        """)

    def create_figures_table(self):
        # Change paper_id to paper_arxiv_id VARCHAR referencing papers(arxiv_id)
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS figures (
            id SERIAL PRIMARY KEY,
            paper_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
            path VARCHAR(500),
            caption TEXT,
            label TEXT,
            name TEXT
        )
        """)

    def create_tables_table(self):
        # Note: 'tables' here is literal; change paper_id to paper_arxiv_id referencing papers(arxiv_id)
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS tables (
            id SERIAL PRIMARY KEY,
            paper_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
            path VARCHAR(500),
            caption TEXT,
            label TEXT,
            table_text TEXT
        )
        """)

    def create_paper_authors_table(self):
        # Link by paper_arxiv_id (string) and author_id (int)
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_authors (
            paper_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
            author_id VARCHAR(100) NOT NULL REFERENCES authors(semantic_scholar_id) ON DELETE CASCADE,
            author_sequence INT NOT NULL,
            PRIMARY KEY (paper_arxiv_id, author_id)
        )
        """)

    def create_paper_category_table(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_category (
            paper_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
            category_id INT NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
            PRIMARY KEY (paper_arxiv_id, category_id)
        )
        """)

    def create_citations_table(self):
        # citation edges by arxiv_id strings
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS citations (
            citing_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
            cited_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
            PRIMARY KEY (citing_arxiv_id, cited_arxiv_id),
            CHECK (citing_arxiv_id <> cited_arxiv_id)
        )
        """)

    def create_paper_figures_table(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_figures (
            paper_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
            figure_id INT NOT NULL REFERENCES figures(id) ON DELETE CASCADE,
            PRIMARY KEY (paper_arxiv_id, figure_id)
        )
        """)

    def create_paper_tables_table(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_tables (
            paper_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
            table_id INT NOT NULL REFERENCES tables(id) ON DELETE CASCADE,
            PRIMARY KEY (paper_arxiv_id, table_id)
        )
        """)

    def create_author_affiliation_table(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS author_affiliation (
            author_id INT NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
            institution_id INT NOT NULL REFERENCES institutions(id) ON DELETE CASCADE,
            PRIMARY KEY (author_id, institution_id)
        )
        """)

    def create_all(self):
        # Create tables in order respecting dependencies
        self.create_papers_table()
        self.create_authors_table()
        self.create_categories_table()
        self.create_institutions_table()
        self.create_figures_table()
        self.create_tables_table()
        self.create_paper_authors_table()
        self.create_paper_category_table()
        self.create_citations_table()
        self.create_paper_figures_table()
        self.create_paper_tables_table()
        self.create_author_affiliation_table()

    # Insert methods for papers/authors/etc remain the same for papers:
    def insert_paper(self, arxiv_id, title, abstract=None, submit_date=None, metadata=None):
        """
        Insert a paper. Returns the generated paper id.
        - arxiv_id: str (unique) or None
        - title: str
        - abstract: str or None
        - submit_date: datetime.date or ISO-format str or None
        - metadata: dict or None
        """
        sql = """
        INSERT INTO papers (arxiv_id, title, abstract, submit_date, metadata)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (arxiv_id) DO NOTHING
        RETURNING id
        """
        meta_val = Json(metadata) if metadata is not None else None
        self.cur.execute(sql, (arxiv_id, title, abstract, submit_date, meta_val))
        res = self.cur.fetchone()
        return res[0] if res else None

    def insert_author(self, semantic_scholar_id, name, homepage=None):
        sql = """
        INSERT INTO authors (semantic_scholar_id, name, homepage)
        VALUES (%s, %s, %s)
        ON CONFLICT (semantic_scholar_id) DO NOTHING
        RETURNING id
        """
        self.cur.execute(sql, (semantic_scholar_id, name, homepage))
        res = self.cur.fetchone()
        return res[0] if res else None

    def insert_category(self, name, description=None):
        sql = """
        INSERT INTO categories (name, description)
        VALUES (%s, %s)
        ON CONFLICT (name) DO NOTHING
        RETURNING id
        """
        self.cur.execute(sql, (name, description))
        res = self.cur.fetchone()
        if res:
            return res[0]
        # conflict: fetch existing
        self.cur.execute("SELECT id FROM categories WHERE name = %s", (name,))
        return self.cur.fetchone()[0]

    def insert_institution(self, name, location=None):
        sql = """
        INSERT INTO institutions (name, location)
        VALUES (%s, %s)
        ON CONFLICT (name) DO NOTHING
        RETURNING id
        """
        self.cur.execute(sql, (name, location))
        res = self.cur.fetchone()
        return res[0] if res else None

    # Insert figure: now paper_arxiv_id instead of numeric paper_id
    def insert_figure(self, paper_arxiv_id, path, caption=None, label=None, name=None):
        """
        Insert a figure. Returns the generated figure id.
        - paper_arxiv_id: str (the arxiv_id of the paper)
        - path: str or None
        - caption: str or None
        - label: str or None
        - name: str or None
        """
        sql = """
        INSERT INTO figures (paper_arxiv_id, path, caption, label, name)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """
        # No ON CONFLICT here because we may allow multiple figures per paper.
        self.cur.execute(sql, (paper_arxiv_id, path, caption, label, name))
        res = self.cur.fetchone()
        return res[0] if res else None

    # Insert table record: now paper_arxiv_id
    def insert_table(self, paper_arxiv_id, path=None, caption=None, label=None, table_text=None):
        """
        Insert a table record. Returns the generated table id.
        - paper_arxiv_id: str
        - path: str or None
        - caption: str or None
        - label: str or None
        - table_text: str or None
        """
        sql = """
        INSERT INTO tables (paper_arxiv_id, path, caption, label, table_text)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """
        self.cur.execute(sql, (paper_arxiv_id, path, caption, label, table_text))
        res = self.cur.fetchone()
        return res[0] if res else None

    # Link methods now take paper_arxiv_id (string)
    def insert_paper_author(self, paper_arxiv_id, author_id, author_sequence):
        """
        Link a paper and an author via arxiv_id. Returns True if inserted, False if already exists.
        """
        sql = """
        INSERT INTO paper_authors (paper_arxiv_id, author_id, author_sequence)
        VALUES (%s, %s, %s)
        ON CONFLICT (paper_arxiv_id, author_id) DO NOTHING
        """
        self.cur.execute(sql, (paper_arxiv_id, author_id, author_sequence))
        return self.cur.rowcount == 1

    def insert_paper_category(self, paper_arxiv_id, category_id):
        """
        Link a paper and a category via arxiv_id. Returns True if inserted, False if exists.
        """
        sql = """
        INSERT INTO paper_category (paper_arxiv_id, category_id)
        VALUES (%s, %s)
        ON CONFLICT (paper_arxiv_id, category_id) DO NOTHING
        """
        self.cur.execute(sql, (paper_arxiv_id, category_id))
        return self.cur.rowcount == 1

    def insert_citation(self, citing_arxiv_id, cited_arxiv_id):
        """
        Insert a citation edge by arxiv_id strings. Returns True if inserted, False if exists or invalid.
        """
        if citing_arxiv_id == cited_arxiv_id:
            return False
        sql = """
        INSERT INTO citations (citing_arxiv_id, cited_arxiv_id)
        VALUES (%s, %s)
        ON CONFLICT (citing_arxiv_id, cited_arxiv_id) DO NOTHING
        """
        self.cur.execute(sql, (citing_arxiv_id, cited_arxiv_id))
        return self.cur.rowcount == 1

    def insert_paper_figure(self, paper_arxiv_id, figure_id):
        """
        Link paper to figure explicitly by arxiv_id. Returns True if inserted, False if exists.
        """
        sql = """
        INSERT INTO paper_figures (paper_arxiv_id, figure_id)
        VALUES (%s, %s)
        ON CONFLICT (paper_arxiv_id, figure_id) DO NOTHING
        """
        self.cur.execute(sql, (paper_arxiv_id, figure_id))
        return self.cur.rowcount == 1

    def insert_paper_table(self, paper_arxiv_id, table_id):
        """
        Link paper to table explicitly by arxiv_id. Returns True if inserted, False if exists.
        """
        sql = """
        INSERT INTO paper_tables (paper_arxiv_id, table_id)
        VALUES (%s, %s)
        ON CONFLICT (paper_arxiv_id, table_id) DO NOTHING
        """
        self.cur.execute(sql, (paper_arxiv_id, table_id))
        return self.cur.rowcount == 1

    def insert_author_affiliation(self, author_id, institution_id):
        """
        Link an author to an institution. Returns True if inserted, False if exists.
        """
        sql = """
        INSERT INTO author_affiliation (author_id, institution_id)
        VALUES (%s, %s)
        ON CONFLICT (author_id, institution_id) DO NOTHING
        """
        self.cur.execute(sql, (author_id, institution_id))
        return self.cur.rowcount == 1

    def close(self):
        self.cur.close()
        self.conn.close()

    def drop_all(self):
        """
        Drop all tables in reverse dependency order.
        """
        # List tables in order so that dependent tables are dropped before the ones they reference
        tables = [
            "paper_figures",
            "paper_tables",
            "paper_authors",
            "paper_category",
            "citations",
            "author_affiliation",
            "figures",
            "tables",
            "categories",
            "authors",
            "institutions",
            "papers"
        ]
        for tbl in tables:
            # Use IF EXISTS to avoid errors if a table is already gone
            # Use CASCADE for safety in case there are lingering dependencies
            self.cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
