import psycopg2
from psycopg2.extras import Json

class Database:

    def __init__(self):
        # Store connection and cursor for reuse
        self.conn = psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="Lcs20031121!", port="5432")
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
            homepage VARCHAR(255) UNIQUE
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
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS figures (
            id SERIAL PRIMARY KEY,
            paper_id INT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            figure_index INT NOT NULL,
            path VARCHAR(500),
            caption TEXT,
            label TEXT,
            UNIQUE(paper_id, figure_index)
        )
        """)

    def create_tables_table(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS tables (
            id SERIAL PRIMARY KEY,
            paper_id INT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            table_index INT NOT NULL,
            path VARCHAR(500),
            caption TEXT,
            UNIQUE(paper_id, table_index)
        )
        """)

    def create_paper_authors_table(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_authors (
            paper_id INT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            author_id INT NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
            author_sequence INT NOT NULL,
            PRIMARY KEY (paper_id, author_id)
        )
        """)

    def create_paper_category_table(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_category (
            paper_id INT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            category_id INT NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
            PRIMARY KEY (paper_id, category_id)
        )
        """)

    def create_citations_table(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS citations (
            citing_paper_id INT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            cited_paper_id INT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            PRIMARY KEY (citing_paper_id, cited_paper_id),
            CHECK (citing_paper_id <> cited_paper_id)
        )
        """)

    def create_paper_figures_table(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_figures (
            paper_id INT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            figure_id INT NOT NULL REFERENCES figures(id) ON DELETE CASCADE,
            PRIMARY KEY (paper_id, figure_id)
        )
        """)

    def create_paper_tables_table(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_tables (
            paper_id INT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            table_id INT NOT NULL REFERENCES tables(id) ON DELETE CASCADE,
            PRIMARY KEY (paper_id, table_id)
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
        # Create tables in dependency order
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


    def insert_paper(self, arxiv_id, title, abstract=None, submit_date=None, metadata=None):
        """
        Insert a paper. Returns the generated paper id.
        - arxiv_id: str or None
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
        # Wrap metadata in Json if provided
        meta_val = Json(metadata) if metadata is not None else None
        self.cur.execute(sql, (arxiv_id, title, abstract, submit_date, meta_val))
        res = self.cur.fetchone()
        return res[0] if res else None

    def insert_author(self, semantic_scholar_id, name, homepage=None):
        """
        Insert an author. Returns the generated author id.
        - semantic_scholar_id: str or None
        - name: str
        - orcid: str or None
        """
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
        """
        Insert a category. Returns the generated category id.
        - name: str
        - description: str or None
        """
        sql = """
        INSERT INTO categories (name, description)
        VALUES (%s, %s)
        ON CONFLICT (name) DO NOTHING
        RETURNING id
        """
        self.cur.execute(sql, (name, description))
        res = self.cur.fetchone()
        return res[0] if res else None

    def insert_institution(self, name, location=None):
        """
        Insert an institution. Returns the generated institution id.
        - name: str
        - location: str or None
        """
        sql = """
        INSERT INTO institutions (name, location)
        VALUES (%s, %s)
        ON CONFLICT (name) DO NOTHING
        RETURNING id
        """
        self.cur.execute(sql, (name, location))
        res = self.cur.fetchone()
        return res[0] if res else None

    def insert_figure(self, paper_id, figure_index, path, caption=None, label=None):
        """
        Insert a figure. Returns the generated figure id.
        - paper_id: int
        - figure_index: int
        - path: str or None
        - caption: str or None
        """
        sql = """
        INSERT INTO figures (paper_id, figure_index, path, caption, label)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (paper_id, figure_index) DO NOTHING
        RETURNING id
        """
        self.cur.execute(sql, (paper_id, figure_index, path, caption))
        res = self.cur.fetchone()
        return res[0] if res else None

    def insert_table(self, paper_id, table_index, path, caption=None):
        """
        Insert a table record. Returns the generated table id.
        - paper_id: int
        - table_index: int
        - path: str or None
        - caption: str or None
        """
        sql = """
        INSERT INTO tables (paper_id, table_index, path, caption)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (paper_id, table_index) DO NOTHING
        RETURNING id
        """
        self.cur.execute(sql, (paper_id, table_index, path, caption))
        res = self.cur.fetchone()
        return res[0] if res else None

    # Insert into edge tables. Return True if inserted, False if conflict/no-op:
    def insert_paper_author(self, paper_id, author_id, author_sequence):
        """
        Link a paper and an author. Returns True if inserted, False if already exists.
        """
        sql = """
        INSERT INTO paper_authors (paper_id, author_id, author_sequence)
        VALUES (%s, %s, %s)
        ON CONFLICT (paper_id, author_id) DO NOTHING
        """
        self.cur.execute(sql, (paper_id, author_id, author_sequence))
        # Check rowcount: 1 if inserted, 0 if skipped
        return self.cur.rowcount == 1

    def insert_paper_category(self, paper_id, category_id):
        """
        Link a paper and a category. Returns True if inserted, False if already exists.
        """
        sql = """
        INSERT INTO paper_category (paper_id, category_id)
        VALUES (%s, %s)
        ON CONFLICT (paper_id, category_id) DO NOTHING
        """
        self.cur.execute(sql, (paper_id, category_id))
        return self.cur.rowcount == 1

    def insert_citation(self, citing_paper_id, cited_paper_id):
        """
        Insert a citation edge. Returns True if inserted, False if exists or invalid.
        """
        if citing_paper_id == cited_paper_id:
            return False
        sql = """
        INSERT INTO citations (citing_paper_id, cited_paper_id)
        VALUES (%s, %s)
        ON CONFLICT (citing_paper_id, cited_paper_id) DO NOTHING
        """
        self.cur.execute(sql, (citing_paper_id, cited_paper_id))
        return self.cur.rowcount == 1

    def insert_paper_figure(self, paper_id, figure_id):
        """
        Link paper to figure explicitly. Returns True if inserted, False if exists.
        """
        sql = """
        INSERT INTO paper_figures (paper_id, figure_id)
        VALUES (%s, %s)
        ON CONFLICT (paper_id, figure_id) DO NOTHING
        """
        self.cur.execute(sql, (paper_id, figure_id))
        return self.cur.rowcount == 1

    def insert_paper_table(self, paper_id, table_id):
        """
        Link paper to table explicitly. Returns True if inserted, False if exists.
        """
        sql = """
        INSERT INTO paper_tables (paper_id, table_id)
        VALUES (%s, %s)
        ON CONFLICT (paper_id, table_id) DO NOTHING
        """
        self.cur.execute(sql, (paper_id, table_id))
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


if __name__ == "__main__":
    db = Database()
    db.create_all()
    print("All tables created.")
    db.close()
