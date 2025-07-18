import psycopg2
from psycopg2.extras import Json
import json

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
            base_arxiv_id VARCHAR(100),
            version VARCHAR(100),
            title TEXT NOT NULL,
            abstract TEXT,
            submit_date DATE,
            metadata JSONB
        )
        """)

    def create_sections_table(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS sections (
            id SERIAL PRIMARY KEY,
            content TEXT,
            title TEXT,
            appendix BOOLEAN,
            paper_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE
        )
        """)

    def create_paragraphs_table(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS paragraphs (
            id SERIAL PRIMARY KEY,
            paragraph_id INT NOT NULL,
            content TEXT,
            paper_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
            paper_section TEXT,
            UNIQUE (paragraph_id, paper_arxiv_id, paper_section)
        );
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
            id SERIAL PRIMARY KEY,
            citing_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
            cited_arxiv_id VARCHAR(100),
            bib_title TEXT,
            bib_key VARCHAR(255),
            author_cited_paper VARCHAR(255),
            citing_sections TEXT[] DEFAULT '{}',
            citing_paragraphs INT[] DEFAULT '{}',
            UNIQUE (citing_arxiv_id, cited_arxiv_id)
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
    
    def create_paragraph_citations_table(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS paragraph_citations (
            id SERIAL PRIMARY KEY,
            paragraph_id INT NOT NULL,
            paper_section TEXT,
            citing_arxiv_id VARCHAR(100) NOT NULL
                REFERENCES papers(arxiv_id)
                ON DELETE CASCADE,
            bib_key VARCHAR(255)
        );
        """)


    def create_paragraph_references_table(self):
        """
        Stores references that appear within paragraphs—
        e.g. figures, tables, equations, etc.
        """
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS paragraph_references (
            id SERIAL PRIMARY KEY,
            paragraph_id   INT    NOT NULL,
            paper_section TEXT,
            paper_arxiv_id VARCHAR(100) NOT NULL
                REFERENCES papers(arxiv_id)
                ON DELETE CASCADE,
            reference_label TEXT    NOT NULL,
            reference_type  TEXT
        );
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
        self.create_sections_table()
        self.create_paragraphs_table()
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
        self.create_paragraph_references_table()
        self.create_paragraph_citations_table()

    # def create_papers_table(self):
    #     self.cur.execute("""
    #     CREATE TABLE IF NOT EXISTS papers (
    #         id SERIAL PRIMARY KEY,
    #         arxiv_id VARCHAR(100) UNIQUE,
    #         base_arxiv_id VARCHAR(100),
    #         version VARCHAR(100),
    #         title TEXT NOT NULL,
    #         abstract TEXT,
    #         submit_date DATE,
    #         metadata JSONB
    #     )
    #     """)

    # Insert methods for papers/authors/etc remain the same for papers:
    def insert_paper(self, arxiv_id, base_arxiv_id, version, title, abstract=None, submit_date=None, metadata=None):
        """
        Insert a paper. Returns the generated paper id.
        - arxiv_id: str (unique) or None
        - title: str
        - abstract: str or None
        - submit_date: datetime.date or ISO-format str or None
        - metadata: dict or None
        """
        sql = """
        INSERT INTO papers (arxiv_id, base_arxiv_id, version, title, abstract, submit_date, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (arxiv_id) DO NOTHING
        RETURNING id
        """
        meta_val = Json(metadata) if metadata is not None else None
        self.cur.execute(sql, (arxiv_id, base_arxiv_id, version, title, abstract, submit_date, meta_val))
        res = self.cur.fetchone()
        return res[0] if res else None

    def insert_section(self, content, title, is_appendix, paper_arxiv_id):
        """
        Insert a section. Returns the generated section id
        - content: str
        - title: str
        - is_appendix: boolean
        - paper_arxiv_id: str
        """
        sql = """
        INSERT INTO sections (content, title, appendix, paper_arxiv_id)
        VALUES(%s, %s, %s, %s)
        RETURNING id
        """
        self.cur.execute(sql, (content, title, is_appendix, paper_arxiv_id))
        res = self.cur.fetchone()
        return res[0] if res else None
    
    # def create_paragraphs_table(self):
    #     self.cur.execute("""
    #     CREATE TABLE IF NOT EXISTS paragraphs (
    #         id SERIAL PRIMARY KEY,
    #         paragraph_id VARCHAR(100),
    #         content TEXT,
    #         paper_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    #         paper_section TEXT NOT NULL REFERENCES sections(title) ON DELETE CASCADE
    #     )
    #     """)

    # Here we don't have an extra paragraph-section or paragraph-paper table since the link information is already included here.
    def insert_paragraph(self, paragraph_id, content, paper_arxiv_id, paper_section):
        sql = """
        INSERT INTO paragraphs (paragraph_id, content, paper_arxiv_id, paper_section)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (paragraph_id, paper_arxiv_id, paper_section) DO NOTHING
        RETURNING id
        """
        self.cur.execute(sql, (paragraph_id, content, paper_arxiv_id, paper_section))
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

    # def create_citations_table(self):
    #     # citation edges by arxiv_id strings
    #     self.cur.execute("""
    #     CREATE TABLE IF NOT EXISTS citations (
    #         citing_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    #         cited_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    #         citing_sections TEXT[] DEFAULT '{}',
    #         PRIMARY KEY (citing_arxiv_id, cited_arxiv_id),
    #         CHECK (citing_arxiv_id <> cited_arxiv_id)
    #     )
    #     """)

    def insert_citation(self, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, author_cited_paper, citing_sections):
        """
        Insert a citation edge by arxiv_id strings. Returns True if inserted, False if exists or invalid.
        """

        # self.cur.execute("""
        # CREATE TABLE IF NOT EXISTS citations (
        #     id SERIAL PRIMARY KEY,
        #     citing_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
        #     cited_arxiv_id VARCHAR(100),
        #     bib_title TEXT,
        #     bib_key VARCHAR(255),
        #     author_cited_paper VARCHAR(255),
        #     citing_sections TEXT[] DEFAULT '{}',
        # )
        # """)

        if citing_arxiv_id == cited_arxiv_id:
            return False
        sql = """
        INSERT INTO citations (citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, author_cited_paper, citing_sections)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (citing_arxiv_id, cited_arxiv_id) DO NOTHING
        """
        self.cur.execute(sql, (citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, author_cited_paper, citing_sections))
        return self.cur.rowcount == 1


    def insert_citation_paragraph(self, paper_arxiv_id: str, paragraph_id: str, bib_key: str) -> None:
        """
        Add a paragraph reference to the citing_paragraphs array for a given citation.
        If the paragraph_id is already present, this will add a duplicate; if you
        want to avoid duplicates, see the note below.
        """
        sql = """
        UPDATE citations
        SET citing_paragraphs = array_append(
            COALESCE(citing_paragraphs, '{}'), %s)
        WHERE citing_arxiv_id = %s
        AND bib_key = %s
        """
        try:
            self.cur.execute(sql, (paragraph_id, paper_arxiv_id, bib_key))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise

        # Optional: check rowcount to see if an update actually happened
        if self.cur.rowcount == 0:
            # no existing citation row to update
            # you could choose to INSERT a new citation here if that makes sense:
            # self.insert_citation(paper_arxiv_id, ..., citing_paragraphs=[paragraph_id])
            pass



        # CREATE TABLE IF NOT EXISTS citations (
        #     id SERIAL PRIMARY KEY,
        #     paragraph_id   INT    NOT NULL,
        #     paper_section TEXT,
        #     citing_arxiv_id VARCHAR(100) NOT NULL
        #         REFERENCES papers(arxiv_id)
        #         ON DELETE CASCADE,
        #     bib_key VARCHAR(255),
        # )


    def insert_paragraph_citations(self, paragraph_id, paper_section, citing_arxiv_id, bib_key):
        sql = """
        INSERT INTO paragraph_citations
        (paragraph_id, paper_section, citing_arxiv_id, bib_key)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """
        self.cur.execute(sql, (paragraph_id, paper_section, citing_arxiv_id, bib_key))
        res = self.cur.fetchone()
        return res[0] if res else None

        # self.cur.execute("""
        # CREATE TABLE IF NOT EXISTS paragraph_references (
        #     id SERIAL PRIMARY KEY,
        #     paragraph_id   INT    NOT NULL,
        #     paper_section TEXT,
        #     paper_arxiv_id VARCHAR(100) NOT NULL
        #         REFERENCES papers(arxiv_id)
        #         ON DELETE CASCADE,
        #     reference_label TEXT    NOT NULL,
        #     reference_type  TEXT
        # );
        # """)

    def insert_paragraph_reference(self, paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type=None):
        sql = """
        INSERT INTO paragraph_references
        (paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """
        self.cur.execute(sql,
                        (paragraph_id,
                        paper_section,
                        paper_arxiv_id,
                        reference_label,
                        reference_type))
        res = self.cur.fetchone()
        return res[0] if res else None

        

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

    def check_exist_figure(self, bib_key):
        entry = f"\\label{{{bib_key}}}"
        sql = """
        SELECT EXISTS(
            SELECT 1
            FROM figures
            WHERE label = %s
        );
        """

        self.cur.execute(sql, (entry,))
        exists, = self.cur.fetchone()
        return exists
    
    def check_exist_table(self, bib_key):
        entry = f"\\label{{{bib_key}}}"
        print(entry)
        sql = """
        SELECT EXISTS(
            SELECT 1
            FROM tables
            WHERE label = %s
        );
        """
        
        self.cur.execute(sql, (entry,))
        exists, = self.cur.fetchone()
        return exists

    def check_exist(self, paper_arxiv_id):
        """
        Check if the paper with given arxiv id exists in the database
        Return True or False as boolean value
        - paper_arxiv_id: str
        """

        # TODO: this should be removed later

        # return False
        sql = """
        SELECT EXISTS(
            SELECT 1
            FROM papers
            WHERE arxiv_id = %s
        );
        """
        # execute the query
        self.cur.execute(sql, (paper_arxiv_id,))
        # fetchone returns a tuple like (True,) or (False,)
        exists, = self.cur.fetchone()
        return exists

    def paper_authors_exist(self, paper_arxiv_id):
        """
        Check if the paper with given arxiv id exists in the author database.
        If not, it means that the paper with arxiv id is not yet added into the semantic scholar or previous fetching failed.
        - paper_arxiv_id: str
        """

        sql = """
        SELECT EXISTS(
            SELECT 1
            FROM paper_authors
            WHERE paper_arxiv_id = %s
        )
        """

        self.cur.execute(sql, (paper_arxiv_id,))

        exists, = self.cur.fetchone()

        return exists

    def _dict_from_cursor(self, cursor, parser= None):
        """Convert last SELECT into list of dicts, applying parser to string fields if given."""
        cols = [col.name for col in cursor.description]
        result = []
        for row in cursor.fetchall():
            rd = {}
            for col, val in zip(cols, row):
                rd[col] = parser(val) if parser and isinstance(val, str) else val
            result.append(rd)
        return result
    
    def serialize_table(self, table_name, schema, parser=None):
        """Fetch all rows from schema.table_name and return list of dicts."""
        sql = f'SELECT * FROM "{schema}"."{table_name}"'
        with self.conn.cursor() as cur:
            cur.execute(sql)
            return self._dict_from_cursor(cur, parser)


    def serialize_all(self, tables, schema = 'public', parser = None):
        """Dump multiple tables into a dict, applying parser to string fields."""
        if tables is None:
            tables = [
                'papers', 'sections', 'authors', 'categories', 'institutions',
                'figures', 'tables', 'paper_authors', 'paper_category',
                'citations', 'paper_figures', 'paper_tables', 'author_affiliation'
            ]
        out = {}
        for t in tables:
            out[t] = self.serialize_table(t, schema, parser)
        return out

    def export_to_json(self, path: str, tables = None, parser = None, **json_kwargs) -> None:
        """Write out schema (or subset) to JSON file, parsing text if provided."""
        data = self.serialize_all(tables, parser=parser)
        with open(path, 'w') as f:
            json.dump(data, f, **json_kwargs)


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
            "papers",
            "sections"
        ]
        for tbl in tables:
            # Use IF EXISTS to avoid errors if a table is already gone
            # Use CASCADE for safety in case there are lingering dependencies
            self.cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
    
