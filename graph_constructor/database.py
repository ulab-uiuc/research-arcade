import psycopg2
from psycopg2.extras import Json
import json
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..research_arcade.data import *

class Database:

    def __init__(self, host="localhost", port="5433", dbname="postgres", 
                 user="cl195", password=None, autocommit=True):
        """
        Initialize database configuration.
        Note: Connection is created per operation, not stored.
        """
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.autocommit = autocommit
    
    def _get_connection(self):
        """
        Create and return a new database connection.
        """
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password
        )
        conn.autocommit = self.autocommit
        return conn
    
    def create_papers_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
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
            cur.close()
        finally:
            conn.close()

    def create_sections_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS sections (
                id SERIAL PRIMARY KEY,
                content TEXT,
                title TEXT,
                appendix BOOLEAN,
                paper_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE
            )
            """)
            cur.close()
        finally:
            conn.close()

    def create_paragraphs_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS paragraphs (
                id SERIAL PRIMARY KEY,
                paragraph_id INT NOT NULL,
                content TEXT,
                paper_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
                paper_section TEXT,
                UNIQUE (paragraph_id, paper_arxiv_id, paper_section)
            );
            """)
            cur.close()
        finally:
            conn.close()

    def create_authors_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS authors (
                id SERIAL PRIMARY KEY,
                semantic_scholar_id VARCHAR(100) UNIQUE,
                name VARCHAR(255) NOT NULL,
                homepage VARCHAR(255)
            )
            """)
            cur.close()
        finally:
            conn.close()

    def create_categories_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL,
                description TEXT
            )
            """)
            cur.close()
        finally:
            conn.close()

    def create_institutions_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS institutions (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                location VARCHAR(255)
            )
            """)
            cur.close()
        finally:
            conn.close()

    def create_figures_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS figures (
                id SERIAL PRIMARY KEY,
                paper_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
                path VARCHAR(500),
                caption TEXT,
                label TEXT,
                name TEXT
            )
            """)
            cur.close()
        finally:
            conn.close()

    def create_tables_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS tables (
                id SERIAL PRIMARY KEY,
                paper_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
                path VARCHAR(500),
                caption TEXT,
                label TEXT,
                table_text TEXT
            )
            """)
            cur.close()
        finally:
            conn.close()

    def create_paper_authors_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_authors (
                paper_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
                author_id VARCHAR(100) NOT NULL REFERENCES authors(semantic_scholar_id) ON DELETE CASCADE,
                author_sequence INT NOT NULL,
                PRIMARY KEY (paper_arxiv_id, author_id)
            )
            """)
            cur.close()
        finally:
            conn.close()

    def create_paper_category_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_category (
                paper_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
                category_id INT NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
                PRIMARY KEY (paper_arxiv_id, category_id)
            )
            """)
            cur.close()
        finally:
            conn.close()

    def create_citations_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
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
            cur.close()
        finally:
            conn.close()

    def create_paper_figures_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_figures (
                paper_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
                figure_id INT NOT NULL REFERENCES figures(id) ON DELETE CASCADE,
                PRIMARY KEY (paper_arxiv_id, figure_id)
            )
            """)
            cur.close()
        finally:
            conn.close()

    def create_paper_tables_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_tables (
                paper_arxiv_id VARCHAR(100) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
                table_id INT NOT NULL REFERENCES tables(id) ON DELETE CASCADE,
                PRIMARY KEY (paper_arxiv_id, table_id)
            )
            """)
            cur.close()
        finally:
            conn.close()
    
    def create_paragraph_citations_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
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
            cur.close()
        finally:
            conn.close()

    def create_paragraph_references_table(self):
        """
        Stores references that appear within paragraphs—
        e.g. figures, tables, equations, etc.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
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
            cur.close()
        finally:
            conn.close()

    def create_author_affiliation_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS author_affiliation (
                author_id INT NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
                institution_id INT NOT NULL REFERENCES institutions(id) ON DELETE CASCADE,
                PRIMARY KEY (author_id, institution_id)
            )
            """)
            cur.close()
        finally:
            conn.close()

    def create_citation_sch_table(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS citation_sch(
                id SERIAL PRIMARY KEY,
                arxiv_id VARCHAR(100) NOT NULL,
                paper_id VARCHAR(100),
                title TEXT NOT NULL,
                year VARCHAR(100),
                abstract TEXT,
                external_ids TEXT
            );
            """)
            cur.close()
        finally:
            conn.close()
        
    def create_all(self):
        """Create all tables in order respecting dependencies"""
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
        self.create_citation_sch_table()

    def insert_paper(self, arxiv_id, base_arxiv_id, version, title, abstract=None, submit_date=None, metadata=None):
        """Insert a paper. Returns the generated paper id."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO papers (arxiv_id, base_arxiv_id, version, title, abstract, submit_date, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (arxiv_id) DO NOTHING
            RETURNING id
            """
            meta_val = Json(metadata) if metadata is not None else None
            cur.execute(sql, (arxiv_id, base_arxiv_id, version, title, abstract, submit_date, meta_val))
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def insert_section(self, content, title, is_appendix, paper_arxiv_id):
        """Insert a section. Returns the generated section id"""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO sections (content, title, appendix, paper_arxiv_id)
            VALUES(%s, %s, %s, %s)
            RETURNING id
            """
            cur.execute(sql, (content, title, is_appendix, paper_arxiv_id))
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()
    
    def insert_paragraph(self, paragraph_id, content, paper_arxiv_id, paper_section):
        """Insert a paragraph. Returns the generated paragraph id"""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO paragraphs (paragraph_id, content, paper_arxiv_id, paper_section)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (paragraph_id, paper_arxiv_id, paper_section) DO NOTHING
            RETURNING id
            """
            cur.execute(sql, (paragraph_id, content, paper_arxiv_id, paper_section))
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def insert_author(self, semantic_scholar_id, name, homepage=None):
        """Insert an author. Returns the generated author id"""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO authors (semantic_scholar_id, name, homepage)
            VALUES (%s, %s, %s)
            ON CONFLICT (semantic_scholar_id) DO NOTHING
            RETURNING id
            """
            cur.execute(sql, (semantic_scholar_id, name, homepage))
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def insert_category(self, name, description=None):
        """Insert a category. Returns the category id"""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO categories (name, description)
            VALUES (%s, %s)
            ON CONFLICT (name) DO NOTHING
            RETURNING id
            """
            cur.execute(sql, (name, description))
            res = cur.fetchone()
            if res:
                cur.close()
                return res[0]
            # conflict: fetch existing
            cur.execute("SELECT id FROM categories WHERE name = %s", (name,))
            result = cur.fetchone()[0]
            cur.close()
            return result
        finally:
            conn.close()

    def insert_institution(self, name, location=None):
        """Insert an institution. Returns the generated institution id"""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO institutions (name, location)
            VALUES (%s, %s)
            RETURNING id
            """
            cur.execute(sql, (name, location))
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def insert_figure(self, paper_arxiv_id, path, caption=None, label=None, name=None):
        """Insert a figure. Returns the generated figure id."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO figures (paper_arxiv_id, path, caption, label, name)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """
            cur.execute(sql, (paper_arxiv_id, path, caption, label, name))
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def insert_table(self, paper_arxiv_id, path=None, caption=None, label=None, table_text=None):
        """Insert a table record. Returns the generated table id."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO tables (paper_arxiv_id, path, caption, label, table_text)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """
            cur.execute(sql, (paper_arxiv_id, path, caption, label, table_text))
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def insert_paper_author(self, paper_arxiv_id, author_id, author_sequence):
        """Link a paper and an author. Returns True if inserted, False if already exists."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO paper_authors (paper_arxiv_id, author_id, author_sequence)
            VALUES (%s, %s, %s)
            ON CONFLICT (paper_arxiv_id, author_id) DO NOTHING
            """
            cur.execute(sql, (paper_arxiv_id, author_id, author_sequence))
            result = cur.rowcount == 1
            cur.close()
            return result
        finally:
            conn.close()

    def insert_paper_category(self, paper_arxiv_id, category_id):
        """Link a paper and a category. Returns True if inserted, False if exists."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO paper_category (paper_arxiv_id, category_id)
            VALUES (%s, %s)
            ON CONFLICT (paper_arxiv_id, category_id) DO NOTHING
            """
            cur.execute(sql, (paper_arxiv_id, category_id))
            result = cur.rowcount == 1
            cur.close()
            return result
        finally:
            conn.close()

    def insert_citation(self, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, author_cited_paper, citing_sections):
        """Insert a citation edge. Returns True if inserted, False if exists or invalid."""
        if citing_arxiv_id == cited_arxiv_id:
            return False
        
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO citations (citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, author_cited_paper, citing_sections)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (citing_arxiv_id, cited_arxiv_id) DO NOTHING
            """
            cur.execute(sql, (citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, author_cited_paper, citing_sections))
            result = cur.rowcount == 1
            cur.close()
            return result
        finally:
            conn.close()

    def insert_citation_paragraph(self, paper_arxiv_id, paragraph_id, bib_key):
        """Add a paragraph reference to the citing_paragraphs array for a given citation."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            UPDATE citations
            SET citing_paragraphs = array_append(
                COALESCE(citing_paragraphs, '{}'), %s)
            WHERE citing_arxiv_id = %s
            AND bib_key = %s
            """
            cur.execute(sql, (paragraph_id, paper_arxiv_id, bib_key))
            if not self.autocommit:
                conn.commit()
            cur.close()
        except Exception as e:
            if not self.autocommit:
                conn.rollback()
            raise
        finally:
            conn.close()

    def insert_paragraph_citations(self, paragraph_id, paper_section, citing_arxiv_id, bib_key):
        """Insert paragraph citation reference. Returns the generated id"""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO paragraph_citations
            (paragraph_id, paper_section, citing_arxiv_id, bib_key)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """
            cur.execute(sql, (paragraph_id, paper_section, citing_arxiv_id, bib_key))
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def insert_paragraph_reference(self, paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type=None):
        """Insert paragraph reference. Returns the generated id"""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO paragraph_references
            (paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """
            cur.execute(sql, (paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type))
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def insert_paper_figure(self, paper_arxiv_id, figure_id):
        """Link paper to figure. Returns True if inserted, False if exists."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO paper_figures (paper_arxiv_id, figure_id)
            VALUES (%s, %s)
            ON CONFLICT (paper_arxiv_id, figure_id) DO NOTHING
            """
            cur.execute(sql, (paper_arxiv_id, figure_id))
            result = cur.rowcount == 1
            cur.close()
            return result
        finally:
            conn.close()

    def insert_paper_table(self, paper_arxiv_id, table_id):
        """Link paper to table. Returns True if inserted, False if exists."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO paper_tables (paper_arxiv_id, table_id)
            VALUES (%s, %s)
            ON CONFLICT (paper_arxiv_id, table_id) DO NOTHING
            """
            cur.execute(sql, (paper_arxiv_id, table_id))
            result = cur.rowcount == 1
            cur.close()
            return result
        finally:
            conn.close()

    def insert_author_affiliation(self, author_id, institution_id):
        """Link an author to an institution. Returns True if inserted, False if exists."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO author_affiliation (author_id, institution_id)
            VALUES (%s, %s)
            ON CONFLICT (author_id, institution_id) DO NOTHING
            """
            cur.execute(sql, (author_id, institution_id))
            result = cur.rowcount == 1
            cur.close()
            return result
        finally:
            conn.close()

    def check_exist_figure(self, bib_key):
        """Check if a figure with the given label exists"""
        entry = f"\\label{{{bib_key}}}"
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            SELECT EXISTS(
                SELECT 1
                FROM figures
                WHERE label = %s
            );
            """
            cur.execute(sql, (entry,))
            exists, = cur.fetchone()
            cur.close()
            return exists
        finally:
            conn.close()
    
    def check_exist_table(self, bib_key):
        """Check if a table with the given label exists"""
        entry = f"\\label{{{bib_key}}}"
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            SELECT EXISTS(
                SELECT 1
                FROM tables
                WHERE label = %s
            );
            """
            cur.execute(sql, (entry,))
            exists, = cur.fetchone()
            cur.close()
            return exists
        finally:
            conn.close()

    def check_exist(self, paper_arxiv_id):
        """Check if the paper with given arxiv id exists in the database"""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            SELECT EXISTS(
                SELECT 1
                FROM papers
                WHERE arxiv_id = %s
            );
            """
            cur.execute(sql, (paper_arxiv_id,))
            exists, = cur.fetchone()
            cur.close()
            return exists
        finally:
            conn.close()

    def paper_authors_exist(self, paper_arxiv_id):
        """Check if the paper with given arxiv id exists in the author database."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            SELECT EXISTS(
                SELECT 1
                FROM paper_authors
                WHERE paper_arxiv_id = %s
            )
            """
            cur.execute(sql, (paper_arxiv_id,))
            exists, = cur.fetchone()
            cur.close()
            return exists
        finally:
            conn.close()
    def insert_paper_obj(self, paper: Paper) -> Optional[int]:
        """Insert a paper from a Paper object. Returns the generated paper id."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO papers (arxiv_id, base_arxiv_id, version, title, abstract, submit_date, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (arxiv_id) DO NOTHING
            RETURNING id
            """
            from psycopg2.extras import Json
            meta_val = Json(paper.metadata) if paper.metadata is not None else None
            cur.execute(sql, (paper.arxiv_id, paper.base_arxiv_id, paper.version, 
                            paper.title, paper.abstract, paper.submit_date, meta_val))
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()
    
    def insert_section_obj(self, section: Section) -> Optional[int]:
        """Insert a section from a Section object. Returns the generated section id."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO sections (content, title, appendix, paper_arxiv_id)
            VALUES(%s, %s, %s, %s)
            RETURNING id
            """
            cur.execute(sql, (section.content, section.title, section.appendix, section.paper_arxiv_id))
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()
    
    def insert_paragraph_obj(self, paragraph: Paragraph) -> Optional[int]:
        """Insert a paragraph from a Paragraph object. Returns the generated paragraph id."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO paragraphs (paragraph_id, content, paper_arxiv_id, paper_section)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (paragraph_id, paper_arxiv_id, paper_section) DO NOTHING
            RETURNING id
            """
            cur.execute(sql, (paragraph.paragraph_id, paragraph.content, 
                            paragraph.paper_arxiv_id, paragraph.paper_section))
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()
    
    def insert_author_obj(self, author: Author) -> Optional[int]:
        """Insert an author from an Author object. Returns the generated author id."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO authors (semantic_scholar_id, name, homepage)
            VALUES (%s, %s, %s)
            ON CONFLICT (semantic_scholar_id) DO NOTHING
            RETURNING id
            """
            cur.execute(sql, (author.semantic_scholar_id, author.name, author.homepage))
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()
    
    def insert_category_obj(self, category: Category) -> Optional[int]:
        """Insert a category from a Category object. Returns the category id."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO categories (name, description)
            VALUES (%s, %s)
            ON CONFLICT (name) DO NOTHING
            RETURNING id
            """
            cur.execute(sql, (category.name, category.description))
            res = cur.fetchone()
            if res:
                cur.close()
                return res[0]
            cur.execute("SELECT id FROM categories WHERE name = %s", (category.name,))
            result = cur.fetchone()[0]
            cur.close()
            return result
        finally:
            conn.close()
    
    def insert_figure_obj(self, figure: Figure) -> Optional[int]:
        """Insert a figure from a Figure object. Returns the generated figure id."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO figures (paper_arxiv_id, path, caption, label, name)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """
            cur.execute(sql, (figure.paper_arxiv_id, figure.path, figure.caption, 
                            figure.label, figure.name))
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()
    
    def insert_table_obj(self, table: Table) -> Optional[int]:
        """Insert a table from a Table object. Returns the generated table id."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO tables (paper_arxiv_id, path, caption, label, table_text)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """
            cur.execute(sql, (table.paper_arxiv_id, table.path, table.caption, 
                            table.label, table.table_text))
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()
    
    def insert_paper_author_obj(self, paper_author: PaperAuthor) -> bool:
        """Insert a paper-author relationship. Returns True if inserted, False if exists."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO paper_authors (paper_arxiv_id, author_id, author_sequence)
            VALUES (%s, %s, %s)
            ON CONFLICT (paper_arxiv_id, author_id) DO NOTHING
            """
            cur.execute(sql, (paper_author.paper_arxiv_id, paper_author.author_id, 
                            paper_author.author_sequence))
            result = cur.rowcount == 1
            cur.close()
            return result
        finally:
            conn.close()
    
    def insert_citation_obj(self, citation: Citation) -> bool:
        """Insert a citation from a Citation object. Returns True if inserted, False if exists."""
        if citation.citing_arxiv_id == citation.cited_arxiv_id:
            return False
        
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO citations (citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, 
                                 author_cited_paper, citing_sections)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (citing_arxiv_id, cited_arxiv_id) DO NOTHING
            """
            cur.execute(sql, (citation.citing_arxiv_id, citation.cited_arxiv_id, 
                            citation.bib_title, citation.bib_key, citation.author_cited_paper, 
                            citation.citing_sections or []))
            result = cur.rowcount == 1
            cur.close()
            return result
        finally:
            conn.close()
    
    def insert_paragraph_citation_obj(self, para_citation: ParagraphCitation) -> Optional[int]:
        """Insert a paragraph citation. Returns the generated id."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO paragraph_citations (paragraph_id, paper_section, citing_arxiv_id, bib_key)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """
            cur.execute(sql, (para_citation.paragraph_id, para_citation.paper_section, 
                            para_citation.citing_arxiv_id, para_citation.bib_key))
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()
    
    def insert_paragraph_reference_obj(self, para_ref: ParagraphReference) -> Optional[int]:
        """Insert a paragraph reference. Returns the generated id."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO paragraph_references 
            (paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """
            cur.execute(sql, (para_ref.paragraph_id, para_ref.paper_section, 
                            para_ref.paper_arxiv_id, para_ref.reference_label, 
                            para_ref.reference_type))
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()
    
    # ============ OBJECT-BASED QUERY METHODS ============
    
    def get_paper(self, arxiv_id: str) -> Optional[Paper]:
        """Retrieve a paper by arxiv_id and return a Paper object."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            SELECT id, arxiv_id, base_arxiv_id, version, title, abstract, submit_date, metadata
            FROM papers WHERE arxiv_id = %s
            """
            cur.execute(sql, (arxiv_id,))
            row = cur.fetchone()
            cur.close()
            if row:
                return Paper(
                    id=row[0],
                    arxiv_id=row[1],
                    base_arxiv_id=row[2],
                    version=row[3],
                    title=row[4],
                    abstract=row[5],
                    submit_date=str(row[6]) if row[6] else None,
                    metadata=row[7]
                )
            return None
        finally:
            conn.close()
    
    def get_sections_for_paper(self, paper_arxiv_id: str) -> List[Section]:
        """Retrieve all sections for a paper."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            SELECT id, content, title, appendix, paper_arxiv_id
            FROM sections WHERE paper_arxiv_id = %s
            ORDER BY id
            """
            cur.execute(sql, (paper_arxiv_id,))
            rows = cur.fetchall()
            cur.close()
            return [Section(
                id=row[0],
                content=row[1],
                title=row[2],
                appendix=row[3],
                paper_arxiv_id=row[4]
            ) for row in rows]
        finally:
            conn.close()
    
    def get_paragraphs_for_paper(self, paper_arxiv_id: str) -> List[Paragraph]:
        """Retrieve all paragraphs for a paper."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            SELECT id, paragraph_id, content, paper_arxiv_id, paper_section
            FROM paragraphs WHERE paper_arxiv_id = %s
            ORDER BY paragraph_id
            """
            cur.execute(sql, (paper_arxiv_id,))
            rows = cur.fetchall()
            cur.close()
            return [Paragraph(
                id=row[0],
                paragraph_id=row[1],
                content=row[2],
                paper_arxiv_id=row[3],
                paper_section=row[4]
            ) for row in rows]
        finally:
            conn.close()
    
    def get_authors_for_paper(self, paper_arxiv_id: str) -> List[Tuple[Author, int]]:
        """Retrieve all authors for a paper with their sequence. Returns list of (Author, sequence) tuples."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            SELECT a.id, a.semantic_scholar_id, a.name, a.homepage, pa.author_sequence
            FROM authors a
            JOIN paper_authors pa ON a.semantic_scholar_id = pa.author_id
            WHERE pa.paper_arxiv_id = %s
            ORDER BY pa.author_sequence
            """
            cur.execute(sql, (paper_arxiv_id,))
            rows = cur.fetchall()
            cur.close()
            return [(Author(
                id=row[0],
                semantic_scholar_id=row[1],
                name=row[2],
                homepage=row[3]
            ), row[4]) for row in rows]
        finally:
            conn.close()
    
    def get_figures_for_paper(self, paper_arxiv_id: str) -> List[Figure]:
        """Retrieve all figures for a paper."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            SELECT id, paper_arxiv_id, path, caption, label, name
            FROM figures WHERE paper_arxiv_id = %s
            ORDER BY id
            """
            cur.execute(sql, (paper_arxiv_id,))
            rows = cur.fetchall()
            cur.close()
            return [Figure(
                id=row[0],
                paper_arxiv_id=row[1],
                path=row[2],
                caption=row[3],
                label=row[4],
                name=row[5]
            ) for row in rows]
        finally:
            conn.close()
    
    def get_citations_for_paper(self, citing_arxiv_id: str) -> List[Citation]:
        """Retrieve all citations made by a paper."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            sql = """
            SELECT id, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, 
                   author_cited_paper, citing_sections, citing_paragraphs
            FROM citations WHERE citing_arxiv_id = %s
            """
            cur.execute(sql, (citing_arxiv_id,))
            rows = cur.fetchall()
            cur.close()
            return [Citation(
                id=row[0],
                citing_arxiv_id=row[1],
                cited_arxiv_id=row[2],
                bib_title=row[3],
                bib_key=row[4],
                author_cited_paper=row[5],
                citing_sections=row[6],
                citing_paragraphs=row[7]
            ) for row in rows]
        finally:
            conn.close()


    def drop_all(self):
        """Drop all tables in reverse dependency order."""
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
            "sections",
            "paragraphs",
            "paragraph_citations",
            "paragraph_references",
            "citation_sch",
            "papers"
        ]
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            for tbl in tables:
                cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
            cur.close()
        finally:
            conn.close()


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
            "sections",
            "paragraphs",
            "paragraph_citations",
            "paragraph_references"
        ]
        for tbl in tables:
            # Use IF EXISTS to avoid errors if a table is already gone
            # Use CASCADE for safety in case there are lingering dependencies
            self.cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
    
