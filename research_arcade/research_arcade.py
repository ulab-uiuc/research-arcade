# OpenReview CSV 
from .csv_database import (
    CSVOpenReviewArxiv, CSVOpenReviewAuthors, CSVOpenReviewPapersAuthors,
    CSVOpenReviewPapersReviews, CSVOpenReviewPapersRevisions, CSVOpenReviewPapers,
    CSVOpenReviewReviews, CSVOpenReviewRevisionsReviews, CSVOpenReviewRevisions,
    CSVOpenReviewParagraphs
)
# OpenReview SQL
from .sql_database import (
    SQLOpenReviewArxiv, SQLOpenReviewAuthors, SQLOpenReviewPapersAuthors,
    SQLOpenReviewPapersReviews, SQLOpenReviewPapersRevisions, SQLOpenReviewPapers,
    SQLOpenReviewReviews, SQLOpenReviewRevisionsReviews, SQLOpenReviewRevisions,
    SQLOpenReviewParagraphs
)

# Arxiv CSV
from .csv_database import (
    CSVArxivAuthors, CSVArxivCategory, CSVArxivCitation, CSVArxivFigure,
    CSVArxivPaperAuthor, CSVArxivPaperCategory, CSVArxivPaperFigure,
    CSVArxivPaperTable, CSVArxivPapers, CSVArxivParagraphReference,
    CSVArxivParagraphs, CSVArxivSections, CSVArxivTable
)

# Arxiv SQL
from .sql_database import (
    SQLArxivAuthors, SQLArxivCategory, SQLArxivCitation, SQLArxivFigure,
    SQLArxivPaperAuthor, SQLArxivPaperCategory, SQLArxivPaperFigure,
    SQLArxivPaperTable, SQLArxivPapers, SQLArxivParagraphReference,
    SQLArxivParagraphs, SQLArxivSections, SQLArxivTable
)
import os
from dotenv import load_dotenv
from typing import Optional
import pandas as pd

class ResearchArcade:
    def __init__(self, db_type: str, config: dict) -> None:
        load_dotenv()
        if db_type == 'csv':
            if config["csv_dir"] is None:
                config["csv_dir"] = os.getenv('CSV_DATASET_FOLDER_PATH')
                
            """
            Below is the arxiv dataset
            """
            self.arxiv_authors = CSVArxivAuthors(**config)
            self.arxiv_categories = CSVArxivCategory(**config)
            self.arxiv_figures = CSVArxivFigure(**config)
            self.arxiv_tables = CSVArxivTable(**config)
            self.arxiv_papers = CSVArxivPapers(**config)
            self.arxiv_paragraphs = CSVArxivParagraphs(**config)
            self.arxiv_sections = CSVArxivSections(**config)
            self.arxiv_citation = CSVArxivCitation(**config)
            self.arxiv_paper_author = CSVArxivPaperAuthor(**config)
            self.arxiv_paper_category = CSVArxivPaperCategory(**config)
            self.arxiv_paper_figure = CSVArxivPaperFigure(**config)
            self.arxiv_paper_table = CSVArxivPaperTable(**config)
            self.arxiv_paragraph_reference = CSVArxivParagraphReference(**config)
            
            """
            Below is the openreview dataset
            """
            self.openreview_arxiv = CSVOpenReviewArxiv(**config)
            self.openreview_authors = CSVOpenReviewAuthors(**config)
            self.openreview_papers_authors = CSVOpenReviewPapersAuthors(**config)
            self.openreview_papers_reviews = CSVOpenReviewPapersReviews(**config)
            self.openreview_papers_revisions = CSVOpenReviewPapersRevisions(**config)
            self.openreview_papers = CSVOpenReviewPapers(**config)
            self.openreview_reviews = CSVOpenReviewReviews(**config)
            self.openreview_revisions_reviews = CSVOpenReviewRevisionsReviews(**config)
            self.openreview_revisions = CSVOpenReviewRevisions(**config)
            self.openreview_paragraphs = CSVOpenReviewParagraphs(**config)
        elif db_type == 'sql':
            if config["host"] is None:
                config["csv_dir"] = os.getenv('CSV_DATASET_FOLDER_PATH')
                
            """
            Below is the arxiv dataset
            """
            self.arxiv_authors = SQLArxivAuthors(**config)
            self.arxiv_categories = SQLArxivCategory(**config)
            self.arxiv_figures = SQLArxivFigure(**config)
            self.arxiv_tables = SQLArxivTable(**config)
            self.arxiv_papers = SQLArxivPapers(**config)
            self.arxiv_paragraphs = SQLArxivParagraphs(**config)
            self.arxiv_sections = SQLArxivSections(**config)
            self.arxiv_citation = SQLArxivCitation(**config)
            self.arxiv_paper_author = SQLArxivPaperAuthor(**config)
            self.arxiv_paper_category = SQLArxivPaperCategory(**config)
            self.arxiv_paper_figure = SQLArxivPaperFigure(**config)
            self.arxiv_paper_table = SQLArxivPaperTable(**config)
            self.arxiv_paragraph_reference = SQLArxivParagraphReference(**config)

            
            """
            Below is the openreview dataset
            """
            self.openreview_arxiv = SQLOpenReviewArxiv(**config)
            self.openreview_authors = SQLOpenReviewAuthors(**config)
            self.openreview_papers_authors = SQLOpenReviewPapersAuthors(**config)
            self.openreview_papers_reviews = SQLOpenReviewPapersReviews(**config)
            self.openreview_papers_revisions = SQLOpenReviewPapersRevisions(**config)
            self.openreview_papers = SQLOpenReviewPapers(**config)
            self.openreview_reviews = SQLOpenReviewReviews(**config)
            self.openreview_revisions_reviews = SQLOpenReviewRevisionsReviews(**config)
            self.openreview_revisions = SQLOpenReviewRevisions(**config)
            self.openreview_paragraphs = SQLOpenReviewParagraphs(**config)
    

    def insert_node(self, table: str, node_features: dict) -> Optional[tuple]:
        # Tables in openreview dataset
        if table == 'openreview_authors':
            return self.openreview_authors.insert_author(**node_features)
        elif table == 'openreview_papers':
            return self.openreview_papers.insert_paper(**node_features)
        elif table == 'openreview_reviews':
            return self.openreview_reviews.insert_review(**node_features)
        elif table == 'openreview_revisions':
            return self.openreview_revisions.insert_revision(**node_features)
        elif table == 'openreview_paragraphs':
            return self.openreview_paragraphs.insert_paragraph(**node_features)
        # Tables in arxiv dataset
        elif table == 'arxiv_authors':
            return self.arxiv_authors.insert_author(**node_features)
        elif table == 'arxiv_categories':
            return self.arxiv_categories.insert_category(**node_features)
        elif table == 'arxiv_figures':
            return self.arxiv_figures.insert_figure(**node_features)
        elif table == 'arxiv_tables':
            return self.arxiv_tables.insert_table(**node_features)
        elif table == 'arxiv_papers':
            return self.arxiv_papers.insert_paper(**node_features)
        elif table == 'arxiv_paragraphs':
            return self.arxiv_paragraphs.insert_paragraph(**node_features)
        elif table == 'arxiv_sections':
            return self.arxiv_sections.insert_section(**node_features)
        else:
            print(f"Table {table} not found.")
            return None
    
    def delete_node_by_id(self, table: str, primary_key: dict) -> Optional[pd.DataFrame]:
        # Tables in openreview dataset
        if table == 'openreview_authors':
            return self.openreview_authors.delete_author_by_id(**primary_key)
        elif table == 'openreview_papers':
            return self.openreview_papers.delete_paper_by_id(**primary_key)
        elif table == 'openreview_reviews':
            return self.openreview_reviews.delete_review_by_id(**primary_key)
        elif table == 'openreview_revisions':
            return self.openreview_revisions.delete_revision_by_id(**primary_key)
        elif table == 'openreview_paragraphs':
            return self.openreview_paragraphs.delete_paragraphs_by_paper_id(**primary_key)
        # Tables in arxiv dataset
        elif table == 'arxiv_authors':
            return self.arxiv_authors.delete_author_by_id(**primary_key)
        elif table == 'arxiv_categories':
            return self.arxiv_categories.delete_category_by_id(**primary_key)
        elif table == 'arxiv_figures':
            return self.arxiv_figures.delete_figure_by_id(**primary_key)
        elif table == 'arxiv_tables':
            return self.arxiv_tables.delete_table_by_id(**primary_key)
        elif table == 'arxiv_papers':
            return self.arxiv_papers.delete_paper_by_id(**primary_key)
        elif table == 'arxiv_paragraphs':
            return self.arxiv_paragraphs.delete_paragraph_by_id(**primary_key)
        elif table == 'arxiv_sections':
            return self.arxiv_sections.delete_section_by_id(**primary_key)

        else:
            print(f"Table {table} not found.")
            return None

    def update_node(self, table: str, node_features: dict) -> Optional[pd.DataFrame]:
        # Tables in openreview dataset
        if table == 'openreview_authors':
            return self.openreview_authors.update_author(**node_features)
        elif table == 'openreview_papers':
            return self.openreview_papers.update_paper(**node_features)
        elif table == 'openreview_reviews':
            return self.openreview_reviews.update_review(**node_features)
        elif table == 'openreview_revisions':
            return self.openreview_revisions.update_revision(**node_features)
        # Tables in arxiv dataset
        elif table == 'arxiv_authors':
            return self.arxiv_authors.update_author(**node_features)
        elif table == 'arxiv_categories':
            return self.arxiv_categories.update_category(**node_features)
        elif table == 'arxiv_figures':
            return self.arxiv_figures.update_figure(**node_features)
        elif table == 'arxiv_tables':
            return self.arxiv_tables.update_table(**node_features)
        elif table == 'arxiv_papers':
            return self.arxiv_papers.update_paper(**node_features)
        elif table == 'arxiv_paragraphs':
            return self.arxiv_paragraphs.update_paragraph(**node_features)
        elif table == 'arxiv_sections':
            return self.arxiv_sections.update_section(**node_features)
        else:
            print(f"Table {table} not found.")
            return None
    
    def get_node_features_by_id(self, table: str, primary_key: dict) -> Optional[pd.DataFrame]:
        # Tables in openreview dataset
        if table == 'openreview_authors':
            return self.openreview_authors.get_author_by_id(**primary_key)
        elif table == 'openreview_papers':
            return self.openreview_papers.get_paper_by_id(**primary_key)
        elif table == 'openreview_reviews':
            return self.openreview_reviews.get_review_by_id(**primary_key)
        elif table == 'openreview_revisions':
            return self.openreview_revisions.get_revision_by_id(**primary_key)
        elif table == 'openreview_paragraphs':
            return self.openreview_paragraphs.get_paragraphs_by_paper_id(**primary_key)
        # Tables in arxiv dataset
        elif table == 'arxiv_authors':
            return self.arxiv_authors.get_author_by_id(**primary_key)
        elif table == 'arxiv_categories':
            return self.arxiv_categories.get_category_by_id(**primary_key)
        elif table == 'arxiv_figures':
            return self.arxiv_figures.get_figure_by_id(**primary_key)
        elif table == 'arxiv_tables':
            return self.arxiv_tables.get_table_by_id(**primary_key)
        elif table == 'arxiv_papers':
            return self.arxiv_papers.get_paper_by_id(**primary_key)
        elif table == 'arxiv_paragraphs':
            return self.arxiv_paragraphs.get_paragraph_by_id(**primary_key)
        elif table == 'arxiv_sections':
            return self.arxiv_sections.get_section_by_id(**primary_key)

        else:
            print(f"Table {table} not found.")
            return None
    
    def get_all_node_features(self, table: str) -> Optional[pd.DataFrame]:
        # Openreview tables
        if table == 'openreview_authors':
            return self.openreview_authors.get_all_authors(is_all_features=True)
        elif table == 'openreview_papers':
            return self.openreview_papers.get_all_papers(is_all_features=True)
        elif table == 'openreview_reviews':
            return self.openreview_reviews.get_all_reviews(is_all_features=True)
        elif table == 'openreview_revisions':
            return self.openreview_revisions.get_all_revisions(is_all_features=True)
        elif table == 'openreview_paragraphs':
            return self.openreview_paragraphs.get_all_paragraphs(is_all_features=True)

        # Arxiv tables
        elif table == 'arxiv_authors':
            return self.arxiv_authors.get_all_authors(is_all_features=True)
        elif table == 'arxiv_categories':
            return self.arxiv_categories.get_all_categories(is_all_features=True)
        elif table == 'arxiv_figures':
            return self.arxiv_figures.get_all_figures(is_all_features=True)
        elif table == 'arxiv_tables':
            return self.arxiv_tables.get_all_tables(is_all_features=True)
        elif table == 'arxiv_papers':
            return self.arxiv_papers.get_all_papers(is_all_features=True)
        elif table == 'arxiv_paragraphs':
            return self.arxiv_paragraphs.get_all_paragraphs(is_all_features=True)
        elif table == 'arxiv_sections':
            return self.arxiv_sections.get_all_sections(is_all_features=True)
        else:
            print(f"Table {table} not found.")
            return None
    
    def insert_edge(self, table: str, edge_features: dict) -> Optional[pd.DataFrame]:
        # openreview
        if table == 'openreview_arxiv':
            return self.openreview_arxiv.insert_openreview_arxiv(**edge_features)
        elif table == "openreview_papers_authors":
            return self.openreview_papers_authors.insert_paper_authors(**edge_features)
        elif table == "openreview_papers_reviews":
            return self.openreview_papers_reviews.insert_paper_reviews(**edge_features)
        elif table == "openreview_papers_revisions":
            return self.openreview_papers_revisions.insert_paper_revisions(**edge_features)
        elif table == "openreview_revisions_reviews":
            return self.openreview_revisions_reviews.insert_revision_reviews(**edge_features)
        
        # arxiv
        elif table == 'arxiv_citation':
            return self.arxiv_citation.insert_citation(**edge_features)
        elif table == 'arxiv_paper_author':
            return self.arxiv_paper_author.insert_paper_author(**edge_features)
        elif table == 'arxiv_paper_category':
            return self.arxiv_paper_category.insert_paper_category(**edge_features)
        elif table == 'arxiv_paper_figure':
            return self.arxiv_paper_figure.insert_paper_figure(**edge_features)
        elif table == 'arxiv_paper_table':
            return self.arxiv_paper_table.insert_paper_table(**edge_features)
        elif table == 'arxiv_paragraph_reference':
            return self.arxiv_paragraph_reference.insert_paragraph_reference(**edge_features)
        else:
            print(f"Table {table} not found.")
            return None
    
    def delete_edge_by_id(self, table: str, primary_key: dict) -> Optional[pd.DataFrame]:
        # Openreview
        if table == 'openreview_arxiv':
            if "paper_openreview_id" in primary_key and "arxiv_id" in primary_key:
                return self.openreview_arxiv.delete_openreview_arxiv_by_id(**primary_key)
            elif "arxiv_id" in primary_key:
                return self.openreview_arxiv.delete_openreview_arxiv_by_arxiv_id(**primary_key)
            elif "paper_openreview_id" in primary_key:
                return self.openreview_arxiv.delete_openreview_arxiv_by_openreview_id(**primary_key)
            else:
                print("For openreview_arxiv table, the primary key should be 'paper_openreview_id' or 'arxiv_id'.")
                return None
        elif table == "openreview_papers_authors":
            if "paper_openreview_id" in primary_key and "author_openreview_id" in primary_key:
                return self.openreview_papers_authors.delete_paper_author_by_id(**primary_key)
            elif "paper_openreview_id" in primary_key:
                return self.openreview_papers_authors.delete_paper_author_by_paper_id(**primary_key)
            elif "author_openreview_id" in primary_key:
                return self.openreview_papers_authors.delete_paper_author_by_author_id(**primary_key)
            else:
                print("For openreview_papers_authors table, the primary key should be 'paper_openreview_id' or 'author_openreview_id'.")
                return None
        elif table == "openreview_papers_reviews":
            if "paper_openreview_id" in primary_key and "review_openreview_id" in primary_key:
                return self.openreview_papers_reviews.delete_paper_review_by_id(**primary_key)
            elif "paper_openreview_id" in primary_key:
                return self.openreview_papers_reviews.delete_paper_review_by_paper_id(**primary_key)
            elif "review_openreview_id" in primary_key:
                return self.openreview_papers_reviews.delete_paper_review_by_review_id(**primary_key)
            else:
                print("For openreview_papers_reviews table, the primary key should be 'paper_openreview_id' or 'review_openreview_id'.")
                return None
        elif table == "openreview_papers_revisions":
            if "paper_openreview_id" in primary_key and "revision_openreview_id" in primary_key:
                return self.openreview_papers_revisions.delete_paper_revision_by_id(**primary_key)
            elif "paper_openreview_id" in primary_key:
                return self.openreview_papers_revisions.delete_paper_revision_by_paper_id(**primary_key)
            elif "revision_openreview_id" in primary_key:
                return self.openreview_papers_revisions.delete_paper_revision_by_revision_id(**primary_key)
            else:
                print("For openreview_papers_revisions table, the primary key should be 'paper_openreview_id' or 'revision_openreview_id'.")
                return None
        elif table == "openreview_revisions_reviews":
            if "revision_openreview_id" in primary_key and "review_openreview_id" in primary_key:
                return self.openreview_revisions_reviews.delete_revision_review_by_id(**primary_key)
            elif "revision_openreview_id" in primary_key:
                return self.openreview_revisions_reviews.delete_revision_review_by_revision_id(**primary_key)
            elif "review_openreview_id" in primary_key:
                return self.openreview_revisions_reviews.delete_revision_review_by_review_id(**primary_key)
            else:
                print("For openreview_revisions_reviews table, the primary key should be 'revision_openreview_id' or 'review_openreview_id'.")
                return None
        # Arxiv
        elif table == 'arxiv_citation':
            # Expect keys: 'citing_paper_id' and 'cited_paper_id'
            if "citing_paper_id" in primary_key and "cited_paper_id" in primary_key:
                return self.arxiv_citation.delete_citation_by_id(**primary_key)
            elif "citing_paper_id" in primary_key:
                return self.arxiv_citation.delete_citation_by_citing_id(**primary_key)
            elif "cited_paper_id" in primary_key:
                return self.arxiv_citation.delete_citation_by_cited_id(**primary_key)
            else:
                print("For arxiv_citation, primary key should include 'citing_paper_id' and/or 'cited_paper_id'.")
                return None
        elif table == 'arxiv_paper_author':
            # Expect keys: 'paper_id' and/or 'author_id'
            if "paper_arxiv_id" in primary_key and "author_id" in primary_key:
                return self.arxiv_paper_author.delete_paper_author_by_id(**primary_key)
            elif "paper_arxiv_id" in primary_key:
                return self.arxiv_paper_author.delete_paper_author_by_paper_id(**primary_key)
            elif "author_id" in primary_key:
                return self.arxiv_paper_author.delete_paper_author_by_author_id(**primary_key)
            else:
                print("For arxiv_paper_author, primary key should include 'paper_id' and/or 'author_id'.")
                return None
        elif table == 'arxiv_paper_category':
            # Expect keys: 'paper_id' and/or 'category_id'
            if "paper_arxiv_id" in primary_key and "category_id" in primary_key:
                return self.arxiv_paper_category.delete_paper_category_by_id(**primary_key)
            elif "paper_arxiv_id" in primary_key:
                return self.arxiv_paper_category.delete_paper_category_by_paper_id(**primary_key)
            elif "category_id" in primary_key:
                return self.arxiv_paper_category.delete_paper_category_by_category_id(**primary_key)
            else:
                print("For arxiv_paper_category, primary key should include 'paper_id' and/or 'category_id'.")
                return None
        elif table == 'arxiv_paper_figure':
            # Expect keys: 'paper_id' and/or 'figure_id'
            if "paper_arxiv_id" in primary_key and "figure_id" in primary_key:
                return self.arxiv_paper_figure.delete_paper_figure_by_id(**primary_key)
            elif "paper_arxiv_id" in primary_key:
                return self.arxiv_paper_figure.delete_paper_figure_by_paper_id(**primary_key)
            elif "figure_id" in primary_key:
                return self.arxiv_paper_figure.delete_paper_figure_by_figure_id(**primary_key)
            else:
                print("For arxiv_paper_figure, primary key should include 'paper_id' and/or 'figure_id'.")
                return None
        elif table == 'arxiv_paper_table':
            # Expect keys: 'paper_id' and/or 'table_id'
            if "paper_arxiv_id" in primary_key and "table_id" in primary_key:
                return self.arxiv_paper_table.delete_paper_table_by_id(**primary_key)
            elif "paper_arxiv_id" in primary_key:
                return self.arxiv_paper_table.delete_paper_table_by_paper_id(**primary_key)
            elif "table_id" in primary_key:
                return self.arxiv_paper_table.delete_paper_table_by_table_id(**primary_key)
            else:
                print("For arxiv_paper_table, primary key should include 'paper_id' and/or 'table_id'.")
                return None
        elif table == 'arxiv_paragraph_reference':
            # Expect keys: 'paragraph_id' and/or 'reference_id'
            if "paragraph_id" in primary_key and "reference_id" in primary_key:
                return self.arxiv_paragraph_reference.delete_paragraph_reference_by_id(**primary_key)
            elif "paragraph_id" in primary_key:
                return self.arxiv_paragraph_reference.delete_paragraph_reference_by_paragraph_id(**primary_key)
            elif "reference_id" in primary_key:
                return self.arxiv_paragraph_reference.delete_paragraph_reference_by_reference_id(**primary_key)
            else:
                print("For arxiv_paragraph_reference, primary key should include 'paragraph_id' and/or 'reference_id'.")
                return None
        
        else:
            print(f"Table {table} not found.")
            return None

    def get_all_edge_features(self, table: str) -> Optional[pd.DataFrame]:
        # Openreview
        if table == 'openreview_arxiv':
            return self.openreview_arxiv.get_all_openreview_arxiv()
        elif table == 'openreview_papers_authors':
            return self.openreview_papers_authors.get_all_papers_authors()
        elif table == 'openreview_papers_reviews':
            return self.openreview_papers_reviews.get_all_papers_reviews()
        elif table == 'openreview_papers_revisions':
            return self.openreview_papers_revisions.get_all_papers_revisions()
        elif table == 'openreview_revisions_reviews':
            return self.openreview_revisions_reviews.get_all_revisions_reviews()

        # Arxiv
        elif table == 'arxiv_citation':
            return self.arxiv_citation.get_all_citations()
        elif table == 'arxiv_paper_author':
            return self.arxiv_paper_author.get_all_paper_authors()
        elif table == 'arxiv_paper_category':
            return self.arxiv_paper_category.get_all_paper_categories()
        elif table == 'arxiv_paper_figure':
            return self.arxiv_paper_figure.get_all_paper_figures()
        elif table == 'arxiv_paper_table':
            return self.arxiv_paper_table.get_all_paper_tables()
        elif table == 'arxiv_paragraph_reference':
            return self.arxiv_paragraph_reference.get_all_paragraph_references()
        else:
            print(f"Table {table} not found.")
            return None
        
    def get_neighborhood(self, table: str, primary_key: dict) -> Optional[pd.DataFrame]:
        # Openreview
        if table == 'openreview_arxiv':
            if "paper_openreview_id" in primary_key:
                return self.openreview_arxiv.get_openreview_neighboring_arxivs(**primary_key)
            elif "arxiv_id" in primary_key:
                return self.openreview_arxiv.get_arxiv_neighboring_openreviews(**primary_key)
            else:
                print("For openreview_arxiv table, the primary key should be either 'paper_openreview_id' or 'arxiv_id'.")
                return None
        elif table == 'openreview_papers_authors':
            if "paper_openreview_id" in primary_key:
                return self.openreview_papers_authors.get_paper_neighboring_authors(**primary_key)
            elif "author_openreview_id" in primary_key:
                return self.openreview_papers_authors.get_author_neighboring_papers(**primary_key)
            else:
                print("For openreview_papers_authors table, the primary key should be either 'paper_openreview_id' or 'author_openreview_id'.")
                return None
        elif table == 'openreview_papers_reviews':
            if "paper_openreview_id" in primary_key:
                return self.openreview_papers_reviews.get_paper_neighboring_reviews(**primary_key)
            elif "review_openreview_id" in primary_key:
                return self.openreview_papers_reviews.get_review_neighboring_papers(**primary_key)
            else:
                print("For openreview_papers_reviews table, the primary key should be either 'paper_openreview_id' or 'review_openreview_id'.")
                return None
        elif table == 'openreview_papers_revisions':
            if "paper_openreview_id" in primary_key:
                return self.openreview_papers_revisions.get_paper_neighboring_revisions(**primary_key)
            elif "revision_openreview_id" in primary_key:
                return self.openreview_papers_revisions.get_revision_neighboring_papers(**primary_key)
            else:
                print("For openreview_papers_revisions table, the primary key should be either 'paper_openreview_id' or 'revision_openreview_id'.")
                return None
        elif table == 'openreview_revisions_reviews':
            if "revision_openreview_id" in primary_key:
                return self.openreview_revisions_reviews.get_revision_neighboring_reviews(**primary_key)
            elif "review_openreview_id" in primary_key:
                return self.openreview_revisions_reviews.get_review_neighboring_revisions(**primary_key)
            else:
                print("For openreview_revisions_reviews table, the primary key should be either 'revision_openreview_id' or 'review_openreview_id'.")
                return None
        # Arxiv
        elif table == 'arxiv_citation':
            # Expect either 'citing_paper_id' or 'cited_paper_id'
            if "citing_paper_id" in primary_key:
                return self.arxiv_citation.get_citing_neighboring_cited(**primary_key)
            elif "cited_paper_id" in primary_key:
                return self.arxiv_citation.get_cited_neighboring_citing(**primary_key)
            else:
                print("For arxiv_citation, provide 'citing_paper_id' or 'cited_paper_id'.")
                return None
        elif table == 'arxiv_paper_author':
            if "paper_arxiv_id" in primary_key:
                return self.arxiv_paper_author.get_paper_neighboring_authors(**primary_key)
            elif "author_id" in primary_key:
                return self.arxiv_paper_author.get_author_neighboring_papers(**primary_key)
            else:
                print("For arxiv_paper_author, provide 'paper_id' or 'author_id'.")
                return None
        elif table == 'arxiv_paper_category':
            if "paper_arxiv_id" in primary_key:
                return self.arxiv_paper_category.get_paper_neighboring_categories(**primary_key)
            elif "category_id" in primary_key:
                return self.arxiv_paper_category.get_category_neighboring_papers(**primary_key)
            else:
                print("For arxiv_paper_category, provide 'paper_id' or 'category_id'.")
                return None
        elif table == 'arxiv_paper_figure':
            if "paper_arxiv_id" in primary_key:
                return self.arxiv_paper_figure.get_paper_neighboring_figures(**primary_key)
            elif "figure_id" in primary_key:
                return self.arxiv_paper_figure.get_figure_neighboring_papers(**primary_key)
            else:
                print("For arxiv_paper_figure, provide 'paper_id' or 'figure_id'.")
                return None
        elif table == 'arxiv_paper_table':
            if "paper_arxiv_id" in primary_key:
                return self.arxiv_paper_table.get_paper_neighboring_tables(**primary_key)
            elif "table_id" in primary_key:
                return self.arxiv_paper_table.get_table_neighboring_papers(**primary_key)
            else:
                print("For arxiv_paper_table, provide 'paper_id' or 'table_id'.")
                return None
        elif table == 'arxiv_paragraph_reference':
            if "paragraph_id" in primary_key:
                return self.arxiv_paragraph_reference.get_paragraph_neighboring_references(**primary_key)
            elif "reference_id" in primary_key:
                return self.arxiv_paragraph_reference.get_reference_neighboring_paragraphs(**primary_key)
            else:
                print("For arxiv_paragraph_reference, provide 'paragraph_id' or 'reference_id'.")
                return None
        
        else:
            print(f"Table {table} not found.")
            return None

    def construct_table_from_api(self, table: str, config: dict) -> Optional[pd.DataFrame]:
        if table == "openreview_papers":
            self.openreview_papers.construct_papers_table_from_api(**config)
        elif table == "openreview_authors":
            self.openreview_authors.construct_authors_table_from_api(**config)
        elif table == "openreview_reviews":
            self.openreview_reviews.construct_reviews_table_from_api(**config)
        elif table == "openreview_revisions":
            if config["pdf_dir"] is None:
                config["pdf_dir"] = os.getenv("PDF_FOLDER_PATH")
            self.openreview_revisions.construct_revisions_table_from_api(**config)
        elif table == "openreview_papers_authors":
            self.openreview_papers_authors.construct_papers_authors_table_from_api(**config)
        elif table == "openreview_papers_reviews":
            self.openreview_papers_reviews.construct_papers_reviews_table_from_api(**config)
        elif table == "openreview_papers_revisions":
            self.openreview_papers_revisions.construct_papers_revisions_table_from_api(**config)
        elif table == "openreview_revisions_reviews":
            self.openreview_revisions_reviews.construct_revisions_reviews_table(**config)
        elif table == "openreview_arxiv":
            self.openreview_arxiv.construct_openreview_arxiv_table_from_api(**config)
        elif table == "openreview_paragraphs":
            if config["pdf_dir"] is None:
                config["pdf_dir"] = os.getenv("PDF_FOLDER_PATH")
            self.openreview_paragraphs.construct_paragraphs_table_from_api(**config)
        elif table == "arxiv_papers":
            self.arxiv_papers.construct_papers_table_from_api(**config)
        elif table == "arxiv_authors":
            self.arxiv_authors.construct_authors_table_from_api(**config)
        elif table == "arxiv_categories":
            self.arxiv_categories.construct_category_table_from_api(**config)
        elif table == "arxiv_figures":
            self.arxiv_figures.construct_figures_table_from_api(**config)
        elif table == "arxiv_tables":
            self.arxiv_tables.construct_tables_table_from_api(**config)
        elif table == "arxiv_sections":
            self.arxiv_sections.construct_sections_table_from_api(**config)
        elif table == "arxiv_paragraphs":
            self.arxiv_paragraphs.construct_paragraphs_table_from_api(**config)
        elif table == "arxiv_categories":
            self.arxiv_categories.construct_category_table_from_api(**config)
        elif table == "arxiv_paper_authors":
            self.arxiv_paper_authors.construct_papers_table_from_api(**config)
        elif table == "arxiv_paper_figures":
            self.arxiv_paper_figures.construct_papers_table_from_api(**config)
        elif table == "arxiv_paper_tables":
            self.arxiv_paper_tables.construct_papers_table_from_api(**config)
        elif table == "arxiv_paper_categories":
            self.arxiv_paper_tables.construct_papers_table_from_api(**config)
        elif table == "arxiv_citations":
            self.arxiv_citations.construct_papers_table_from_api(**config)
        elif table == "arxiv_paragraph_references":
            self.arxiv_paragraph_references.construct_papers_table_from_api(**config)
        else:
            print(f"Table {table} does not support construction from API")
            
    def construct_table_from_csv(self, table: str, config: dict) -> Optional[pd.DataFrame]:
        if table == "openreview_papers":
            self.openreview_papers.construct_papers_table_from_csv(**config)
        elif table == "openreview_authors":
            self.openreview_authors.construct_authors_table_from_csv(**config)
        elif table == "openreview_reviews":
            self.openreview_reviews.construct_reviews_table_from_csv(**config)
        elif table == "openreview_revisions":
            self.openreview_revisions.construct_revisions_table_from_csv(**config)
        elif table == "openreview_papers_authors":
            self.openreview_papers_authors.construct_papers_authors_table_from_csv(**config)
        elif table == "openreview_papers_reviews":
            self.openreview_papers_reviews.construct_papers_reviews_table_from_csv(**config)
        elif table == "openreview_papers_revisions":
            self.openreview_papers_revisions.construct_papers_revisions_table_from_csv(**config)
        elif table == "openreview_revisions_reviews":
            self.openreview_revisions_reviews.construct_revisions_reviews_table_from_csv(**config)
        elif table == "openreview_arxiv":
            self.openreview_arxiv.construct_openreview_arxiv_table_from_csv(**config)
        elif table == "openreview_paragraphs":
            self.openreview_paragraphs.construct_paragraphs_table_from_csv(**config)
        else:
            print(f"Table {table} does not support construction from CSV")
    
    def construct_table_from_json(self, table: str, config: dict) -> Optional[pd.DataFrame]:
        if table == "openreview_papers":
            self.openreview_papers.construct_papers_table_from_json(**config)
        elif table == "openreview_authors":
            self.openreview_authors.construct_authors_table_from_json(**config)
        elif table == "openreview_reviews":
            self.openreview_reviews.construct_reviews_table_from_json(**config)
        elif table == "openreview_revisions":
            self.openreview_revisions.construct_revisions_table_from_json(**config)
        elif table == "openreview_papers_authors":
            self.openreview_papers_authors.construct_papers_authors_table_from_json(**config)
        elif table == "openreview_papers_reviews":
            self.openreview_papers_reviews.construct_papers_reviews_table_from_json(**config)
        elif table == "openreview_papers_revisions":
            self.openreview_papers_revisions.construct_papers_revisions_table_from_json(**config)
        elif table == "openreview_revisions_reviews":
            self.openreview_revisions_reviews.construct_revisions_reviews_table_from_json(**config)
        elif table == "openreview_arxiv":
            self.openreview_arxiv.construct_openreview_arxiv_table_from_json(**config)
        elif table == "openreview_paragraphs":
            self.openreview_paragraphs.construct_paragraphs_table_from_json(**config)
        else:
            print(f"Table {table} does not support construction from JSON")       
