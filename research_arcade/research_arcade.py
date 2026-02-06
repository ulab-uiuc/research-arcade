import json
from pathlib import Path
from typing import Dict, Tuple

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
    CSVArxivParagraphs, CSVArxivSections, CSVArxivTable, CSVArxivParagraphCitation, CSVArxivParagraphFigure, CSVArxivParagraphTable
)
# Arxiv SQL
from .sql_database import (
    SQLArxivAuthors, SQLArxivCategory, SQLArxivCitation, SQLArxivFigure,
    SQLArxivPaperAuthor, SQLArxivPaperCategory, SQLArxivPaperFigure,
    SQLArxivPaperTable, SQLArxivPapers, SQLArxivParagraphReference,
    SQLArxivParagraphs, SQLArxivSections, SQLArxivTable, SQLArxivParagraphCitation, SQLArxivParagraphFigure, SQLArxivParagraphTable
)

import os
# from paper_crawler.crawler_job import CrawlerJob

from dotenv import load_dotenv
from typing import List, Optional
import pandas as pd
from datetime import date, datetime, timedelta
import os
import time
from typing import Dict
from arxiv_utils.continuous_crawling import run_single_crawl, get_interval_seconds

class ResearchArcade:
    def __init__(self, db_type: str, config: dict) -> None:
        load_dotenv()
        self.db_type = db_type
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
            self.arxiv_paragraph_citation = CSVArxivParagraphCitation(**config)
            self.arxiv_paragraph_figure = CSVArxivParagraphFigure(**config)
            self.arxiv_paragraph_table = CSVArxivParagraphTable(**config)
            
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
            self.arxiv_paragraph_citation = SQLArxivParagraphCitation(**config)
            self.arxiv_paragraph_figure = SQLArxivParagraphFigure(**config)
            self.arxiv_paragraph_table = SQLArxivParagraphTable(**config)

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
        elif table == 'arxiv_paragraph_figure':
            return self.arxiv_paragraph_figure.insert_paragraph_figure_table(**edge_features)
        elif table == 'arxiv_paragraph_table':
            return self.arxiv_paragraph_table.insert_paragraph_table_table(**edge_features)
        elif table == 'arxiv_paragraph_citation':
            return self.arxiv_paragraph_citation.insert_paragraph_reference(**edge_features)
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
        elif table == 'arxiv_paragraph_figure':
            # Expect keys: 'paragraph_id' and/or 'reference_id'
            if "paragraph_id" in primary_key and "figure_id" in primary_key:
                return self.arxiv_paragraph_figure.delete_paragraph_figure_by_paragraph_figure_id(**primary_key)
            elif "paragraph_id" in primary_key:
                return self.arxiv_paragraph_figure.delete_paragraph_figure_by_paragraph_id(**primary_key)
            elif "figure_id" in primary_key:
                return self.arxiv_paragraph_figure.delete_paragraph_figure_by_figure_id(**primary_key)
            else:
                print("For arxiv_paragraph_reference, primary key should include 'paragraph_id' and/or 'reference_id'.")
                return None
        elif table == 'arxiv_paragraph_table':
            # Expect keys: 'paragraph_id' and/or 'reference_id'
            if "paragraph_id" in primary_key and "table_id" in primary_key:
                return self.arxiv_paragraph_table.delete_paragraph_table_by_paragraph_table_id(**primary_key)
            elif "paragraph_id" in primary_key:
                return self.arxiv_paragraph_table.delete_paragraph_table_by_paragraph_id(**primary_key)
            elif "table_id" in primary_key:
                return self.arxiv_paragraph_table.delete_paragraph_table_by_table_id(**primary_key)
            else:
                print("For arxiv_paragraph_reference, primary key should include 'paragraph_id' and/or 'reference_id'.")
                return None
        elif table == 'arxiv_paragraph_citation':
            if "paragraph_id" in primary_key:
                return self.arxiv_paragraph_citation.delete_paragraph_citation_by_paragraph_id(**primary_key)
            elif "citing_arxiv_id" in primary_key:
                return self.arxiv_paragraph_citation.delete_paragraph_citation_by_citing_arxiv_id(**primary_key)
            elif "cited_arxiv_id" in primary_key:
                return self.arxiv_paragraph_citation.delete_paragraph_citation_by_cited_arxiv_id(**primary_key)
            elif "id" in primary_key:
                return self.arxiv_paragraph_citation.delete_paragraph_citation_by_id(**primary_key)
            else:
                print("For arxiv_paragraph_citation, primary key should include 'paragraph_id', 'citing_arxiv_id', 'cited_arxiv_id', or 'id'.")
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
        elif table == 'arxiv_paragraph_figure':
            return self.arxiv_paragraph_figure.get_all_paragraph_figures()
        elif table == 'arxiv_paragraph_table':
            return self.arxiv_paragraph_table.get_all_paragraph_tables()
        elif table == 'arxiv_paragraph_citation':
            return self.arxiv_paragraph_citation.get_all_paragraph_references()
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
        elif table == 'arxiv_paragraph_figure':
            if "paragraph_id" in primary_key:
                return self.arxiv_paragraph_figure.get_paragraph_neighboring_figures(**primary_key)
            elif "figure_id" in primary_key:
                return self.arxiv_paragraph_figure.get_figure_neighboring_paragraphs(**primary_key)
            else:
                print("For arxiv_paragraph_figure, provide 'paragraph_id' or 'figure_id'.")
                return None
        elif table == 'arxiv_paragraph_table':
            if "paragraph_id" in primary_key:
                return self.arxiv_paragraph_table.get_paragraph_neighboring_tables(**primary_key)
            elif "table_id" in primary_key:
                return self.arxiv_paragraph_table.get_table_neighboring_paragraphs(**primary_key)
            else:
                print("For arxiv_paragraph_table, provide 'paragraph_id' or 'table_id'.")
                return None
        elif table == 'arxiv_paragraph_citation':
            if "paragraph_id" in primary_key:
                return self.arxiv_paragraph_citation.get_paragraph_neighboring_citations(**primary_key)
            elif "paragraph_global_id" in primary_key:
                return self.arxiv_paragraph_citation.get_paragraph_global_id_neighboring_citations(**primary_key)
            elif "citing_arxiv_id" in primary_key:
                return self.arxiv_paragraph_citation.get_citations_by_citing_arxiv_id(**primary_key)
            elif "cited_arxiv_id" in primary_key:
                return self.arxiv_paragraph_citation.get_citations_by_cited_arxiv_id(**primary_key)
            else:
                print("For arxiv_paragraph_citation, provide 'paragraph_id', 'paragraph_global_id', 'citing_arxiv_id', or 'cited_arxiv_id'.")
                return None
        else:
            print(f"Table {table} not found.")
            return None


    def get_k_hop_neighbor_node_features(
        self, 
        table: str, 
        node_id: int, 
        k: int, 
        accumulative: bool = False, 
        neighbor_name: List[str] = None, 
        neighbourhood_features: List[List[str]] = None
    ) -> Optional[Dict[str, List[pd.DataFrame]]]:
        """
        Retrieve k-hop neighborhood node features using BFS traversal.
        
        :param table: The name of table that the target node belongs to
        :param node_id: The ID of the starting node
        :param k: The number of hops of neighborhood to retrieve
        :param accumulative: Whether to accumulate features from all hops (True) 
                            or just return features from the k-th hop (False)
        :param neighbor_name: Name of neighborhood tables to retrieve. 
                            If None, retrieve all neighboring tables.
        :param neighbourhood_features: Features to retrieve for each neighboring table.
                                    If None, retrieve all features.
        :return: Dictionary mapping table names to lists of feature DataFrames
        """
        
        if not neighbor_name:
            neighbor_name = self.get_all_neighbor_tables(table)
        if not neighbourhood_features:
            neighbourhood_features = [self.get_table_columns(n) for n in neighbor_name]
        
        # Edge table mapping: (source_table, target_table) -> edge_table_name
        edge_table_mapping = self.get_edge_table_mappings()
        
        # BFS state: track current frontier as (table_name, node_id) tuples
        current_frontier = [(table, node_id)]
        visited = {(table, node_id)}  # Avoid revisiting nodes
        
        # Result storage
        neighbor_features_dict = {}
        
        current_hop = 0
        while current_hop < k and current_frontier:
            next_frontier = []
            
            # Process each node in the current frontier
            for source_table, source_id in current_frontier:
                
                # Determine which neighbor tables to explore from this source table
                if source_table == table:
                    # From the original table, use the specified neighbor_name
                    tables_to_explore = neighbor_name
                    features_to_retrieve = neighbourhood_features
                else:
                    # From intermediate tables, explore all neighbors
                    tables_to_explore = self.get_all_neighbor_tables(source_table)
                    features_to_retrieve = [self.get_table_columns(n) for n in tables_to_explore]
                
                # Explore each neighboring table
                for i, target_table in enumerate(tables_to_explore):
                    edge_table_key = (source_table, target_table)
                    reverse_edge_key = (target_table, source_table)
                    
                    # Check both directions for edge table
                    if edge_table_key in edge_table_mapping:
                        edge_table_name = edge_table_mapping[edge_table_key]
                        source_col = f"{source_table}_id"
                        target_col = f"{target_table}_id"
                    elif reverse_edge_key in edge_table_mapping:
                        edge_table_name = edge_table_mapping[reverse_edge_key]
                        source_col = f"{source_table}_id"
                        target_col = f"{target_table}_id"
                    else:
                        print(f"No edge table found for {source_table} <-> {target_table}")
                        continue
                    
                    # Get neighboring nodes filtered by source node ID
                    neighboring_edges = self.get_neighborhood(
                        edge_table_name, 
                        {source_col: source_id}  # Filter by actual source node
                    )
                    
                    if neighboring_edges is None or neighboring_edges.empty:
                        continue
                    
                    # Process each neighboring node
                    for _, row in neighboring_edges.iterrows():
                        neighbor_id = row[target_col]
                        neighbor_key = (target_table, neighbor_id)
                        
                        # Skip if already visited
                        if neighbor_key in visited:
                            continue
                        visited.add(neighbor_key)
                        
                        # Add to next frontier for further exploration
                        next_frontier.append(neighbor_key)
                        
                        # Get node features
                        primary_key = {f"{target_table}_id": neighbor_id}
                        neighbor_features = self.get_node_features_by_id(target_table, primary_key)
                        
                        if neighbor_features is None or neighbor_features.empty:
                            print(f"No features found for {target_table} node {neighbor_id}")
                            continue
                        
                        # Select only required features
                        try:
                            selected_features = neighbor_features[features_to_retrieve[i]]
                        except (KeyError, IndexError):
                            selected_features = neighbor_features
                        
                        # Store features based on accumulative flag
                        # accumulative=True: store all hops
                        # accumulative=False: only store final hop (k-th hop)
                        if accumulative or (current_hop == k - 1):
                            neighbor_features_dict.setdefault(target_table, []).append(selected_features)
            
            # Move to next hop
            current_frontier = next_frontier
            current_hop += 1
        
        return neighbor_features_dict


    def path_search(self, start_table: str, start_id: str, target_table: str, target_id: str, max_depth: int = 5) -> Optional[List[List[str]]]:

        """
        Docstring for path_search
        This method searches for shortest paths from a starting node to a target node within a specified maximum depth in the heterogeneous graph in BFS manner.
        :param self: Description
        :param start_table: The name of the starting table
        :type start_table: str
        :param start_id: The ID of the starting node
        :type start_id: str
        :param target_table: The name of the target table
        :type target_table: str
        :param target_id: The ID of the target node
        :type target_id: str
        :param max_depth: The maximum depth to search for the target node, default is 5
        :type max_depth: int
        :return: Description
        :rtype: List[List[str]] | None
        """

        # We first load the edge table mappings
        edge_table_mapping = self.get_edge_table_mappings()

        # For each mapping, we will be able to see the neighborhood relationships
        neighbor_map = {}
        for (src_table, dst_table), edge_table in edge_table_mapping.items():
            neighbor_map.setdefault(src_table, []).append((dst_table, edge_table))
            neighbor_map.setdefault(dst_table, []).append((src_table, edge_table))
        # Now we perform BFS to find all paths from start to target within max_depth
        paths = []
        queue = [([start_table], start_id)]
        while queue:
            current_path, current_id = queue.pop(0)
            current_table = current_path[-1]
            if len(current_path) > max_depth:
                continue
            if current_table == target_table and current_id == target_id:
                paths.append(current_path)
                continue
            for neighbor_table, edge_table in neighbor_map.get(current_table, []):
                neighboring_nodes = self.get_neighborhood(edge_table, {f"{current_table}_id": current_id})
                for _, row in neighboring_nodes.iterrows():
                    neighbor_id = row[f"{neighbor_table}_id"]
                    new_path = current_path + [neighbor_table]
                    queue.append((new_path, neighbor_id))
        if paths:
            return paths
        else:
            print(f"No paths found from {start_table}({start_id}) to {target_table}({target_id}) within depth {max_depth}.")
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
            self.arxiv_paper_authors.construct_paper_authors_table_from_api(**config)
        elif table == "arxiv_paper_figures":
            self.arxiv_paper_figure.construct_paper_figures_table_from_api(**config)
        elif table == "arxiv_paper_tables":
            self.arxiv_paper_tables.construct_paper_tables_table_from_api(**config)
        elif table == "arxiv_paper_categories":
            self.arxiv_paper_category.construct_paper_category_table_from_api(**config)
        elif table == "arxiv_citations":
            self.arxiv_citation.construct_citations_table_from_api(**config)
        elif table == "arxiv_paragraph_references":
            self.arxiv_paragraph_reference.construct_paragraph_references_table_from_api(**config)
        elif table == "arxiv_paragraph_citations":
            self.arxiv_paragraph_citation.construct_citations_table_from_api(**config)
        elif table == "arxiv_paragraph_figures":
            self.arxiv_paragraph_figure.construct_paragraph_figures_table_from_api(**config)
        elif table == "arxiv_paragraph_tables":
            self.arxiv_paragraph_table.construct_paragraph_tables_table_from_api(**config)
        else:
            print(f"Table {table} does not support construction from API")

    def construct_table_from_csv(self, table: str, config: dict) -> Optional[pd.DataFrame]:
        # OpenReview tables
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
        # ArXiv tables - NODES
        elif table == "arxiv_papers":
            self.arxiv_papers.construct_table_from_csv(**config)
        elif table == "arxiv_authors":
            self.arxiv_authors.construct_table_from_csv(**config)
        elif table == "arxiv_categories":
            self.arxiv_categories.construct_table_from_csv(**config)
        elif table == "arxiv_figures":
            self.arxiv_figures.construct_table_from_csv(**config)
        elif table == "arxiv_tables":
            self.arxiv_tables.construct_table_from_csv(**config)
        elif table == "arxiv_sections":
            self.arxiv_sections.construct_table_from_csv(**config)
        elif table == "arxiv_paragraphs":
            self.arxiv_paragraphs.construct_table_from_csv(**config)
        # ArXiv tables - EDGES
        elif table == "arxiv_paper_citation":
            self.arxiv_citation.construct_table_from_csv(**config)
        elif table == "arxiv_paper_author":
            self.arxiv_paper_author.construct_table_from_csv(**config)
        elif table == "arxiv_paper_category":
            self.arxiv_paper_category.construct_table_from_csv(**config)
        elif table == "arxiv_paper_figure":
            self.arxiv_paper_figure.construct_table_from_csv(**config)
        elif table == "arxiv_paper_table":
            self.arxiv_paper_table.construct_table_from_csv(**config)
        elif table == "arxiv_paragraph_reference":
            self.arxiv_paragraph_reference.construct_table_from_csv(**config)
        elif table == "arxiv_paragraph_figure":
            self.arxiv_paragraph_figure.construct_table_from_csv(**config)
        elif table == "arxiv_paragraph_table":
            self.arxiv_paragraph_table.construct_table_from_csv(**config)
        elif table == "arxiv_paragraph_citation":
            self.arxiv_paragraph_citation.construct_table_from_csv(**config)
        else:
            print(f"Table {table} does not support construction from CSV")

    def construct_table_from_json(self, table: str, config: dict) -> Optional[pd.DataFrame]:
        # OpenReview tables
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
        # ArXiv tables - NODES
        elif table == "arxiv_papers":
            self.arxiv_papers.construct_table_from_json(**config)
        elif table == "arxiv_authors":
            self.arxiv_authors.construct_table_from_json(**config)
        elif table == "arxiv_categories":
            self.arxiv_categories.construct_table_from_json(**config)
        elif table == "arxiv_figures":
            self.arxiv_figures.construct_table_from_json(**config)
        elif table == "arxiv_tables":
            self.arxiv_tables.construct_table_from_json(**config)
        elif table == "arxiv_sections":
            self.arxiv_sections.construct_table_from_json(**config)
        elif table == "arxiv_paragraphs":
            self.arxiv_paragraphs.construct_table_from_json(**config)
        # ArXiv tables - EDGES
        elif table == "arxiv_paper_citation":
            self.arxiv_citation.construct_table_from_json(**config)
        elif table == "arxiv_paper_author":
            self.arxiv_paper_author.construct_table_from_json(**config)
        elif table == "arxiv_paper_category":
            self.arxiv_paper_category.construct_table_from_json(**config)
        elif table == "arxiv_paper_figure":
            self.arxiv_paper_figure.construct_table_from_json(**config)
        elif table == "arxiv_paper_table":
            self.arxiv_paper_table.construct_table_from_json(**config)
        elif table == "arxiv_paragraph_reference":
            self.arxiv_paragraph_reference.construct_table_from_json(**config)
        elif table == "arxiv_paragraph_figure":
            self.arxiv_paragraph_figure.construct_table_from_json(**config)
        elif table == "arxiv_paragraph_table":
            self.arxiv_paragraph_table.construct_table_from_json(**config)
        elif table == "arxiv_paragraph_citation":
            self.arxiv_paragraph_citation.construct_table_from_json(**config)
        else:
            print(f"Table {table} does not support construction from JSON")


    def construct_tables_from_arxiv_ids(self, config: dict) -> Optional[pd.DataFrame]:

        # Use sequential construction
        self.arxiv_papers.construct_papers_table_from_api(**config)
        self.arxiv_sections.construct_sections_table_from_api(**config)
        self.arxiv_authors.construct_authors_table_from_api(**config)
        self.arxiv_categories.construct_category_table_from_api(**config)
        self.arxiv_figures.construct_figures_table_from_api(**config)
        self.arxiv_tables.construct_tables_table_from_api(**config)
        self.arxiv_categories.construct_category_table_from_api(**config)
        self.arxiv_paper_author.construct_paper_authors_table_from_api(**config)
        self.arxiv_paper_figure.construct_paper_figures_table_from_api(**config)
        self.arxiv_paper_table.construct_paper_tables_table_from_api(**config)
        self.arxiv_paper_category.construct_paper_category_table_from_api(**config)
        self.arxiv_citation.construct_citations_table_from_api(**config)
        self.arxiv_paragraphs.construct_paragraphs_table_from_api(**config)
        self.arxiv_paragraph_reference.construct_paragraph_references_table_from_api(**config)
        self.arxiv_paragraph_citation.construct_citations_table_from_api(**config)
        self.arxiv_paragraph_figure.construct_paragraph_figures_table_from_api(**config)
        self.arxiv_paragraph_table.construct_paragraph_tables_table_from_api(**config)

    def construct_tables_from_venue(self, config: dict) -> Optional[pd.DataFrame]:
        self.openreview_arxiv.construct_openreview_arxiv_table_from_api(config)
        self.openreview_authors.construct_authors_table_from_api(config)
        self.openreview_papers.construct_papers_table_from_api(config)
        self.openreview_reviews.construct_reviews_table_from_api(config)
        self.openreview_paragraphs.construct_paragraphs_table_from_api(config)
        self.openreview_revisions.construct_revisions_table_from_api(config)
        self.openreview_papers_authors.construct_papers_authors_table_from_api(config)
        self.openreview_papers_reviews.construct_papers_reviews_table_from_api(config)
        self.openreview_papers_revisions.construct_papers_revisions_table_from_api(config)
        #
        papers_reviews_df = self.get_all_edge_features("openreview_papers_reviews")
        papers_revisions_df = self.get_all_edge_features("openreview_papers_revisions")
        new_config = {"papers_reviews_df": papers_reviews_df, "papers_revisions_df": papers_revisions_df}
        self.openreview_revisions_reviews.construct_revisions_reviews_table("openreview_revisions_reviews", new_config)
    
    def sample_nodes(self, table_name: str, sample_size: int) -> Optional[pd.DataFrame]:
        """
        Samples a subset of nodes from the specified table and return the node ids and features
        :param table_name: The name of the table to sample from
        :type table_name: str
        :param sample_size: The number of nodes to sample
        :type sample_size: int
        :return: A DataFrame containing the sampled nodes, including ids and features
        :rtype: pd.DataFrame | None
        """
        if table_name == "arxiv_papers":
            return self.arxiv_papers.sample_papers(sample_size)
        elif table_name == "arxiv_authors":
            return self.arxiv_authors.sample_authors(sample_size)
        elif table_name == "arxiv_categories":
            return self.arxiv_categories.sample_categories(sample_size)
        elif table_name == "arxiv_figures":
            return self.arxiv_figures.sample_figures(sample_size)
        elif table_name == "arxiv_tables":
            return self.arxiv_tables.sample_tables(sample_size)
        elif table_name == "arxiv_sections":
            return self.arxiv_sections.sample_sections(sample_size)
        elif table_name == "arxiv_paragraphs":
            return self.arxiv_paragraphs.sample_paragraphs(sample_size)
        elif table_name == "openreview_papers":
            return self.openreview_papers.sample_papers(sample_size)
        elif table_name == "openreview_authors":
            return self.openreview_authors.sample_authors(sample_size)
        elif table_name == "openreview_reviews":
            return self.openreview_reviews.sample_reviews(sample_size)
        elif table_name == "openreview_revisions":
            return self.openreview_revisions.sample_revisions(sample_size)
        else:
            print(f"Table {table_name} not found for sampling.")
            return None
    
        
    def continuous_crawling(self, interval_days, delay_days, paper_category, dest_dir, arxiv_id_dest):
        """
        Runs the crawl process in an infinite loop.
        """
        interval_seconds = get_interval_seconds(interval_days)

        print(f"Starting continuous crawl mode")
        print(f"  Interval: {interval_days} days")
        print(f"  Delay: {delay_days} days")
        print(f"  Paper Categories: {paper_category or 'all'}")

        while True:
            start_date = (date.today() - timedelta(days=interval_days + delay_days - 1)).isoformat()
            end_date = (date.today() - timedelta(days=delay_days)).isoformat()

            arxiv_ids = run_single_crawl(
                start_date=start_date,
                end_date=end_date,
                paper_category=paper_category,
                dest_dir=dest_dir,
                arxiv_id_dest=arxiv_id_dest
            )

            if arxiv_ids is not None:
                # Process papers using self
                config = {
                    'arxiv_ids': arxiv_ids,
                    'dest_dir': os.getenv('PAPER_FOLDER_PATH')
                }
                self.construct_tables_from_arxiv_ids(config=config)
                print(f"[{datetime.now()}] Batch completed. Sleeping for {interval_days} days...")
            else:
                print(f"[{datetime.now()}] Batch failed. Will retry after sleep.")
                
            time.sleep(interval_seconds)
    
    def get_edge_table_mappings(self) -> Dict[Tuple[str, str], dict]:
        """
        Returns mapping:
        (source_node_type, target_node_type) -> edge metadata
        """
        path = "research_arcade/research_arcade/edge_mappings.json"

        with open(path, "r") as f:
            raw = json.load(f)

        mappings = {}
        for edge_name, spec in raw.items():
            key = (spec["source"], spec["target"])
            mappings[key] = {
                "edge_name": edge_name,
                **spec
            }

        return mappings
