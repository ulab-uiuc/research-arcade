from .csv_database import CSVOpenReviewArxiv, CSVOpenReviewAuthors, CSVOpenReviewPapersAuthors, \
    CSVOpenReviewPapersReviews, CSVOpenReviewPapersRevisions, CSVOpenReviewPapers, CSVOpenReviewReviews, \
    CSVOpenReviewRevisionsReviews, CSVOpenReviewRevisions
from .sql_database import SQLOpenReviewArxiv, SQLOpenReviewAuthors, SQLOpenReviewPapersAuthors, \
    SQLOpenReviewPapersReviews, SQLOpenReviewPapersRevisions, SQLOpenReviewPapers, SQLOpenReviewReviews, \
    SQLOpenReviewRevisionsReviews, SQLOpenReviewRevisions
    
class ResearchArcade:
    def __init__(self, db_type: str, config: dict):
        if db_type == 'csv':
            self.openreview_arxiv = CSVOpenReviewArxiv(**config)
            self.openreview_authors = CSVOpenReviewAuthors(**config)
            self.openreview_papers_authors = CSVOpenReviewPapersAuthors(**config)
            self.openreview_papers_reviews = CSVOpenReviewPapersReviews(**config)
            self.openreview_papers_revisions = CSVOpenReviewPapersRevisions(**config)
            self.openreview_papers = CSVOpenReviewPapers(**config)
            self.openreview_reviews = CSVOpenReviewReviews(**config)
            self.openreview_revisions_reviews = CSVOpenReviewRevisionsReviews(**config)
            self.openreview_revisions = CSVOpenReviewRevisions(**config)
        elif db_type == 'sql':
            self.openreview_arxiv = SQLOpenReviewArxiv(**config)
            self.openreview_authors = SQLOpenReviewAuthors(**config)
            self.openreview_papers_authors = SQLOpenReviewPapersAuthors(**config)
            self.openreview_papers_reviews = SQLOpenReviewPapersReviews(**config)
            self.openreview_papers_revisions = SQLOpenReviewPapersRevisions(**config)
            self.openreview_papers = SQLOpenReviewPapers(**config)
            self.openreview_reviews = SQLOpenReviewReviews(**config)
            self.openreview_revisions_reviews = SQLOpenReviewRevisionsReviews(**config)
            self.openreview_revisions = SQLOpenReviewRevisions(**config)
    
    def insert_node(self, table: str, node_features: dict):
        if table == 'openreview_authors':
            return self.openreview_authors.insert_author(**node_features)
        elif table == 'openreview_papers':
            return self.openreview_papers.insert_paper(**node_features)
        elif table == 'openreview_reviews':
            return self.openreview_reviews.insert_review(**node_features)
        elif table == 'openreview_revisions':
            return self.openreview_revisions.insert_revision(**node_features)
        else:
            print(f"Table {table} not found.")
            return None
    
    def delete_node_by_id(self, table: str, primary_key: dict):
        if table == 'openreview_authors':
            return self.openreview_authors.delete_author_by_id(**primary_key)
        elif table == 'openreview_papers':
            return self.openreview_papers.delete_paper_by_id(**primary_key)
        elif table == 'openreview_reviews':
            return self.openreview_reviews.delete_review_by_id(**primary_key)
        elif table == 'openreview_revisions':
            return self.openreview_revisions.delete_revision_by_id(**primary_key)
        else:
            print(f"Table {table} not found.")
            return None

    def update_node(self, table: str, node_features: dict):
        if table == 'openreview_authors':
            return self.openreview_authors.update_author(**node_features)
        elif table == 'openreview_papers':
            return self.openreview_papers.update_paper(**node_features)
        elif table == 'openreview_reviews':
            return self.openreview_reviews.update_review_by_id(**node_features)
        elif table == 'openreview_revisions':
            return self.openreview_revisions.update_revision_by_id(**node_features)
        else:
            print(f"Table {table} not found.")
            return None
    
    def get_node_features_by_id(self, table: str, primary_key: dict):
        if table == 'openreview_authors':
            return self.openreview_authors.get_author_by_id(**primary_key)
        elif table == 'openreview_papers':
            return self.openreview_papers.get_paper_by_id(**primary_key)
        elif table == 'openreview_reviews':
            return self.openreview_reviews.get_review_by_id(**primary_key)
        elif table == 'openreview_revisions':
            return self.openreview_revisions.get_revision_by_id(**primary_key)
        else:
            print(f"Table {table} not found.")
            return None
    
    def get_all_node_features(self, table: str):
        if table == 'openreview_authors':
            return self.openreview_authors.get_all_authors(is_all_features=True)
        elif table == 'openreview_papers':
            return self.openreview_papers.get_all_papers(is_all_features=True)
        elif table == 'openreview_reviews':
            return self.openreview_reviews.get_all_reviews(is_all_features=True)
        elif table == 'openreview_revisions':
            return self.openreview_revisions.get_all_revisions(is_all_features=True)
        else:
            print(f"Table {table} not found.")
            return None
    
    def get_all_edge_features(self, table: str):
        if table == 'openreview_arxiv':
            return self.openreview_arxiv.get_all_edge_features()
        elif table == 'openreview_papers_authors':
            return self.openreview_papers_authors.get_all_edge_features()
        elif table == 'openreview_papers_reviews':
            return self.openreview_papers_reviews.get_all_edge_features()
        elif table == 'openreview_papers_revisions':
            return self.openreview_papers_revisions.get_all_edge_features()
        elif table == 'openreview_revisions_reviews':
            return self.openreview_revisions_reviews.get_all_edge_features()
        else:
            print(f"Table {table} not found.")
            return None