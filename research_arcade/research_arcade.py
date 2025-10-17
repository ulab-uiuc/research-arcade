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
            return self.openreview_reviews.update_review(**node_features)
        elif table == 'openreview_revisions':
            return self.openreview_revisions.update_revision(**node_features)
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
    
    def insert_edge(self, table: str, edge_features: dict):
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
        else:
            print(f"Table {table} not found.")
            return None
    
    def delete_edge_by_id(self, table: str, primary_key: dict):
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
    
    def get_all_edge_features(self, table: str):
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
        else:
            print(f"Table {table} not found.")
            return None
        
    def get_neighborhood(self, table: str, primary_key: dict):
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
        else:
            print(f"Table {table} not found.")
            return None
        
    def construct_table_from_api(self, table: str, config: dict):
        if table == "openreview_papers":
            self.openreview_papers.construct_papers_table_from_api(**config)
        elif table == "openreview_authors":
            self.openreview_authors.construct_authors_table_from_api(**config)
        elif table == "openreview_reviews":
            self.openreview_reviews.construct_reviews_table_from_api(**config)
        elif table == "openreview_revisions":
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
            
    def construct_table_from_csv(self, table: str, config: dict):
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
            
    def construct_table_from_json(self, table: str, config: dict):
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