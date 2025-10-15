import sys
import os
import ast
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from openreview_database import Database
import json

database = Database()

# paper
# paper_csv_path = "./csv_paper_example.csv"
# database.construct_paper_table_by_csv(paper_csv_path, with_pdf=False)
# df = database.get_node_features_by_id("papers", {"paper_openreview_id": "xujj_test"})
# print(df)
# database.delete_node_by_id("papers", {"paper_openreview_id": "xujj_test"})
# df = database.get_node_features_by_id("papers", {"paper_openreview_id": "xujj_test"})
# print(df)

# revision
# revision_csv_path = "./csv_revision_example.csv"
# database.construct_revision_table_by_csv(revision_csv_path, with_pdf=False)
# df = database.get_node_features_by_id("revisions", {"revision_openreview_id": "xujj_test"})
# print(df)
# database.delete_node_by_id("revisions", {"revision_openreview_id": "xujj_test"})
# df = database.get_node_features_by_id("revisions", {"revision_openreview_id": "xujj_test"})
# print(df)

# review
# review_csv_path = "./csv_review_example.csv"
# database.construct_review_table_by_csv(review_csv_path)
# df = database.get_node_features_by_id("reviews", {"review_openreview_id": "xujj_test"})
# print(df)
# database.delete_node_by_id("reviews", {"review_openreview_id": "xujj_test"})
# df = database.get_node_features_by_id("reviews", {"review_openreview_id": "xujj_test"})
# print(df)

# author
# author_csv_path = "./csv_author_example.csv"
# database.construct_author_table_by_csv(author_csv_path)
# df = database.get_node_features_by_id("authors", {"author_openreview_id": "xujj_test"})
# print(df)
# database.delete_node_by_id("authors", {"author_openreview_id": "xujj_test"})
# df = database.get_node_features_by_id("authors", {"author_openreview_id": "xujj_test"})
# print(df)