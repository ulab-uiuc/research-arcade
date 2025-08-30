import sys
import os
import ast
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from openreview_database import Database
import json

database = Database()

# paper
# paper_json_path = "./json_paper_example.json"
# database.construct_paper_table_by_json(paper_json_path, with_pdf=False)
# df = database.get_node_features_by_id("papers", {"paper_openreview_id": "xujj_test"})
# print(df)
# database.delete_node_by_id("papers", {"paper_openreview_id": "xujj_test"})
# df = database.get_node_features_by_id("papers", {"paper_openreview_id": "xujj_test"})
# print(df)

# revision
# revision_json_path = "./json_revision_example.json"
# database.construct_revision_table_by_json(revision_json_path, with_pdf=False)
# df = database.get_node_features_by_id("revisions", {"revision_openreview_id": "xujj_test"})
# print(df)
# database.delete_node_by_id("revisions", {"revision_openreview_id": "xujj_test"})
# df = database.get_node_features_by_id("revisions", {"revision_openreview_id": "xujj_test"})
# print(df)

# review
# review_json_path = "./json_review_example.json"
# database.construct_review_table_by_json(review_json_path)
# df = database.get_node_features_by_id("reviews", {"review_openreview_id": "xujj_test"})
# print(df)
# database.delete_node_by_id("reviews", {"review_openreview_id": "xujj_test"})
# df = database.get_node_features_by_id("reviews", {"review_openreview_id": "xujj_test"})
# print(df)

# author
# author_json_path = "./json_author_example.json"
# database.construct_author_table_by_json(author_json_path)
# df = database.get_node_features_by_id("authors", {"author_openreview_id": "xujj_test"})
# print(df)
# database.delete_node_by_id("authors", {"author_openreview_id": "xujj_test"})
# df = database.get_node_features_by_id("authors", {"author_openreview_id": "xujj_test"})
# print(df)