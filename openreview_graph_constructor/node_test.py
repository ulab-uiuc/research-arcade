from sqlDatabaseConstructor import sqlDatabaseConstructor
from database import Database
import pandas as pd

sql_database = sqlDatabaseConstructor()

############ get nodes by venue ############
# 2025
# venue = 'ICLR.cc/2025/Conference'
# papers
'''
results = sql_database.get_node_features_by_venue("papers", venue=venue)
print(len(results))
'''
# revisions
'''
results = sql_database.get_node_features_by_venue("revisions", venue=venue)
print(len(results))
'''

# 2024
venue = 'ICLR.cc/2024/Conference'
# papers
'''
results = sql_database.get_node_features_by_venue("papers", venue=venue)
print(len(results))
'''
# revisions
'''
results = sql_database.get_node_features_by_venue("revisions", venue=venue)
print(len(results))
'''

############ get single node features ############
# papers table
'''
node = sql_database.get_node_features_by_id("papers", {"paper_openreview_id": "00SnKBGTsz"})
print(node)
'''

# reviews table
'''
node = sql_database.get_node_features_by_id("reviews", {"review_openreview_id": "Is5Qh2Gs5x"})
print(node)
'''

# authors table
'''
node = sql_database.get_node_features_by_id("authors", {"author_openreview_id": "~A-Long_Jin1"})
print(node)
'''

# revisions table
'''
node = sql_database.get_node_features_by_id("revisions", {"modified_openreview_id": "AhzvtTGKFO"})
print(node)
'''

############ get all nodes without features from tables ############
# papers table
'''
df = sql_database.get_all_nodes("papers")
print(len(df))
print(df.iloc[0])
'''

# reviews table
'''
df = sql_database.get_all_nodes("reviews")
print(len(df))
print(df.iloc[0])
'''

# authors table
'''
df = sql_database.get_all_nodes("authors")
print(len(df))
print(df.iloc[0])
'''

# revisions table
'''
df = sql_database.get_all_nodes("revisions")
print(len(df))
print(df.iloc[0])
'''

############ get all nodes with features from tables ############
# papers table
'''
df = sql_database.get_all_node_features("papers")
print(len(df))
print(df.iloc[0])
'''

# reviews table
'''
df = sql_database.get_all_node_features("reviews")
print(len(df))
print(df.iloc[0])
'''

# authors table
'''
df = sql_database.get_all_node_features("authors")
print(len(df))
print(df.iloc[0])
'''

# revisions table
'''
df = sql_database.get_all_node_features("revisions")
print(len(df))
print(df.iloc[0])
'''

############ delete a single node and insert it back ############
# papers table
'''
original_node_features = sql_database.delete_node_by_id("papers", {"paper_openreview_id": "zGej22CBnS"})
node = sql_database.get_node_features_by_id("papers", {"paper_openreview_id": "zGej22CBnS"})
print(node)

sql_database.insert_node("papers", node_features=original_node_features)
node = sql_database.get_node_features_by_id("papers", {"paper_openreview_id": "zGej22CBnS"})
print(node)
'''

# reviews table
'''
original_node_features = sql_database.delete_node_by_id("reviews", {"review_openreview_id": "DHwZxFryth"})
node = sql_database.get_node_features_by_id("reviews", {"review_openreview_id": "DHwZxFryth"})
print(node)

sql_database.insert_node("reviews", node_features=original_node_features)
node = sql_database.get_node_features_by_id("reviews", {"review_openreview_id": "DHwZxFryth"})
print(node)
'''

# revisions table
'''
original_node_features = sql_database.delete_node_by_id("revisions", {"modified_openreview_id": "yfHQOp5zWc"})
node = sql_database.get_node_features_by_id("revisions", {"modified_openreview_id": "yfHQOp5zWc"})
print(node)

sql_database.insert_node("revisions", node_features=original_node_features)
node = sql_database.get_node_features_by_id("revisions", {"modified_openreview_id": "yfHQOp5zWc"})
print(node)
'''

# authors table
'''
original_node_features = sql_database.delete_node_by_id("authors", {"author_openreview_id": "~ishmam_zabir1"})
node = sql_database.get_node_features_by_id("authors", {"author_openreview_id": "~ishmam_zabir1"})
print(node)

sql_database.insert_node("authors", node_features=original_node_features)
node = sql_database.get_node_features_by_id("authors", {"author_openreview_id": "~ishmam_zabir1"})
print(node)
'''

############ get the features from a node and update it back ############
# papers tables
'''
node = sql_database.get_node_features_by_id("papers", {"paper_openreview_id": "zGej22CBnS"})
print(node)

sql_database.update_node("papers", node_features=node)
node = sql_database.get_node_features_by_id("papers", {"paper_openreview_id": "zGej22CBnS"})
print(node)
'''

# reviews table
'''
node = sql_database.get_node_features_by_id("reviews", {"review_openreview_id": "DHwZxFryth"})
print(node)

sql_database.update_node("reviews", node_features=node)
node = sql_database.get_node_features_by_id("reviews", {"review_openreview_id": "DHwZxFryth"})
print(node)
'''

# revisions table
'''
node = sql_database.get_node_features_by_id("revisions", {"modified_openreview_id": "yfHQOp5zWc"})
print(node)

sql_database.update_node("revisions", node_features=node)
node = sql_database.get_node_features_by_id("revisions", {"modified_openreview_id": "yfHQOp5zWc"})
print(node)
'''

# authors table
'''
node = sql_database.get_node_features_by_id("authors", {"author_openreview_id": "~ishmam_zabir1"})
print(node)

sql_database.update_node("authors", node_features=node)
node = sql_database.get_node_features_by_id("authors", {"author_openreview_id": "~ishmam_zabir1"})
print(node)
'''