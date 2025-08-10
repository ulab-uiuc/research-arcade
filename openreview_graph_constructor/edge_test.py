from sqlDatabaseConstructor import sqlDatabaseConstructor
from database import Database

db = Database()
sql_database = sqlDatabaseConstructor()

############ get single edge features ############
# papers_authors table
'''
edge = sql_database.get_edge_features_by_id("papers_authors", {"paper_openreview_id": "00SnKBGTsz", "author_openreview_id": "~Elias_Stengel-Eskin1"})
print(edge)
'''

# papers_reviews table
'''
edge = sql_database.get_edge_features_by_id("papers_reviews", {"paper_openreview_id": "00SnKBGTsz", "review_openreview_id": "13mj0Rtn5W"})
print(edge)
'''

# papers_revisions table
'''
edge = sql_database.get_edge_features_by_id("papers_revisions", {"paper_openreview_id": "00SnKBGTsz", "revision_openreview_id": "2hQgqdDwZq"})
print(edge)
'''

############ get all edges from tables ############
# papers_authors table
'''
df = sql_database.get_all_edge_features("papers_authors")
print(len(df))
print(df.head())
'''

# papers_reviews table
'''
df = sql_database.get_all_edge_features("papers_reviews")
print(len(df))
print(df.head())
'''

# papers_revisions table
'''
df = sql_database.get_all_edge_features("papers_revisions")
print(len(df))
print(df.head())
'''

############ delete a single edge and insert it back ############
# papers_authors table
'''
original_edge_features = sql_database.delete_edge_by_id("papers_authors", {"paper_openreview_id": "zqzsZ5cXbB", "author_openreview_id": "~Zhi_Zhang4"})
edge = sql_database.get_edge_features_by_id("papers_authors", {"paper_openreview_id": "zqzsZ5cXbB", "author_openreview_id": "~Zhi_Zhang4"})
print(edge)

sql_database.insert_edge("papers_authors", edge_features=original_edge_features)
edge = sql_database.get_edge_features_by_id("papers_authors", {"paper_openreview_id": "zqzsZ5cXbB", "author_openreview_id": "~Zhi_Zhang4"})
print(edge)
'''

# papers_reviews table
'''
original_edge_features = sql_database.delete_edge_by_id("papers_reviews", {"paper_openreview_id": "zxbQLztmwb", "review_openreview_id": "9DZHHl3PgM"})
edge = sql_database.get_edge_features_by_id("papers_reviews", {"paper_openreview_id": "zxbQLztmwb", "review_openreview_id": "9DZHHl3PgM"})
print(edge)

sql_database.insert_edge("papers_reviews", edge_features=original_edge_features)
edge = sql_database.get_edge_features_by_id("papers_reviews", {"paper_openreview_id": "zxbQLztmwb", "review_openreview_id": "9DZHHl3PgM"})
print(edge)
'''

# papers_revisions table
'''
original_edge_features = sql_database.delete_edge_by_id("papers_revisions", {"paper_openreview_id": "zmmfsJpYcq", "revision_openreview_id": "GSKUBuUoMz"})
edge = sql_database.get_edge_features_by_id("papers_revisions", {"paper_openreview_id": "zmmfsJpYcq", "revision_openreview_id": "GSKUBuUoMz"})
print(edge)

sql_database.insert_edge("papers_revisions", edge_features=original_edge_features)
edge = sql_database.get_edge_features_by_id("papers_revisions", {"paper_openreview_id": "zmmfsJpYcq", "revision_openreview_id": "GSKUBuUoMz"})
print(edge)
'''

############ Find the neighborhoods ############
# papers_authors table
# paper neighboring authors
'''
df = sql_database.get_neighborhood_by_id("papers_authors", {"paper_openreview_id": "00SnKBGTsz"})
print(len(df))
print(df.head())
'''
# author neighboring papers
'''
df = sql_database.get_neighborhood_by_id("papers_authors", {"author_openreview_id": "~Elias_Stengel-Eskin1"})
print(len(df))
print(df.head())
'''

# papers_reviews table
# paper neighboring reviews
'''
df = sql_database.get_neighborhood_by_id("papers_reviews", {"paper_openreview_id": "00SnKBGTsz"})
print(len(df))
print(df.head())
'''
# review neighboring papers
'''
df = sql_database.get_neighborhood_by_id("papers_reviews", {"review_openreview_id": "13mj0Rtn5W"})
print(len(df))
print(df.head())
'''

# papers_revisions table
# paper neighboring revisions
'''
df = sql_database.get_neighborhood_by_id("papers_revisions", {"paper_openreview_id": "00SnKBGTsz"})
print(len(df))
print(df.head())
'''
# revision neighboring papers
'''
df = sql_database.get_neighborhood_by_id("papers_revisions", {"revision_openreview_id": "2hQgqdDwZq"})
print(len(df))
print(df.head())
'''