import sys
from pathlib import Path
# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from research_arcade import ResearchArcade 

db_type = "sql"
config = {
    "host": "localhost",
    "dbname": "iclr_openreview_database",
    "user": "jingjunx",
    "password": "",
    "port": "5432"
}

research_arcade = ResearchArcade(db_type=db_type, config=config)

########## openreview_authors ##########
# get_all_node_features
# openreview_authors_df = research_arcade.get_all_node_features("openreview_authors")
# openreview_authors_df.to_csv("/data/jingjunx/my_research_arcade_data/openreview_authors.csv", index=False)
# print(len(openreview_authors_df))

########## openreview_papers ##########
# get_all_node_features
# openreview_papers_df = research_arcade.get_all_node_features("openreview_papers")
# openreview_papers_df.to_csv("/data/jingjunx/my_research_arcade_data/openreview_papers.csv", index=False)
# print(len(openreview_papers_df))

########## openreview_reviews ##########
# get_all_node_features
# openreview_reviews_df = research_arcade.get_all_node_features("openreview_reviews")
# openreview_reviews_df.to_csv("/data/jingjunx/my_research_arcade_data/openreview_reviews.csv", index=False)
# print(len(openreview_reviews_df))

########## openreview_revisions ##########
# get_all_node_features
# openreview_revisions_df = research_arcade.get_all_node_features("openreview_revisions")
# openreview_revisions_df.to_csv("/data/jingjunx/my_research_arcade_data/openreview_revisions.csv", index=False)
# print(len(openreview_revisions_df))

########## openreview_paragraphs ##########
# get_all_node_features
# openreview_paragraphs_df = research_arcade.get_all_node_features("openreview_paragraphs")
# openreview_paragraphs_df.to_csv("/data/jingjunx/my_research_arcade_data/openreview_paragraphs.csv", index=False)
# print(len(openreview_paragraphs_df))

########## openreview_arxiv ##########
# get_all_edge_features
# openreview_arxiv_df = research_arcade.get_all_edge_features("openreview_arxiv")
# openreview_arxiv_df.to_csv("/data/jingjunx/my_research_arcade_data/openreview_arxiv.csv", index=False)
# print(len(openreview_arxiv_df))

########## openreview_papers_authors ##########
# get_all_edge_features
# openreview_papers_authors = research_arcade.get_all_edge_features("openreview_papers_authors")
# openreview_papers_authors.to_csv("/data/jingjunx/my_research_arcade_data/openreview_papers_authors.csv", index=False)
# print(len(openreview_papers_authors))

########## openreview_papers_reviews ##########
# get_all_edge_features
# openreview_papers_reviews = research_arcade.get_all_edge_features("openreview_papers_reviews")
# openreview_papers_reviews.to_csv("/data/jingjunx/my_research_arcade_data/openreview_papers_reviews.csv", index=False)
# print(len(openreview_papers_reviews))

########## openreview_papers_revisions ##########
# get_all_edge_features
# openreview_papers_revisions = research_arcade.get_all_edge_features("openreview_papers_revisions")
# openreview_papers_revisions.to_csv("/data/jingjunx/my_research_arcade_data/openreview_papers_revisions.csv", index=False)
# print(len(openreview_papers_revisions))

########## openreview_revisions_reviews ##########
# get_all_edge_features
# openreview_revisions_reviews = research_arcade.get_all_edge_features("openreview_revisions_reviews")
# openreview_revisions_reviews.to_csv("/data/jingjunx/my_research_arcade_data/openreview_revisions_reviews.csv", index=False)
# print(len(openreview_revisions_reviews))