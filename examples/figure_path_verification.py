import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_constructor.utils import figure_latex_path_to_path

from graph_constructor.database import Database

download_path = "download"

# Connect to db 

db = Database()

# Select all the rows, from which we select the paper_arxiv_id and path entries

sql = """
SELECT paper_arxiv_id, path FROM figures
"""

db.cur.execute(sql)
rows = db.cur.fetchall()

all_valid = True

for paper_arxiv_id, latex_path in rows:
    fs_path = figure_latex_path_to_path(
        path=download_path,
        arxiv_id=paper_arxiv_id,
        latex_path=latex_path
    )
    if not os.path.isfile(fs_path):
        all_valid = False
        print(f"Invalid path: {fs_path}")
        print(f"Latex path: {latex_path}")
        print(f"Arxiv id: {paper_arxiv_id}")


if all_valid:
    print("All paths in db are valid under conversion")







# path = figure_latex_path_to_path(path="download", arxiv_id="2412.17767v2", latex_path="./figs/community_activity.pdf")

# print(path)

# # Verify if the file of the path exists
# if os.path.isfile(path):
#     print("File exists")
# else:
#     print("File does not exist")