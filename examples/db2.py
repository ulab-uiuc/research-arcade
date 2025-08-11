import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import psycopg2


conn = psycopg2.connect(
    host="localhost",
    port="5433",
    dbname="postgres",
    user="cl195"
)
# Enable autocommit
conn.autocommit = True
cur = conn.cursor()

cur.execute("""
SELECT * FROM paper_task WHERE citation = true;
""")

res = cur.fetchall()
print("fetching result of paper_task")
print(res)
print(len(res))


cur.execute("""
SELECT * FROM papers;
""")

res = cur.fetchall()
print("fetching result of papers")
print(res)
print(len(res))
