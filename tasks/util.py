import psycopg2
def select_papers_with_criteria(min_paragraph = 1, min_figure = 0, min_citation = 0, min_table = 0):
    """
    Select paper with at least one 
    """

    conn = psycopg2.connect(
            host="localhost",
            port="5433",
            dbname="postgres",
            user="cl195"
        )
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("""
    SELECT p.*
    FROM paper AS p
    JOIN (
    SELECT arxiv_id, COUNT(*) AS para_cnt
    FROM paragraphs
    GROUP BY arxiv_id
    HAVING COUNT(*) >= %s
    ) AS par ON par.arxiv_id = p.arxiv_id;
    """, (min_paragraph,))

    return cur.fetchall()
