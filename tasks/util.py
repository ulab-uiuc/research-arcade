import psycopg2

def select_papers_with_criteria(
    min_paragraph=10, min_figure=0, min_citation=0, min_table=0
):
    """
    Return arxiv_ids from papers that meet all provided minimum counts.
    Assumes child tables link to papers via columns named *_arxiv_id.
    Adjust column/table names if yours differ.
    """

    conn = psycopg2.connect(
        host="localhost", dbname="postgres", user="postgres",
        password="Lcs20031121!", port="5432"
    )
    try:
        conditions = []
        params = []

        if min_paragraph > 0:
            conditions.append("""
                (SELECT COUNT(*)
                FROM paragraphs pr
                WHERE pr.paper_arxiv_id = p.arxiv_id) >= %s
            """)
            params.append(min_paragraph)

        if min_figure > 0:
            conditions.append("""
                (SELECT COUNT(*)
                FROM figures fg
                WHERE fg.paper_arxiv_id = p.arxiv_id) >= %s
            """)
            params.append(min_figure)

        if min_citation > 0:
            conditions.append("""
                (SELECT COUNT(*)
                FROM citations ct
                WHERE ct.citing_arxiv_id = p.arxiv_id) >= %s
            """)
            params.append(min_citation)

        if min_table > 0:
            conditions.append("""
                (SELECT COUNT(*)
                FROM tables tb
                WHERE tb.paper_arxiv_id = p.arxiv_id) >= %s
            """)
            params.append(min_table)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql = f"""
            SELECT p.arxiv_id
              FROM papers AS p
              {where_clause}
        """

        with conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()
