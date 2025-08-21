from semanticscholar import SemanticScholar
import os, re

sch = SemanticScholar(api_key=os.getenv("S2_API_KEY"))

def normalize_arxiv(aid: str) -> str:
    core = re.sub(r'(?i)^arxiv:', '', aid)
    core = re.sub(r'v\d+$', '', core)
    return f"ARXIV:{core}"

def cited_abstracts(arxiv_id: str, limit: int = 100):
    pid = normalize_arxiv(arxiv_id)

    # just plain fields — no "paper." prefix
    refs = sch.get_paper_references(
        pid, fields=["paperId", "title", "abstract", "year", "externalIds"], limit=limit
    )

    results = []
    for ref in refs:
        p = ref.paper  # this is already a Paper object
        results.append({
            "paperId": p.paperId,
            "title": p.title,
            "year": p.year,
            "abstract": p.abstract,
            "externalIds": p.externalIds,
        })
    return results


def citation_processing(arxiv_ids: str, limit: int = 100):

    # First connect to database
    conn = psycopg2.connect(
            host="localhost",
            port="5433",
            dbname="postgres",
            user="cl195"
    )
    conn.autocommit = True
    cur = self.conn.cursor()
    sql = """
    INSERT INTO citation_sch (arxiv_id, title, abstract)
    VALUES (%s, %s, %s)
    RETURNING id
    """
    
    for arxiv_id in arxiv_ids:
        data = cited_abstracts(arxiv_id, limit=limit)

        for datum in data:
            title = datum["title"]
            abstract datum["abstract"]
            cur.execute(sql, (arxiv_id, title, abstract))
    
        # Inser the stuff into database
# # Example
# data = cited_abstracts("1706.03762")
# print(len(data), "references")

# for datum in data:

#     print(datum["title"])
#     print(datum["abstract"])


