# Given a section of a paper, we first obtain all the paragraphs, then find the sections of the papers


import psycopg2
import json


def section_data_generator(pairs):

    # pairs = (arxiv_ids, section_names)

    conn = psycopg2.connect(
        host="localhost",
        port="5433",
        dbname="postgres",
        user="cl195"
    )
    conn.autocommit = True
    cur = conn.cursor()

    for arxiv_id, section_name in pairs:

        statement = """
        SELECT * FROM paragraphs WHERE paper_arxiv_id = %s AND paper_section = %s
        """

        cur.execute(statement, (arxiv_id, section_name,))

        result = cur.fetchall()

        for paragraph in result:
            # What do we need to fetch?
            # Everything like before, the only exception is that we need to retrieve the paragraphs.
            # I guess that we cannot just use the paragraph.
            # 

            # Store everything as json object.





