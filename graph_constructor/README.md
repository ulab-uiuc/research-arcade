This directory stores the hetero graph database of information extracted from paper, including paper, figures, author, and so on. The information is stored as nodes and connected with edges reflecting the corresponding relations, including citation authorship, etc.

Current nodes and their attributes:
1. papers: id, arxiv_id, title, abstract, submit_data, authors (which will be included in the edge tables?), metadata(url, version, categories, etc.)
2. authors: id, semantic scholar id, name, orcid, pubs? (to be included in the edge tables)
3. categories?
4. figures: id, paper_id (also in tables?), figure index/path, caption (text)
5. tables: id, paper_id (tables?), path?, caption
6. institution?



Current Edges/Relations:

1. paper_authors: paper id to author id, sequence
2. paper_category: paper id to category id
3. citations: paper id to paper id
4. paper figure: paper id to table/figure id
5. author affiliation: author id to affilitation id
