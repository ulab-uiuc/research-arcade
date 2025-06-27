import arxiv

# Metadata
bib_title = "VRT: A Video Restoration Transformer"
bib_author_lastname = "Liang"

# Step 1: Search using title keywords
query = "ti:VRT Video Restoration Transformer+AND+au:Liang"

print(f"Query: {query}")

search = arxiv.Search(
    query=query,
    max_results=20,
    sort_by=arxiv.SortCriterion.Relevance
)

# Step 2: Filter manually
for result in search.results():
    title_match = result.title.strip().lower() == bib_title.strip().lower()
    author_match = any(bib_author_lastname.lower() in author.name.lower() for author in result.authors)
    
    if title_match and author_match:
        print("Found the paper!")
        print("Title:", result.title)
        print("Authors:", ", ".join(author.name for author in result.authors))
        print("Published:", result.published)
        print("PDF URL:", result.pdf_url)
        print("arXiv ID:", result.entry_id)
        break
else:
    print("No exact match found.")
