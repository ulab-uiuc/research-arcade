"""
As we know, after obtaining the citations, there might be several issues
First, they don't have arxiv ids
Second, the cited paper may not be in the dataset
Methods below solve this issue
"""

from semanticscholar import SemanticScholar
import os
from dotenv import load_dotenv
import pandas as pd
import pandas as pd
from rapidfuzz import fuzz, process
from collections import defaultdict

load_dotenv()

def paper_citation_crawling(arxiv_ids):
    # Initialize the Semantic Scholar API client
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    sch = SemanticScholar(api_key=api_key)
    
    # Collect results in a list, then convert to DataFrame
    results = []
    
    # Process each arXiv ID
    for idx, arxiv_id in enumerate(arxiv_ids, 1):
        try:
            print(f"\n[{idx}/{len(arxiv_ids)}] Processing arXiv:{arxiv_id}")
            
            # Get the paper by arXiv ID with references
            paper_with_refs = sch.get_paper(
                f"arXiv:{arxiv_id}",
                fields=['title', 'references', 'references.title', 'references.externalIds']
            )
            
            if not paper_with_refs:
                print(f"  Paper not found")
                continue
            
            print(f"  Paper: {paper_with_refs.title}")
            print(f"  Total references: {len(paper_with_refs.references) if paper_with_refs.references else 0}")
            
            # Filter for references that are on arXiv
            if paper_with_refs.references:
                for ref in paper_with_refs.references:
                    if ref.externalIds and 'ArXiv' in ref.externalIds:
                        results.append({
                            'citing_arxiv_id': arxiv_id,
                            'cited_arxiv_id': ref.externalIds['ArXiv'],
                            'cited_paper_name': ref.title
                        })
            
            print(f"References on arXiv: {len([r for r in results if r['citing_arxiv_id'] == arxiv_id])}")
            
        except Exception as e:
            print(f"  Error processing {arxiv_id}: {str(e)}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(results, columns=['citing_arxiv_id', 'cited_arxiv_id', 'cited_paper_name'])
    return df


def normalize_title(title):
    """Normalize title for matching"""
    if not title:
        return ""
    return " ".join(title.lower().strip().split())

# CSV version of the citation
def citation_matching_csv(
    df_cit,
    arxiv_ids, 
    csv_dir,
    output_path,
    similarity_threshold=95
):
    # Load bibkey from the arxiv ids
    csv_path = f"{csv_dir}/arxiv_citations.csv"
    df_bib = pd.read_csv(csv_path)

    df_bib["norm_bib_title"] = df_bib["bib_title"].apply(normalize_title)

    # Group bib titles by citing_arxiv_id
    bib_groups = defaultdict(list)
    for _, row in df_bib.iterrows():
        bib_groups[row["citing_arxiv_id"]].append(
            (row["bib_title"], row["norm_bib_title"])
        )

    # Ensure required columns exist
    required_cols = ["citing_arxiv_id", "cited_arxiv_id", "cited_paper_name"]
    for col in required_cols:
        if col not in df_cit.columns:
            raise ValueError(f"Missing required column: {col}")

    results = []

    total = 0
    matched = 0
    no_match = 0

    print(f"Processing {df_cit['citing_arxiv_id'].nunique()} unique citing papers...")
    
    for citing_id, group in df_cit.groupby("citing_arxiv_id"):
        citing_id = str(citing_id)
        if citing_id not in bib_groups:            

            # No bib titles at all for this citing paper
            # for _, row in group.iterrows():
            #     results.append({
            #         "citing_arxiv_id": citing_id,
            #         "cited_arxiv_id": row["cited_arxiv_id"],
            #         "cited_paper_name": row["cited_paper_name"],
            #         "matched_bib_title": None,
            #         "match_score": None
            #     })
            continue

        bib_titles = bib_groups[citing_id]
        norm_title_map = {norm: orig for (orig, norm) in bib_titles}
        norm_title_list = list(norm_title_map.keys())

        for _, row in group.iterrows():
            total += 1
            cited_name = normalize_title(row["cited_paper_name"])

            # if not cited_name or not norm_title_list:
            #     no_match += 1
            #     results.append({
            #         "citing_arxiv_id": citing_id,
            #         "cited_arxiv_id": row["cited_arxiv_id"],
            #         "cited_paper_name": row["cited_paper_name"],
            #         "matched_bib_title": None,
            #         "match_score": None
            #     })
            #     continue

            # Fuzzy match
            best = process.extractOne(
                cited_name,
                norm_title_list,
                scorer=fuzz.ratio,
                score_cutoff=similarity_threshold
            )

            if best:
                norm_match, score, _ = best
                matched_title = norm_title_map[norm_match]
                matched += 1
            else:
                matched_title = None
                score = None
                no_match += 1
            if matched_title != "None" and matched_title != None:
                print(matched_title)
                print(type(matched_title))
                results.append({
                    "citing_arxiv_id": citing_id,
                    "cited_arxiv_id": row["cited_arxiv_id"],
                    "cited_paper_name": row["cited_paper_name"],
                    "matched_bib_title": matched_title,
                    "match_score": score
                })

    print(len(results))
    df_out = pd.DataFrame(results)
    print(f"\nSaved result CSV to: {output_path}")

    print("\n" + "="*50)
    print("Citation Matching Statistics")
    print(f"Total citations processed: {total}")

    return df_out