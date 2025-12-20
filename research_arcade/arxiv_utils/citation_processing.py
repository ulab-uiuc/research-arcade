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

import psycopg2
from psycopg2.extras import execute_values

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


def citation_matching_csv(
    df_cit,
    arxiv_ids, 
    csv_path,
    output_path=None,
    similarity_threshold=95
):
    """
    Match citations from Semantic Scholar with bibliography entries from CSV.
    
    Args:
        df_cit: DataFrame with citations from paper_citation_crawling()
        arxiv_ids: List of arxiv IDs to filter on
        csv_path: Path to the CSV file containing bibliography data
        output_path: Optional path to save results
        similarity_threshold: Minimum fuzzy match score (0-100)
    
    Returns:
        DataFrame with matched citations
    """
    # Load bibkey from the CSV
    df_bib = pd.read_csv(csv_path)
    
    # Filter to only the arxiv_ids we care about
    df_bib = df_bib[df_bib['citing_arxiv_id'].isin(arxiv_ids)]
    
    df_bib["norm_bib_title"] = df_bib["bib_title"].apply(normalize_title)

    # Group bib titles by citing_arxiv_id
    bib_groups = defaultdict(list)
    for _, row in df_bib.iterrows():
        bib_groups[str(row["citing_arxiv_id"])].append(
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
            continue

        bib_titles = bib_groups[citing_id]
        norm_title_map = {norm: orig for (orig, norm) in bib_titles}
        norm_title_list = list(norm_title_map.keys())

        for _, row in group.iterrows():
            total += 1
            cited_name = normalize_title(row["cited_paper_name"])

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
                
                results.append({
                    "citing_arxiv_id": citing_id,
                    "cited_arxiv_id": row["cited_arxiv_id"],
                    "cited_paper_name": row["cited_paper_name"],
                    "matched_bib_title": matched_title,
                    "match_score": score
                })
            else:
                no_match += 1

    df_out = pd.DataFrame(results)
    
    # Save if output path provided
    if output_path:
        df_out.to_csv(output_path, index=False)
        print(f"\nSaved result CSV to: {output_path}")

    print("\n" + "=" * 50)
    print("Citation Matching Statistics")
    print(f"Total citations processed: {total}")
    print(f"Matched: {matched}")
    print(f"No match: {no_match}")
    print("=" * 50)

    return df_out


def citation_matching_sql(
    df_cit,
    arxiv_ids,
    db_config,
    output_path=None,
    similarity_threshold=95
):
    """
    Match citations from Semantic Scholar with bibliography entries from PostgreSQL.
    
    Args:
        df_cit: DataFrame with citations from paper_citation_crawling()
        arxiv_ids: List of arxiv IDs to filter on
        db_config: Dict with keys: host, port, dbname, user, password
        output_path: Optional path to save results
        similarity_threshold: Minimum fuzzy match score (0-100)
    
    Returns:
        DataFrame with matched citations
    """
    conn = psycopg2.connect(**db_config)
    
    try:
        # Query bibliography data from database
        query = """
            SELECT citing_arxiv_id, bib_title
            FROM arxiv_citations
            WHERE citing_arxiv_id = ANY(%s)
        """
        df_bib = pd.read_sql(query, conn, params=(arxiv_ids,))
    finally:
        conn.close()
    
    df_bib["norm_bib_title"] = df_bib["bib_title"].apply(normalize_title)

    # Group bib titles by citing_arxiv_id
    bib_groups = defaultdict(list)
    for _, row in df_bib.iterrows():
        bib_groups[str(row["citing_arxiv_id"])].append(
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
            continue

        bib_titles = bib_groups[citing_id]
        norm_title_map = {norm: orig for (orig, norm) in bib_titles}
        norm_title_list = list(norm_title_map.keys())

        for _, row in group.iterrows():
            total += 1
            cited_name = normalize_title(row["cited_paper_name"])

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
                
                results.append({
                    "citing_arxiv_id": citing_id,
                    "cited_arxiv_id": row["cited_arxiv_id"],
                    "cited_paper_name": row["cited_paper_name"],
                    "matched_bib_title": matched_title,
                    "match_score": score
                })
            else:
                no_match += 1

    df_out = pd.DataFrame(results)
    
    if output_path:
        df_out.to_csv(output_path, index=False)
        print(f"\nSaved result CSV to: {output_path}")

    print("\n" + "=" * 50)
    print("Citation Matching Statistics")
    print(f"Total citations processed: {total}")
    print(f"Matched: {matched}")
    print(f"No match: {no_match}")
    print("=" * 50)

    return df_out