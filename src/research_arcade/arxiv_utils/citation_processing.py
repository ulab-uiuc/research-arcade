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


def citation_matching_csv(
    df_cit,
    arxiv_ids, 
    csv_path,
    similarity_threshold=95
):
    """
    Match citations from Semantic Scholar with bibliography entries from CSV.
    Updates the original CSV file with matched arxiv IDs.
    
    Args:
        df_cit: DataFrame with citations from paper_citation_crawling()
        arxiv_ids: List of arxiv IDs to filter on
        csv_path: Path to the CSV file containing bibliography data
        similarity_threshold: Minimum fuzzy match score (0-100)
    
    Returns:
        DataFrame with matched citations
    """
    # Load bibkey from the CSV
    df_bib = pd.read_csv(csv_path)
    
    # Filter to only the arxiv_ids we care about AND where cited_arxiv_id is null
    mask = df_bib['citing_arxiv_id'].isin(arxiv_ids) & df_bib['cited_arxiv_id'].isna()
    df_bib_filtered = df_bib[mask].copy()
    
    df_bib_filtered["norm_bib_title"] = df_bib_filtered["bib_title"].apply(normalize_title)
    
    # Group bib titles by citing_arxiv_id, also track original index for updating
    bib_groups = defaultdict(list)
    for idx, row in df_bib_filtered.iterrows():
        bib_groups[str(row["citing_arxiv_id"])].append({
            'original_idx': idx,
            'bib_title': row["bib_title"],
            'norm_bib_title': row["norm_bib_title"]
        })
    
    # Ensure required columns exist in citation df
    required_cols = ["citing_arxiv_id", "cited_arxiv_id", "cited_paper_name"]
    for col in required_cols:
        if col not in df_cit.columns:
            raise ValueError(f"Missing required column: {col}")

    results = []
    updates = []  # Track updates to make to original CSV
    total = 0
    matched = 0
    no_match = 0

    print(f"Processing {df_cit['citing_arxiv_id'].nunique()} unique citing papers...")
    
    for citing_id, group in df_cit.groupby("citing_arxiv_id"):
        citing_id = str(citing_id)
        if citing_id not in bib_groups:
            continue

        bib_entries = bib_groups[citing_id]
        # Map normalized title -> (original_idx, original_title)
        norm_title_map = {
            entry['norm_bib_title']: (entry['original_idx'], entry['bib_title']) 
            for entry in bib_entries
        }
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
                original_idx, matched_title = norm_title_map[norm_match]
                matched += 1
                
                results.append({
                    "citing_arxiv_id": citing_id,
                    "cited_arxiv_id": row["cited_arxiv_id"],
                    "cited_paper_name": row["cited_paper_name"],
                    "matched_bib_title": matched_title,
                    "match_score": score
                })
                
                # Track update for original CSV
                updates.append({
                    'original_idx': original_idx,
                    'cited_arxiv_id': row["cited_arxiv_id"]
                })
            else:
                no_match += 1

    # Apply updates to original DataFrame and save back to CSV
    for update in updates:
        df_bib.at[update['original_idx'], 'cited_arxiv_id'] = update['cited_arxiv_id']
    
    df_bib.to_csv(csv_path, index=False)
    print(f"\nUpdated original CSV: {csv_path}")

    print("\n" + "=" * 50)
    print("Citation Matching Statistics")
    print(f"Total citations processed: {total}")
    print(f"Matched: {matched}")
    print(f"No match: {no_match}")
    print(f"Updates applied to database: {len(updates)}")
    print("=" * 50)




def citation_matching_sql(
    df_cit,
    arxiv_ids,
    db_config,
    similarity_threshold=95
):
    
    import psycopg2
    from psycopg2.extras import execute_batch
    """
    Match citations from Semantic Scholar with bibliography entries from PostgreSQL.
    Updates the original database with matched arxiv IDs.
    
    Args:
        df_cit: DataFrame with citations from paper_citation_crawling()
        arxiv_ids: List of arxiv IDs to filter on
        db_config: Dict with keys: host, port, dbname, user, password
        similarity_threshold: Minimum fuzzy match score (0-100)
    
    Returns:
        DataFrame with matched citations
    """
    conn = psycopg2.connect(**db_config)
    
    try:
        # Query bibliography data from database (include primary key for updates)
        query = """
            SELECT id, citing_arxiv_id, bib_title
            FROM arxiv_citations
            WHERE citing_arxiv_id = ANY(%s)
            AND cited_arxiv_id IS NULL
        """
        df_bib = pd.read_sql(query, conn, params=(list(arxiv_ids),))
    except Exception as e:
        conn.close()
        raise e
    
    df_bib["norm_bib_title"] = df_bib["bib_title"].apply(normalize_title)

    # Group bib titles by citing_arxiv_id, also track row id for updating
    bib_groups = defaultdict(list)
    for _, row in df_bib.iterrows():
        bib_groups[str(row["citing_arxiv_id"])].append({
            'id': row["id"],
            'bib_title': row["bib_title"],
            'norm_bib_title': row["norm_bib_title"]
        })

    # Ensure required columns exist
    required_cols = ["citing_arxiv_id", "cited_arxiv_id", "cited_paper_name"]
    for col in required_cols:
        if col not in df_cit.columns:
            raise ValueError(f"Missing required column: {col}")

    results = []
    updates = []  # Track updates to make to database
    total = 0
    matched = 0
    no_match = 0

    print(f"Processing {df_cit['citing_arxiv_id'].nunique()} unique citing papers...")
    
    for citing_id, group in df_cit.groupby("citing_arxiv_id"):
        citing_id = str(citing_id)
        if citing_id not in bib_groups:
            continue

        bib_entries = bib_groups[citing_id]
        # Map normalized title -> (row_id, original_title)
        norm_title_map = {
            entry['norm_bib_title']: (entry['id'], entry['bib_title']) 
            for entry in bib_entries
        }
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
                row_id, matched_title = norm_title_map[norm_match]
                matched += 1
                
                results.append({
                    "citing_arxiv_id": citing_id,
                    "cited_arxiv_id": row["cited_arxiv_id"],
                    "cited_paper_name": row["cited_paper_name"],
                    "matched_bib_title": matched_title,
                    "match_score": score
                })
                
                # Track update for database
                updates.append((row["cited_arxiv_id"], row_id))
            else:
                no_match += 1

    # Apply updates to database
    if updates:
        try:
            cursor = conn.cursor()
            update_query = """
                UPDATE arxiv_citations 
                SET cited_arxiv_id = %s 
                WHERE id = %s
            """
            execute_batch(cursor, update_query, updates)
            conn.commit()
            print(f"\nUpdated {len(updates)} rows in database")
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
    
    conn.close()


    print("\n" + "=" * 50)
    print("Citation Matching Statistics")
    print(f"Total citations processed: {total}")
    print(f"Matched: {matched}")
    print(f"No match: {no_match}")
    print(f"Updates applied to database: {len(updates)}")
    print("=" * 50)
