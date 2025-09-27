import psycopg2
import json
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_arxiv_ids(file_path: str) -> List[str]:
    """Load arXiv IDs from JSON file."""
    try:
        with open(file_path, 'r') as f:
            json_file = json.load(f)
            return list(json_file.keys()) if isinstance(json_file, dict) else json_file
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {file_path}")
        raise

def get_database_connection():
    """Create database connection with error handling."""
    try:
        conn = psycopg2.connect(
            host="localhost",
            port="5433",
            dbname="postgres",
            user="cl195"
        )
        conn.autocommit = True
        return conn
    except psycopg2.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise

def find_existing_ids(cursor, arxiv_ids: List[str]) -> List[str]:
    """Find which arXiv IDs exist in the database."""
    if not arxiv_ids:
        return []
    
    # statement = "SELECT arxiv_id FROM papers WHERE arxiv_id = ANY(%s)"
    statement = "SELECT DISTINCT paper_arxiv_id FROM paragraphs WHERE paper_arxiv_id = ANY(%s)"
    cursor.execute(statement, (arxiv_ids,))
    return list(set([row[0] for row in cursor.fetchall()]))

def find_missing_ids(cursor, arxiv_ids: List[str]) -> List[str]:
    """Find which arXiv IDs are missing from the database."""
    if not arxiv_ids:
        return []
    
    # More efficient approach using a single query
    statement = """
        SELECT DISTINCT id
        FROM unnest(%s::text[]) AS id
        WHERE id NOT IN (
            SELECT paper_arxiv_id
            FROM paragraphs
        );
    """
    cursor.execute(statement, (arxiv_ids, ))
    return list(set([row[0] for row in cursor.fetchall()]))
    
def save_ids_to_json(ids: List[str], file_path: str, original_data: Dict[str, Any] = None):
    """Save IDs to JSON file, preserving original structure if needed."""
    try:
        if original_data and isinstance(original_data, dict):
            # If original was a dict, filter it to only include the specified IDs
            filtered_data = {id_: original_data[id_] for id_ in ids if id_ in original_data}
        else:
            # If original was a list, just save the IDs
            filtered_data = ids
        
        with open(file_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        logger.info(f"Saved {len(ids)} IDs to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save to {file_path}: {e}")
        raise

def main():
    # File paths
    dest_file_existing = "./json/arxiv_old_in2024.json"
    dest_file_missing = "./json/arxiv_old_not_in2024.json"
    
    # Load original data for structure preservation
    arxiv_ids = []
    source_file = "./json/arxiv_2024.json"
    # for i in range(1,11,1):
    # source_file = f"./json/arxiv_old_not_in_{i}.json"
    with open(source_file, 'r') as f:
        original_data = json.load(f)

    # Extract IDs
    arxiv_ids.extend(load_arxiv_ids(source_file))
    logger.info(f"Loaded {len(arxiv_ids)} arXiv IDs from {source_file}")
        
    # Database operations
    conn = get_database_connection()
    try:
        cur = conn.cursor()
        
        # Find existing IDs
        existing_ids = find_existing_ids(cur, arxiv_ids)
        logger.info(f"Found {len(existing_ids)} IDs in database")
        
        # Find missing IDs
        missing_ids = find_missing_ids(cur, arxiv_ids)
        logger.info(f"Found {len(missing_ids)} IDs not in database")
        
        # Verify counts match
        if len(existing_ids) + len(missing_ids) != len(arxiv_ids):
            logger.warning("ID counts don't match! There might be duplicates or other issues.")
        
        # Save results
        save_ids_to_json(existing_ids, dest_file_existing, original_data)
        save_ids_to_json(missing_ids, dest_file_missing, original_data)
        
        # Print summary
        print(f"\nSummary:")
        print(f"Total IDs processed: {len(arxiv_ids)}")
        print(f"IDs in database: {len(existing_ids)} (saved to {dest_file_existing})")
        print(f"IDs not in database: {len(missing_ids)} (saved to {dest_file_missing})")
        
        # Print first few of each for verification
        if existing_ids:
            print(f"\nFirst few existing IDs: {existing_ids[:5]}")
        if missing_ids:
            print(f"First few missing IDs: {missing_ids[:5]}")
            
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()