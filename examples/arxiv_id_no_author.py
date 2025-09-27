import psycopg2
import json
import os

def fetch_and_chunk_arxiv_ids():
    # Create destination directory if it doesn't exist
    dest_dir = "./json/"
    os.makedirs(dest_dir, exist_ok=True)
    
    # Database connection
    conn = psycopg2.connect(
        host="localhost",
        port="5433",
        dbname="postgres",
        user="cl195"
    )
    conn.autocommit = True
    cur = conn.cursor()  # Fixed: removed 'self.'
    
    try:
        # Execute query
        statement = "SELECT base_arxiv_id FROM papers;"
        cur.execute(statement)
        result = cur.fetchall()
        
        # Convert tuples to a flat list of arxiv_ids
        arxiv_ids = [row[0] for row in result]
        result_len = len(arxiv_ids)
        
        print(f"Total papers found: {result_len}")
        
        # Calculate chunk size
        chunk_size = result_len // 5  # 5 chunks
        remainder = result_len % 5    # Handle remainder
        
        # Create 5 chunks
        for i in range(5):
            # Calculate start and end indices for each chunk
            start_idx = i * chunk_size
            
            # For the last chunk, include any remaining items
            if i == 4:  # Last chunk
                end_idx = result_len
            else:
                end_idx = (i + 1) * chunk_size
            
            # Extract chunk
            chunk = arxiv_ids[start_idx:end_idx]
            
            # Define destination path
            dest_path = f"{dest_dir}arxiv_id_author_{i}.json"
            # Write chunk to JSON file
            with open(dest_path, 'w') as f:  # Fixed: added 'as f'
                json.dump(chunk, f, indent=2)
            
            print(f"Chunk {i+1}: {len(chunk)} items saved to {dest_path}")
    
    except Exception as e:
        print(f"Error occurred: {e}")
    
    finally:
        # Close database connection
        cur.close()
        conn.close()
        print("Database connection closed.")

# Run the function
if __name__ == "__main__":
    fetch_and_chunk_arxiv_ids()