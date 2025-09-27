import sys
import os
import argparse
import json
from pathlib import Path

# Add parent directory to path (consider using proper package installation instead)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphserializer.database_serializer import DatabaseSerializer

def main():
    try:
        # Use pathlib for better path handling
        file_path = Path("./json/arxiv_2024.json")
        
        # Check if file exists
        if not file_path.exists():
            print(f"Error: File {file_path} not found")
            return
        
        # Load arxiv IDs
        with open(file_path, "r") as f:
            json_file = json.load(f)
        
        arxiv_ids = list(json_file)
        
        if not arxiv_ids:
            print("Warning: No arxiv IDs found in the JSON file")
            return
        
        # Ensure output directory exists
        output_dir = Path("./csv")
        output_dir.mkdir(exist_ok=True)
        
        # Execute database query
        ds = DatabaseSerializer()
        statement = "SELECT * FROM figures WHERE paper_arxiv_id = ANY(%s)"
        
        ds.query_to_csv_file(
            output_path="./csv/old_iclr/figures_2024.csv",
            query=statement,
            parameters=(arxiv_ids,)
        )
        
        print(f"Successfully exported {len(arxiv_ids)} arxiv IDs to CSV")
        
    except FileNotFoundError as e:
        print(f"File error: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
if __name__ == "__main__":
    main()