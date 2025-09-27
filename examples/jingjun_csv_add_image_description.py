import sys
import os
import argparse
import csv

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_constructor.utils import figure_latex_path_to_path
from tasks.paragraph_generation_local_vllm import visual_adaptation

def prepare_batch_data(rows, download_path):
    """Prepare batch data from CSV rows."""
    batch_paths = []
    batch_rows = []
    
    for row in rows:
        arxiv_id = row.get('paper_arxiv_id', '').strip()
        latex_path = row.get('path', '').strip()
        
        if not arxiv_id or not latex_path:
            print(f"Skipping row with missing data: arxiv_id='{arxiv_id}', latex_path='{latex_path}'")
            # Still add to batch_rows to maintain alignment, but with None path
            batch_rows.append(row)
            batch_paths.append(None)
            continue
        
        try:
            full_path = figure_latex_path_to_path(
                path=download_path,
                arxiv_id=arxiv_id,
                latex_path=latex_path
            )
            batch_paths.append(full_path)
            batch_rows.append(row)
        except Exception as e:
            print(f"Error generating path for {arxiv_id}/{latex_path}: {e}")
            batch_rows.append(row)
            batch_paths.append(None)
    
    return batch_rows, batch_paths

def process_batch(batch_rows, batch_paths):
    """Process a batch of images and return results."""
    # Filter out None paths for processing
    valid_paths = [path for path in batch_paths if path is not None]
    print("Here1")
    
    if not valid_paths:
        print("No valid paths in batch, skipping visual adaptation")
        # Return empty results for all rows in batch
        return [(row, None, None) for row in batch_rows]
    
    try:
        print(f"Processing batch of {len(valid_paths)} images...")
        image_tags_projections = visual_adaptation(image_paths=valid_paths)
        image_tags_list = [item[0] for item in image_tags_projections]
        image_projections_list = [item[1] for item in image_tags_projections]
        
        print(f"image_tags_list: {image_tags_list}")
        # Align results with original batch (including None paths)
        results = []
        valid_idx = 0
        
        for i, (row, path) in enumerate(zip(batch_rows, batch_paths)):
            if path is None:
                results.append((row, None, None))
            else:
                # Get results for this valid path
                image_tags = image_tags_list[valid_idx] if valid_idx < len(image_tags_list) else None
                image_projections = image_projections_list[valid_idx] if valid_idx < len(image_projections_list) else None
                results.append((row, image_tags, image_projections))
                valid_idx += 1
        
        return results
        
    except Exception as e:
        print(f"Error processing batch: {e}")
        return [(row, None, None) for row in batch_rows]

def save_batch_results(batch_results, save_path, write_header=False):
    """
    Save batch results to CSV file immediately to preserve progress.
    Each row contains: original_data + image_tags + image_projections
    """
    if not batch_results:
        print("Warning: No batch results to save")
        return False
    
    # Get fieldnames from first result
    first_row = batch_results[0][0]
    fieldnames = list(first_row.keys()) + ['image_tags', 'image_projections']
    
    try:
        mode = 'w' if write_header else 'a'
        with open(save_path, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if write_header:
                writer.writeheader()
                print(f"Created new output file: {save_path}")
            
            # Write each row immediately (original data + processed results)
            for original_row, image_tags, image_projections in batch_results:
                new_row = original_row.copy()
                new_row['image_tags'] = str(image_tags) if image_tags is not None else ""
                new_row['image_projections'] = str(image_projections) if image_projections is not None else ""
                writer.writerow(new_row)
        
        print(f"✓ Successfully saved batch of {len(batch_results)} results to {save_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error saving batch results: {e}")
        return False

def process_csv_in_batches(source_path, download_path, save_path, batch_size):
    """Process CSV file in batches."""
    try:
        with open(source_path, 'r', newline='', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            
            # Check for required columns
            if 'paper_arxiv_id' not in reader.fieldnames or 'path' not in reader.fieldnames:
                print("Error: CSV must contain 'paper_arxiv_id' and 'path' columns")
                print(f"Found columns: {reader.fieldnames}")
                return
            
            batch = []
            batch_num = 0
            total_processed = 0
            first_batch = True
            
            for row in reader:
                batch.append(row)
                
                # Process batch when it reaches the specified size
                if len(batch) >= batch_size:
                    batch_num += 1
                    print(f"\n=== Processing Batch {batch_num} ({len(batch)} items) ===")
                    
                    # Prepare and process batch
                    batch_rows, batch_paths = prepare_batch_data(batch, download_path)
                    batch_results = process_batch(batch_rows, batch_paths)
                    
                    # Save results immediately after processing each batch
                    print(f"Saving batch {batch_num} results...")
                    save_success = save_batch_results(batch_results, save_path, write_header=first_batch)
                    
                    if save_success:
                        total_processed += len(batch)
                        print(f"✓ Batch {batch_num} completed successfully. Total processed: {total_processed}")
                        print(f"Progress saved to: {save_path}")
                    else:
                        print(f"✗ Failed to save batch {batch_num} results!")
                        # You could choose to continue or exit here depending on requirements
                    
                    # Clear batch and update flags
                    batch = []
                    first_batch = False
            
            # Process remaining rows if any
            if batch:
                batch_num += 1
                print(f"\n=== Processing Final Batch {batch_num} ({len(batch)} items) ===")
                
                batch_rows, batch_paths = prepare_batch_data(batch, download_path)
                batch_results = process_batch(batch_rows, batch_paths)
                
                # Save final batch results immediately
                print(f"Saving final batch {batch_num} results...")
                save_success = save_batch_results(batch_results, save_path, write_header=first_batch)
                
                if save_success:
                    total_processed += len(batch)
                    print(f"✓ Final batch completed successfully.")
                else:
                    print(f"✗ Failed to save final batch results!")
            
            print(f"\n=== Processing Complete ===")
            print(f"Total rows processed: {total_processed}")
            print(f"Results saved to: {save_path}")
        
    except FileNotFoundError:
        print(f"Error: Could not open file '{source_path}'")
    except csv.Error as e:
        print(f"Error reading CSV: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process CSV file in batches to generate image tags/projections')
    parser.add_argument('--source_path', required=True, help='Path to the input CSV file')
    parser.add_argument('--download_path', default='./download', help='Download directory path')
    parser.add_argument('--save_path', required=True, help='Path to save processed results')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of images to process in each batch')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.source_path):
        print(f"Error: CSV file '{args.source_path}' not found")
        return
    
    if args.batch_size < 1:
        print("Error: batch_size must be at least 1")
        return
    
    # Create download directory if needed
    os.makedirs(args.download_path, exist_ok=True)
    
    print(f"Starting batch processing with batch size: {args.batch_size}")
    print(f"Input: {args.source_path}")
    print(f"Output: {args.save_path}")
    print(f"Download path: {args.download_path}")
    
    # Process CSV in batches
    process_csv_in_batches(args.source_path, args.download_path, args.save_path, args.batch_size)

if __name__ == "__main__":
    main()