import csv
import pandas as pd

ref_path = "./jingjun/paragraph_table_more_enhanced.csv"
paragraph_path = "./jingjun/paragraph_enhanced.csv"
output_path = "./jingjun/paragraph_table_more_more_enhanced.csv"

# Read the data using pandas
ref_df = pd.read_csv(ref_path)
paragraphs_df = pd.read_csv(paragraph_path)

print(f"Reference data shape: {ref_df.shape}")
print(f"Paragraph data shape: {paragraphs_df.shape}")

# Add paragraph_global_id column to ref_df (not ref_path)
ref_df['paragraph_global_id'] = None

# Process each row in ref_df to find matches and update paragraph_global_id
for idx, ref_row in ref_df.iterrows():
    # Extract search criteria from ref_row
    arxiv_id = ref_row.iloc[3]
    paragraph_id = ref_row.iloc[1] 
    section_name = ref_row.iloc[2]
    
    # Find matching rows in paragraphs_df
    mask = (
        (paragraphs_df.iloc[:, 3] == arxiv_id) & 
        (paragraphs_df.iloc[:, 4] == section_name) &
        (paragraphs_df.iloc[:, 6] == paragraph_id)
    )

    # Check if any matches found
    if mask.any():  # Use mask.any() instead of not mask.empty
        matching_rows = paragraphs_df[mask]
        paragraph_global_id = matching_rows.iloc[0, 0]  # Get first match, column 0
        ref_df.at[idx, 'paragraph_global_id'] = paragraph_global_id
        print(f"Found match for row {idx}: arxiv_id={arxiv_id}, section={section_name}, paragraph_id={paragraph_id}")
    else: 
        ref_df.at[idx, 'paragraph_global_id'] = None
        print(f"No match found for row {idx}: arxiv_id={arxiv_id}, section={section_name}, paragraph_id={paragraph_id}")

# Write the enhanced dataframe to CSV (ref_df, not paragraphs_df)
ref_df.to_csv(output_path, index=False)

print(f"Processing complete. Output written to {output_path}")
print(f"Total rows in output: {len(ref_df)}")
print(f"Rows with paragraph_global_id: {ref_df['paragraph_global_id'].notna().sum()}")
