import csv
import pandas as pd

section_path = "./jingjun/paper_section_order.csv"
paragraph_path = "./jingjun/paragraphs.csv"
output_path = "./jingjun/paragraph_figure_more_more_enhanced.csv"  # Fixed extension

# Read the data using pandas for easier searching
sections_df = pd.read_csv(section_path)
paragraphs_df = pd.read_csv(paragraph_path)

# Prepare output data
output_data = []

# Process each section
for _, section in sections_df.iterrows():
    section_id = section.iloc[0]  # section[0]
    arxiv_id = section.iloc[1]    # section[1]
    section_name = section.iloc[2]  # section[2]
    section_order = section.iloc[3]  # section[3]
    
    combo = (arxiv_id, section_name)
    # print(combo)
    
    # Find matching paragraphs in the paragraph file
    # Assuming paragraphs.csv has columns that include arxiv_id and section_name
    # You may need to adjust these column names based on your actual CSV structure
    matching_paragraphs = paragraphs_df[
        (paragraphs_df.iloc[:, 3] == arxiv_id) & 
        (paragraphs_df.iloc[:, 4] == section_name)  # Adjust column indices as needed
    ]
    # print(len(matching_paragraphs))
    
    # Assign unique paragraph order within this section
    paragraph_order = 1
    for _, paragraph in matching_paragraphs.iterrows():
        # Create enhanced paragraph record
        enhanced_paragraph = list(paragraph.values)  # Convert to list
        enhanced_paragraph.extend([
            section_id,           # Append section_id
            paragraph_order,      # Append unique paragraph order
            section_order         # Append section order for reference
        ])
        
        output_data.append(enhanced_paragraph)
        paragraph_order += 1

# Write to output file
with open(output_path, 'w', newline='') as output_file:
    output_writer = csv.writer(output_file)
    
    # Write header (adjust based on your paragraph CSV structure)
    if not paragraphs_df.empty:
        header = list(paragraphs_df.columns) + ['section_id', 'paragraph_order', 'section_order']
        output_writer.writerow(header)
    
    # Write all enhanced paragraphs
    output_writer.writerows(output_data)
    
print(f"Processing complete. Output written to {output_path}")
print(f"Total enhanced paragraphs: {len(output_data)}")