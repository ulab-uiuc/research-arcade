#!/usr/bin/env python3
"""
Table Text Column Description Generator using Transformers and Qwen3
This script generates natural language descriptions of table content stored in the table_text column.
"""

import pandas as pd
import json
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import sys
import io

class TableTextDescriptionGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        """
        Initialize the table text description generator with Transformers and Qwen3
        
        Args:
            model_name: The Qwen model to use (default: Qwen3-8B)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def initialize_model(self):
        """Initialize the model using Transformers library"""
        try:
            print(f"Loading {self.model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def generate_with_thinking_control(self, prompt: str, enable_thinking: bool = False) -> str:
        """
        Generate response with direct control over thinking mode
        
        Args:
            prompt: Input prompt
            enable_thinking: Whether to enable thinking mode
            
        Returns:
            Clean content without thinking process
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template with thinking control
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate response
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        if enable_thinking:
            # Parse thinking content as shown in your example
            try:
                # Find </think> token (151668)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            
            # Extract only the content part (after thinking)
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            return content
        else:
            # No thinking mode, return full output
            return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def parse_table_text(self, table_text: str) -> Optional[pd.DataFrame]:
        """
        Parse table_text string into a DataFrame
        Handles various table formats (CSV, TSV, pipe-separated, etc.)
        
        Args:
            table_text: Raw table text content
            
        Returns:
            DataFrame if parsing successful, None otherwise
        """
        if not table_text or pd.isna(table_text):
            return None
            
        try:
            # Clean the table text
            table_text = str(table_text).strip()
            
            # Try different separators
            separators = ['\t', '|', ',', ';']
            
            for sep in separators:
                try:
                    # Use StringIO to read the text as if it were a file
                    df = pd.read_csv(io.StringIO(table_text), sep=sep, engine='python')
                    
                    # Check if parsing was successful (more than 1 column usually indicates success)
                    if len(df.columns) > 1 and len(df) > 0:
                        # Clean column names
                        df.columns = df.columns.str.strip()
                        return df
                except:
                    continue
            
            # If all separators fail, try space separation
            try:
                lines = table_text.split('\n')
                if len(lines) > 1:
                    # Split by whitespace and create DataFrame
                    data = []
                    headers = None
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        parts = line.split()
                        if headers is None:
                            headers = parts
                        else:
                            data.append(parts)
                    
                    if headers and data:
                        df = pd.DataFrame(data, columns=headers)
                        return df
            except:
                pass
                
            return None
            
        except Exception as e:
            print(f"Error parsing table text: {e}")
            return None
    
    def format_table_for_description(self, df: pd.DataFrame, original_text: str) -> str:
        """
        Format a parsed DataFrame for description generation
        
        Args:
            df: The parsed DataFrame
            original_text: Original table text for reference
            
        Returns:
            Formatted content for LLM
        """
        if df is None or df.empty:
            return f"Raw table text:\n{original_text}"
        
        # Get basic info about the table
        num_rows, num_cols = df.shape
        
        # Format the table nicely
        table_display = df.to_string(index=False, max_cols=None, max_rows=20)
        
        # Add metadata
        content = f"""
Table Structure:
- Dimensions: {num_rows} rows × {num_cols} columns
- Column names: {', '.join(df.columns.tolist())}

Table Content:
{table_display}
"""
        
        # If table is too large, add note
        if num_rows > 20:
            content += f"\n(Note: Showing first 20 rows of {num_rows} total rows)"
            
        return content.strip()
    
    def create_description_prompt(self, table_content: str, context_info: Dict = None) -> str:
        """
        Create a prompt for table description generation
        
        Args:
            table_content: Formatted table content
            context_info: Optional context (id, caption, etc.)
            
        Returns:
            Complete prompt string
        """
        context_section = ""
        if context_info:
            context_parts = []
            if context_info.get('id'):
                context_parts.append(f"Table ID: {context_info['id']}")
            if context_info.get('caption'):
                context_parts.append(f"Caption: {context_info['caption']}")
            if context_info.get('paper_arxiv_id'):
                context_parts.append(f"Source Paper: {context_info['paper_arxiv_id']}")
            
            if context_parts:
                context_section = f"\nContext Information:\n" + "\n".join(context_parts) + "\n"
        
        prompt = f"""You are an expert data analyst. Your task is to analyze the given table data and provide a comprehensive, natural language description of its content and key insights.

Please provide:
1. A clear overview of what the table contains and its purpose
2. Description of the data structure (columns, data types, relationships)
3. Key patterns, trends, or notable observations in the data

IMPORTANT: Provide only the final analysis result. Do not show your thinking process or reasoning steps. Start directly with the table description.

Be specific, accurate, and informative while keeping the description accessible and well-structured.

Please analyze and describe the following table data:
{context_section}
Table Data:
{table_content}"""
        
        return prompt
    
    def generate_table_description(self, table_text: str, context_info: Dict = None) -> str:
        """
        Generate description for table content from table_text
        
        Args:
            table_text: Raw table text content
            context_info: Optional context information
            
        Returns:
            Generated description
        """
        if self.model is None:
            self.initialize_model()
        
        # Parse and format table
        df = self.parse_table_text(table_text)
        formatted_content = self.format_table_for_description(df, table_text)
        prompt = self.create_description_prompt(formatted_content, context_info)
        
        # Generate with thinking disabled to get direct content
        try:
            description = self.generate_with_thinking_control(prompt, enable_thinking=False)
            return description
        except Exception as e:
            return f"Error generating description: {e}"
    
    def process_csv_file(self, csv_path: str, output_path: Optional[str] = None, 
                        start_idx: int = 0, end_idx: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process a CSV file and generate descriptions for all table_text entries
        
        Args:
            csv_path: Path to the CSV file
            output_path: Optional path to save the results
            start_idx: Starting index for processing
            end_idx: Ending index for processing (None = process all)
            
        Returns:
            List of dictionaries containing the analysis results
        """
        try:
            # Load the CSV
            df = pd.read_csv(csv_path)
            print(f"Loaded CSV with shape: {df.shape}")
            
            # Validate required columns
            required_cols = ['table_text']
            optional_cols = ['id', 'paper_arxiv_id', 'caption', 'label']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Determine processing range
            total_rows = len(df)
            if end_idx is None:
                end_idx = total_rows
            else:
                end_idx = min(end_idx, total_rows)
            
            print(f"Processing rows {start_idx} to {end_idx-1} ({end_idx-start_idx} rows)")
            
            results = []
            
            # Process each row
            for idx in range(start_idx, end_idx):
                try:
                    row = df.iloc[idx]
                    print(f"Processing row {idx+1}/{end_idx}...")
                    
                    # Extract context information
                    context_info = {}
                    for col in optional_cols:
                        if col in df.columns and not pd.isna(row[col]):
                            context_info[col] = row[col]
                    
                    # Generate description
                    table_text = row['table_text']
                    if pd.isna(table_text) or not str(table_text).strip():
                        description = "No table content available"
                    else:
                        print(f"Generating description for row {idx}...")
                        description = self.generate_table_description(table_text, context_info)
                    
                    print(f"Generated description: {description[:100]}...")
                    
                    # Create result entry
                    result = {
                        "row_index": idx,
                        "id": row.get('id', f'row_{idx}'),
                        "paper_arxiv_id": row.get('paper_arxiv_id', ''),
                        "caption": row.get('caption', ''),
                        "label": row.get('label', ''),
                        "generated_description": description
                    }
                    
                    results.append(result)
                    
                    # Save incrementally every 5 processed rows
                    if output_path and (idx - start_idx + 1) % 5 == 0:
                        temp_output = output_path.replace('.json', f'_temp_{idx}.json')
                        try:
                            with open(temp_output, 'w', encoding='utf-8') as f:
                                json.dump(results, f, indent=2, ensure_ascii=False)
                            print(f"Incremental save completed: {temp_output}")
                        except Exception as save_error:
                            print(f"Warning: Could not save incremental file: {save_error}")
                    
                    # Print progress
                    if (idx - start_idx + 1) % 10 == 0:
                        print(f"Completed {idx - start_idx + 1} descriptions...")
                        
                except Exception as row_error:
                    print(f"Error processing row {idx}: {row_error}")
                    # Continue with next row instead of failing completely
                    error_result = {
                        "row_index": idx,
                        "id": row.get('id', f'row_{idx}') if 'row' in locals() else f'row_{idx}',
                        "paper_arxiv_id": '',
                        "caption": '',
                        "label": '',
                        "generated_description": f"Error processing row: {str(row_error)}"
                    }
                    results.append(error_result)
                    continue
            
            # Save final results
            if output_path:
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"Final results saved to: {output_path}")
                except Exception as save_error:
                    print(f"Error saving final results: {save_error}")
                    # Try saving as backup
                    backup_path = output_path.replace('.json', '_backup.json')
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"Results saved to backup file: {backup_path}")
            
            return results
            
        except Exception as e:
            print(f"Error processing file: {e}")
            import traceback
            traceback.print_exc()
            return [{"error": str(e)}]

    def process_single_entry(self, csv_path: str, row_id: str) -> Dict[str, Any]:
        """
        Process a single entry by ID
        
        Args:
            csv_path: Path to the CSV file
            row_id: ID of the row to process
            
        Returns:
            Dictionary containing the analysis result
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Find the row with matching ID
            if 'id' not in df.columns:
                raise ValueError("CSV must have 'id' column for single entry processing")
            
            matching_rows = df[df['id'] == row_id]
            if len(matching_rows) == 0:
                return {"error": f"No row found with id: {row_id}"}
            
            row = matching_rows.iloc[0]
            
            # Extract context
            context_info = {
                'id': row.get('id', ''),
                'paper_arxiv_id': row.get('paper_arxiv_id', ''),
                'caption': row.get('caption', ''),
                'label': row.get('label', '')
            }
            
            # Generate description
            table_text = row['table_text']
            description = self.generate_table_description(table_text, context_info)
            
            return {
                "id": row_id,
                "paper_arxiv_id": row.get('paper_arxiv_id', ''),
                "caption": row.get('caption', ''),
                "label": row.get('label', ''),
                "generated_description": description
            }
            
        except Exception as e:
            return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='Generate descriptions for table_text column using Transformers and Qwen3')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file to analyze')
    parser.add_argument('--output', type=str, help='Path to save the description output (CSV format)')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-8B', 
                       help='Model name to use (default: Qwen/Qwen3-8B)')
    parser.add_argument('--start', type=int, default=0, 
                       help='Starting row index (default: 0)')
    parser.add_argument('--end', type=int, 
                       help='Ending row index (default: process all rows)')
    parser.add_argument('--single-id', type=str, 
                       help='Process only the row with this specific ID')
    
    args = parser.parse_args()
    
    # Initialize the generator
    generator = TableTextDescriptionGenerator(model_name=args.model)
    
    if args.single_id:
        print(f"Processing single entry with ID: {args.single_id}")
        result_df = generator.process_single_entry(args.csv, args.single_id)
        
        if "error" not in result_df.columns and len(result_df) > 0:
            print("\n" + "="*80)
            print(f"GENERATED DESCRIPTION FOR ID: {args.single_id}")
            print("="*80)
            row = result_df.iloc[0]
            print(f"Caption: {row.get('caption', 'N/A')}")
            print(f"Paper: {row.get('paper_arxiv_id', 'N/A')}")
            print("-"*80)
            print(row.get("generated_description", "No description generated"))
            
            # Save single result if output path provided
            if args.output:
                result_df.to_csv(args.output, index=False, encoding='utf-8')
                print(f"Result saved to: {args.output}")
        else:
            print(f"Error: {result_df.iloc[0].get('error', 'Unknown error') if len(result_df) > 0 else 'Unknown error'}")
    
    else:
        print(f"Processing CSV file: {args.csv}")
        result_df = generator.process_csv_file(args.csv, args.output, args.start, args.end)
        
        if "error" not in result_df.columns and len(result_df) > 0:
            print(f"\nSuccessfully processed {len(result_df)} entries")
            print(f"Output CSV contains {len(result_df.columns)} columns: {list(result_df.columns)}")
            
            # Show a sample result
            if len(result_df) > 0:
                sample = result_df.iloc[0]
                print("\n" + "="*80)
                print("SAMPLE GENERATED DESCRIPTION:")
                print("="*80)
                print(f"ID: {sample.get('id', 'N/A')}")
                print(f"Caption: {sample.get('caption', 'N/A')}")
                print("-"*80)
                print(sample.get("generated_description", "No description generated"))
        else:
            error_msg = result_df.iloc[0].get('error', 'Unknown error') if len(result_df) > 0 else 'Unknown error'
            print(f"Error: {error_msg}")

if __name__ == "__main__":
    main()