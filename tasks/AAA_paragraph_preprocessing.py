import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import argparse
import os
import gc
from typing import Dict, List, Any

class Qwen3HierarchicalEmbedder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", device=None):
        """
        Initialize Qwen3 embedding generator for hierarchical paper data
        """
        # Set memory fraction FIRST to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.85)
            torch.cuda.empty_cache()
            gc.collect()
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.embedding_dim = 1024  # Qwen3-Embedding-0.6B dimension
        
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding side to left for Qwen3
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Memory-optimized model loading
        self.model = AutoModel.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
        
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Print memory usage
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        
        print("Model loaded successfully!")
    
    def clear_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def last_token_pool(self, last_hidden_states, attention_mask):
        """Pool the last token embeddings based on attention mask"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def generate_embeddings(self, texts: List[str], task_instruction: str, 
                          batch_size: int = 8, max_length: int = 4096) -> np.ndarray:
        """
        Generate embeddings for a list of texts with task-specific instructions
        """
        if not texts:
            return np.empty((0, self.embedding_dim))
        
        # Add instruction to each text
        instructed_texts = [f"Instruct: {task_instruction}\nQuery: {text}" for text in texts]
        
        all_embeddings = []
        
        for i in range(0, len(instructed_texts), batch_size):
            batch_texts = instructed_texts[i:i + batch_size]
            
            # Tokenize batch
            batch_dict = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Move to CPU immediately to free GPU memory
            all_embeddings.append(embeddings.cpu().numpy())
            
            # Clear intermediate tensors
            del outputs, embeddings, batch_dict
            
            # Clear cache every few batches
            if i % (batch_size * 4) == 0:
                self.clear_memory()
        
        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, self.embedding_dim))

def process_hierarchical_papers(papers_data: List[Dict], embedder: Qwen3HierarchicalEmbedder) -> List[Dict]:
    """
    Process hierarchical paper data and generate embeddings for each component,
    maintaining the same hierarchical architecture as input
    """
    processed_papers = []
    
    for paper_idx, paper in enumerate(tqdm(papers_data, desc="Processing papers")):
        print(f"\nProcessing paper {paper_idx + 1}/{len(papers_data)}: {paper['title'][:50]}...")
        
        # Clear memory before processing each paper
        embedder.clear_memory()
        
        # Create processed paper with same structure but add embeddings
        processed_paper = {
            "title": paper["title"],
            "arxiv_id": paper["arxiv_id"],
            "title_embedding": None,  # Will be filled below
            "sections": []
        }
        
        # Generate title embedding
        print("  Generating title embedding...")
        title_embedding = embedder.generate_embeddings(
            [paper["title"]],
            "Represent this academic paper title for document classification and topic analysis",
            batch_size=1  # Single item
        )
        processed_paper["title_embedding"] = title_embedding[0].tolist()  # Single embedding as list
        
        # Process each section
        for section_idx, section in enumerate(paper["sections"]):
            print(f"    Processing section {section_idx + 1}: {section['section_title']}")
            
            # Generate section title embedding
            section_embedding = embedder.generate_embeddings(
                [section["section_title"]],
                "Represent this academic paper section title for document structure and topic organization",
                batch_size=1  # Single item
            )
            
            processed_section = {
                "section_title": section["section_title"],
                "section_embedding": section_embedding[0].tolist(),  # Single embedding as list
                "paragraphs": []
            }
            
            # Collect paragraph texts for batch processing
            paragraph_texts = [para["content"] for para in section["paragraphs"]]
            
            if paragraph_texts:
                print(f"      Generating embeddings for {len(paragraph_texts)} paragraphs...")
                # Use smaller batch size for paragraphs (they tend to be longer)
                paragraph_embeddings = embedder.generate_embeddings(
                    paragraph_texts,
                    "Represent this academic paragraph for semantic similarity and document structure analysis",
                    batch_size=4,  # Smaller batch size
                    max_length=2048  # Shorter max length for paragraphs
                )
            else:
                paragraph_embeddings = np.empty((0, embedder.embedding_dim))
            
            # Process each paragraph
            for para_idx, paragraph in enumerate(section["paragraphs"]):
                processed_paragraph = {
                    "paragraph_id": paragraph["paragraph_id"],
                    "content": paragraph["content"],
                    "paragraph_embedding": paragraph_embeddings[para_idx].tolist(),  # Convert to list
                    "figures": [],
                    "tables": []
                }
                
                # Process figures in this paragraph
                if paragraph["figures"]:
                    figure_data_with_captions = []
                    figure_texts = []
                    
                    # Build list of figures with valid captions and corresponding texts
                    for fig in paragraph["figures"]:
                        caption = fig.get("caption", "") or ""
                        
                        # Only process figures with non-empty captions
                        if caption.strip():
                            figure_data_with_captions.append(fig)
                            figure_texts.append(caption)
                    
                    # Generate embeddings only if we have valid captions
                    if figure_texts:
                        figure_embeddings = embedder.generate_embeddings(
                            figure_texts,
                            "Represent this figure caption for visual content analysis and document understanding",
                            batch_size=2  # Small batch for figures
                        )
                        
                        # Process only figures that have valid embeddings
                        for fig_idx, fig in enumerate(figure_data_with_captions):
                            processed_figure = {
                                "label": fig.get("label", ""),
                                "caption": fig.get("caption", ""),
                                "path": fig.get("path"),
                                "figure_embedding": figure_embeddings[fig_idx].tolist()  # Convert to list
                            }
                            processed_paragraph["figures"].append(processed_figure)
                    else:
                        # Handle figures with no valid captions - create entries without embeddings
                        for fig in paragraph["figures"]:
                            processed_figure = {
                                "label": fig.get("label", ""),
                                "caption": fig.get("caption", ""),
                                "path": fig.get("path"),
                                "figure_embedding": [0.0] * embedder.embedding_dim  # Zero embedding for empty captions
                            }
                            processed_paragraph["figures"].append(processed_figure)
                
                # Process tables in this paragraph
                if paragraph["tables"]:
                    table_data_with_texts = []
                    table_texts = []
                    
                    # Build list of tables with valid text and corresponding texts
                    for table in paragraph["tables"]:
                        # Safely handle None values in table text
                        caption = table.get('caption', '') or ''
                        table_content = table.get('table_text', '') or ''
                        
                        # Combine caption and table content
                        table_text = f"{caption} {table_content}" if table_content else caption
                        
                        # Truncate very long table texts to prevent memory issues
                        if table_text and len(table_text) > 2000:
                            table_text = table_text[:2000] + "..."
                        
                        # Only process tables with non-empty text
                        if table_text.strip():
                            table_data_with_texts.append(table)
                            table_texts.append(table_text)
                    
                    # Generate embeddings only if we have valid texts
                    if table_texts:
                        table_embeddings = embedder.generate_embeddings(
                            table_texts,
                            "Represent this table caption and content for tabular data analysis and document understanding",
                            batch_size=2,  # Small batch for tables
                            max_length=1024  # Shorter for tables
                        )
                        
                        # Process only tables that have valid embeddings
                        for table_idx, table in enumerate(table_data_with_texts):
                            processed_table = {
                                "label": table.get("label", ""),
                                "caption": table.get("caption", ""),
                                "table_text": table.get("table_text", ""),
                                "table_embedding": table_embeddings[table_idx].tolist()  # Convert to list
                            }
                            processed_paragraph["tables"].append(processed_table)
                    else:
                        # Handle tables with no valid text - create entries without embeddings
                        for table in paragraph["tables"]:
                            processed_table = {
                                "label": table.get("label", ""),
                                "caption": table.get("caption", ""),
                                "table_text": table.get("table_text", ""),
                                "table_embedding": [0.0] * embedder.embedding_dim  # Zero embedding for empty tables
                            }
                            processed_paragraph["tables"].append(processed_table)
                
                processed_section["paragraphs"].append(processed_paragraph)
            
            processed_paper["sections"].append(processed_section)
        
        # Print summary for this paper
        total_paragraphs = sum(len(section["paragraphs"]) for section in processed_paper["sections"])
        total_figures = sum(len(para["figures"]) for section in processed_paper["sections"] for para in section["paragraphs"])
        total_tables = sum(len(para["tables"]) for section in processed_paper["sections"] for para in section["paragraphs"])
        
        print(f"  Paper completed:")
        print(f"    Title: 1 embedding")
        print(f"    Sections: {len(processed_paper['sections'])} embeddings")
        print(f"    Paragraphs: {total_paragraphs} embeddings")
        print(f"    Figures: {total_figures} embeddings")
        print(f"    Tables: {total_tables} embeddings")
        
        # Print current GPU memory usage
        if torch.cuda.is_available():
            print(f"    GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB cached")
        
        processed_papers.append(processed_paper)
    
    return processed_papers

def convert_to_graph_format(processed_papers: List[Dict]) -> List[Dict]:
    """
    Convert hierarchical embeddings to the format needed for your graph neural network
    """
    graph_format_papers = []
    
    for paper in processed_papers:
        # For your graph model, you need paragraph/figure/table/citation embeddings
        # We'll treat sections as a type of "context" or merge them appropriately
        
        graph_paper = {
            "arxiv_id": paper["arxiv_id"],
            "title": paper["title"],
            
            # Main node embeddings (what your model expects)
            "paragraph_embeddings": paper["embeddings"]["paragraphs"],
            "figure_embeddings": paper["embeddings"]["figures"], 
            "table_embeddings": paper["embeddings"]["tables"],
            "citation_embeddings": np.empty((0, 1024)),  # No citations in your current data
            
            # Relationship mappings (required for graph construction)
            "figure_to_paragraph": paper["figure_table_mappings"]["figure_to_paragraph"],
            "table_to_paragraph": paper["figure_table_mappings"]["table_to_paragraph"],
            "citation_to_paragraphs": {},  # Empty for now
            
            # Additional hierarchical information (optional, for future use)
            "section_embeddings": paper["embeddings"]["sections"],
            "title_embedding": paper["embeddings"]["titles"],
            "structure": paper["structure"],
            "figure_to_section": paper["figure_table_mappings"]["figure_to_section"],
            "table_to_section": paper["figure_table_mappings"]["table_to_section"]
        }
        
        graph_format_papers.append(graph_paper)
    
    return graph_format_papers

def save_embeddings(processed_papers: List[Dict], output_path: str):
    """
    Save processed papers with embeddings to JSON file
    The output maintains the same hierarchical structure as input,
    but with embeddings added at each level
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_papers, f, indent=2, ensure_ascii=False)
    
    print(f"Hierarchical embeddings saved to {output_path}")
    print(f"Structure: Each component (title, section, paragraph, figure, table) now includes its embedding")

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for hierarchical paper data')
    parser.add_argument('--input_path', help='Path to the hierarchical papers JSON file')
    parser.add_argument('--output_path', help='Path to save the hierarchical embeddings JSON file')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for embedding generation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda/cpu)')
    args = parser.parse_args()

    # Print debug information
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"PyTorch device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(current_device)
        print(f"PyTorch GPU {current_device}: {props.name}")
        print(f"Total memory: {props.total_memory/1e9:.2f} GB")
    
    # Load hierarchical paper data
    print(f"Loading papers from {args.input_path}...")
    with open(args.input_path, 'r', encoding='utf-8') as f:
        papers_data = json.load(f)
    
    print(f"Loaded {len(papers_data)} papers")
    
    # Initialize embedder
    embedder = Qwen3HierarchicalEmbedder(device=args.device)
    
    # Process papers and generate embeddings (maintains hierarchical structure)
    processed_papers = process_hierarchical_papers(papers_data, embedder)
    
    # Save hierarchical embeddings
    save_embeddings(processed_papers, args.output_path)
    
    print(f"\nProcessing complete!")
    print(f"Processed {len(processed_papers)} papers")
    print(f"Hierarchical embeddings saved to {args.output_path}")
    
    # Print some statistics
    total_sections = sum(len(paper['sections']) for paper in processed_papers)
    total_paragraphs = sum(len(section['paragraphs']) for paper in processed_papers for section in paper['sections'])
    total_figures = sum(len(para['figures']) for paper in processed_papers for section in paper['sections'] for para in section['paragraphs'])
    total_tables = sum(len(para['tables']) for paper in processed_papers for section in paper['sections'] for para in section['paragraphs'])
    
    print(f"\nDataset statistics:")
    print(f"Total papers: {len(processed_papers)}")
    print(f"Total sections: {total_sections}")
    print(f"Total paragraphs: {total_paragraphs}")
    print(f"Total figures: {total_figures}")
    print(f"Total tables: {total_tables}")
    print(f"\nEach component now has its embedding stored in the same hierarchical structure as the input.")
    
    # Show example structure
    if processed_papers:
        print(f"\nExample structure for first paper:")
        paper = processed_papers[0]
        print(f"  title: '{paper['title'][:50]}...'")
        print(f"  title_embedding: [list of {len(paper['title_embedding'])} floats]")
        print(f"  sections: {len(paper['sections'])} sections")
        if paper['sections']:
            section = paper['sections'][0]
            print(f"    section_title: '{section['section_title']}'")
            print(f"    section_embedding: [list of {len(section['section_embedding'])} floats]")
            print(f"    paragraphs: {len(section['paragraphs'])} paragraphs")
            if section['paragraphs']:
                para = section['paragraphs'][0]
                print(f"      paragraph_embedding: [list of {len(para['paragraph_embedding'])} floats]")
                print(f"      figures: {len(para['figures'])} figures")
                print(f"      tables: {len(para['tables'])} tables")

if __name__ == "__main__":
    main()