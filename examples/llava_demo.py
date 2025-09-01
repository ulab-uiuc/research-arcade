# examples/llava_demo.py
# ------------------------------------------------------------
# LLaVA for direct image-to-text generation from PDF
# ------------------------------------------------------------
import os
from PIL import Image
from pdf2image import convert_from_path
import torch


# -------------------------------
# 0) Config
# -------------------------------
MODEL_ID = "llava-hf/llava-1.5-7b-hf"  # More stable than llava-next
PDF_PATH = "./Konigsberg2.pdf"
JPG_SAVE_PATH = "./Konigsberg2_page1.jpg"   # optional on-disk preview
PDF_DPI = 200


# -------------------------------
# 1) PDF -> PIL image
# -------------------------------
def pdf_first_page_to_image(pdf_path: str, dpi: int = 200, save_jpg_path: str = None) -> Image.Image:
    pages = convert_from_path(pdf_path, dpi=dpi)   # requires poppler
    if not pages:
        raise RuntimeError(f"No pages found in PDF: {pdf_path}")
    img = pages[0].convert("RGB")
    if save_jpg_path:
        img.save(save_jpg_path, "JPEG")
    return img


# -------------------------------
# 2) Load LLaVA model and processor
# -------------------------------
def load_llava():
    """Load LLaVA model and processor with proper error handling"""
    try:
        # Try standard LLaVA 1.5 first (most stable)
        from transformers import LlavaProcessor, LlavaForConditionalGeneration
        
        processor = LlavaProcessor.from_pretrained(MODEL_ID)
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print(f"Loaded standard LLaVA: {MODEL_ID}")
        return model, processor, "standard"
        
    except Exception as e:
        print(f"Failed to load standard LLaVA: {e}")
        
        # Fallback to LLaVA-Next
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            
            next_model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
            processor = LlavaNextProcessor.from_pretrained(next_model_id)
            model = LlavaNextForConditionalGeneration.from_pretrained(
                next_model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            print(f"Loaded LLaVA-Next: {next_model_id}")
            return model, processor, "next"
            
        except Exception as e2:
            raise RuntimeError(f"Failed to load any LLaVA model. Standard: {e}, Next: {e2}")


# -------------------------------
# 3) Generate description with LLaVA
# -------------------------------
def generate_description(model, processor, image: Image.Image, model_type: str, prompt_text: str = None) -> str:
    """Generate text description from image using LLaVA"""
    
    # Create appropriate prompt based on model type
    if prompt_text is None:
        prompt_text = "Describe this image in detail, focusing on its structure, content, and any relationships or patterns you observe."
    
    try:
        if model_type == "standard":
            # LLaVA 1.5 requires specific prompt format
            prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{prompt_text} ASSISTANT:"
            
            # Process inputs - use explicit parameter names for LLaVA 1.5
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            
        else:
            # LLaVA-Next format  
            prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
            inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        # Move inputs to the same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        print(f"Input keys: {list(inputs.keys())}")
        print(f"Input shapes: {[(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in inputs.items()]}")
        
        # Generate response with more conservative parameters
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,  # Use greedy decoding for more stability
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode response properly
        if model_type == "standard":
            # For LLaVA 1.5, exclude input tokens from output
            input_token_len = inputs["input_ids"].shape[1]
            generated_tokens = output_ids[0][input_token_len:]
            response = processor.decode(generated_tokens, skip_special_tokens=True)
        else:
            # For LLaVA-Next, decode full output and extract assistant response
            response = processor.decode(output_ids[0], skip_special_tokens=True)
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
        
        return response.strip()
        
    except Exception as e:
        print(f"Full error details: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating description: {str(e)}"


# -------------------------------
# 4) Alternative prompts for different use cases
# -------------------------------
def get_prompt_variants():
    """Return different prompt options for various analysis needs"""
    return {
        "general": "Describe this image in detail.",
        
        "technical": "Analyze this technical diagram or document. Describe its structure, components, and any mathematical or scientific content you can identify.",
        
        "academic": "This appears to be from an academic document. Please describe what you see, including any text, diagrams, equations, or structured content.",
        
        "concise": "Provide a concise but comprehensive description of this image in 2-3 sentences.",
        
        "detailed": "Provide a detailed analysis of this image. Describe everything you can observe, including layout, text content, visual elements, and their relationships."
    }


# -------------------------------
# 5) Main function
# -------------------------------
def main():
    print("Loading LLaVA model...")
    
    # A) PDF -> image
    try:
        image = pdf_first_page_to_image(PDF_PATH, dpi=PDF_DPI, save_jpg_path=JPG_SAVE_PATH)
        print(f"Converted PDF to image. Saved preview to: {JPG_SAVE_PATH}")
        print(f"Image size: {image.size}")
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return
    
    # B) Load LLaVA
    try:
        model, processor, model_type = load_llava()
        print(f"LLaVA model loaded successfully! Type: {model_type}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # C) Generate descriptions
    prompt_variants = get_prompt_variants()
    
    # Default general description
    print("\n" + "="*60)
    print("GENERAL DESCRIPTION")
    print("="*60)
    description = generate_description(model, processor, image, model_type)
    print(description)
    
    # Technical analysis
    print("\n" + "="*60)
    print("TECHNICAL ANALYSIS")
    print("="*60)
    technical_desc = generate_description(model, processor, image, model_type, prompt_variants["technical"])
    print(technical_desc)
    
    # You can enable more variants:
    # print("\n" + "="*60)
    # print("DETAILED ANALYSIS")
    # print("="*60)
    # detailed_desc = generate_description(model, processor, image, model_type, prompt_variants["detailed"])
    # print(detailed_desc)


# -------------------------------
# 6) Alternative: Batch processing multiple pages
# -------------------------------
def process_all_pdf_pages(pdf_path: str, max_pages: int = 5):
    """Process multiple pages from PDF"""
    print("Loading LLaVA model...")
    model, processor, model_type = load_llava()
    
    print(f"Converting PDF pages (max {max_pages})...")
    pages = convert_from_path(pdf_path, dpi=PDF_DPI)
    
    for i, page in enumerate(pages[:max_pages]):
        print(f"\n{'='*60}")
        print(f"PAGE {i+1} DESCRIPTION")
        print('='*60)
        
        page_rgb = page.convert("RGB")
        description = generate_description(model, processor, page_rgb, model_type)
        print(description)


if __name__ == "__main__":
    main()
    
    # Uncomment to process multiple pages:
    # process_all_pdf_pages(PDF_PATH, max_pages=3)