"""
Augment image with description
"""

# Given an paper and its image information, first obtain the path to the image. After than, given the path, generate the image description. Untimately, we can skip using the model when fine tuning qwen4. In addition, we don't need to re-generate the text description of the images one more time.


def visual_adaptation(image_paths, model_name="llava-hf/llava-1.5-7b-hf", projection_dimension=4096):
    """
    Updated visual_adaptation function that uses LLaVA for image description instead of CLIP tags.
    Returns pairs of [description_list, projection] where description_list contains the LLaVA description.
    """
    descriptions = []
    images = []

    # First, load all images
    for p in image_paths:
        path = Path(p)
        if not path.exists():
            print(f"[WARN] Path not found: {path}")
            images.append(None)
            continue

        ext = path.suffix.lower()
        if ext == ".pdf":
            img = pdf_first_page_to_image(path)
        elif ext in {".jpg", ".jpeg", ".png"}:
            img = _as_rgb_image(path)
        else:
            print(f"[WARN] Unsupported extension '{ext}' for {path}; skipping.")
            img = None

        images.append(img)
    
    # Load LLaVA model once for all images
    llava_model, llava_processor = None, None
    if any(img is not None for img in images):
        try:
            from transformers import LlavaProcessor, LlavaForConditionalGeneration
            
            llava_processor = LlavaProcessor.from_pretrained(model_name)
            llava_model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval()
            print(f"Loaded LLaVA model: {model_name}")
        except Exception as e:
            print(f"[ERROR] Failed to load LLaVA model: {e}")
            # Fallback to CLIP if LLaVA fails
            # return _visual_adaptation_clip_fallback(image_paths, projection_dimension)
            return []
        
    # Process each image
    for image in images:
        if not image:
            pairs.append([None, None])
            continue

        try:
            # Generate description with LLaVA
            description = _generate_llava_description(llava_model, llava_processor, image)
            print(f"LLaVA description: {description[:100]}...")  # Show first 100 chars
            
            # Create projection if needed
            projected = None
            # if projection_dimension:
                # For projection, we still need CLIP features as LLaVA doesn't directly provide embeddings
                # Load CLIP for projection only
                # clip_model, clip_processor = load_model("openai/clip-vit-base-patch32")
                # img_feats = clip_image_features(clip_model, clip_processor, image)
                # projected = project_to_embedding_space(img_feats, target_dim=projection_dimension)
            
            # Return description as a list (to maintain compatibility with existing code structure)
            descriptions.append(description)
            
        except Exception as e:
            print(f"[ERROR] Failed to process image with LLaVA: {e}")
            pairs.append([None, None])

    return descriptions


