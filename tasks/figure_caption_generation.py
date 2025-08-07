from typing import List, Union
from PIL import Image
import torch
from evaluation_metrics import evaluation_metrics

class FigureCaptionGeneration:
    def __init__(self, model, processor, tokenizer, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.processor = processor
        self.tokenizer = tokenizer

    def preprocess(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        return pixel_values.to(self.device)

    def generate_captions(self, images):
        captions = []

        for image in images:
            pixel_values = self.preprocess(image)
            output_ids = self.model.generate(
                pixel_values,
                max_length=self.max_length,
                num_beams=self.num_beams,
                no_repeat_ngram_size=self.no_repeat_ngram_size
            )
            caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            captions.append(caption)

        return captions
    
    def evaluation(self, generated_answer, correct_answer, evaluation_metrics, parameters):
        return evaluation_metrics(evaluation_metrics=evaluation_metrics, generated_answer=generated_answer, correct_answer=correct_answer, parameters=parameters)
        pass