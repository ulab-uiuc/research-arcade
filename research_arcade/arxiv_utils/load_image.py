import os
from PIL import Image

def load_image(data_dir: str, image_path: str) -> Image.Image:
    # Join directory and image filename
    path = os.path.join(data_dir, image_path)

    # Open image in binary mode
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()  # ensure image is fully loaded

    return img
