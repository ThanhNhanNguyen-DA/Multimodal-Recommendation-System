import pandas as pd
from PIL import Image
from typing import List, Tuple

def normalize_text(text: str) -> str:
    # Basic text normalization: lowercasing and stripping whitespace
    return text.lower().strip()

def preprocess_images(image_paths: List[str], target_size: Tuple[int, int] = (224, 224)) -> List[Image.Image]:
    processed_images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = img.resize(target_size, Image.ANTIALIAS)
        processed_images.append(img)
    return processed_images

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Apply text normalization to the title and description
    df['title'] = df['title'].apply(normalize_text)
    df['description'] = df['description'].apply(normalize_text)
    return df