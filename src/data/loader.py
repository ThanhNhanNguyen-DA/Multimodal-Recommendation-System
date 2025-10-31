import os
import json
import gzip
from typing import List, Dict, Any

def load_jsonl_gz(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_fashion_data(data_dir: str) -> List[Dict[str, Any]]:
    fashion_data = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.jsonl.gz'):
            file_path = os.path.join(data_dir, file_name)
            fashion_data.extend(load_jsonl_gz(file_path))
    return fashion_data

def load_images(image_dir: str) -> List[str]:
    image_paths = []
    for file_name in os.listdir(image_dir):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(image_dir, file_name))
    return image_paths