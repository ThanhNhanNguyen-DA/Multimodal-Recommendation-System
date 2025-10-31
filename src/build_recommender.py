import argparse
import os
import json
import gzip
import pandas as pd
from typing import List, Dict, Any, Optional

from data.loader import load_data
from data.preprocess import preprocess_data
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.fusion import FusionModel
from utils.io import save_model, load_model

def build_recommender(data_path: str, out_dir: str, max_rows: Optional[int] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print('Loading data...')
    data = load_data(data_path, max_rows)
    
    print('Preprocessing data...')
    processed_data = preprocess_data(data)

    print('Training text encoder...')
    text_encoder = TextEncoder()
    text_encoder.train(processed_data['text'])

    print('Training image encoder...')
    image_encoder = ImageEncoder()
    image_encoder.train(processed_data['images'])

    print('Fusing models...')
    fusion_model = FusionModel(text_encoder, image_encoder)

    print('Saving models...')
    save_model(text_encoder, os.path.join(out_dir, 'text_encoder.pkl'))
    save_model(image_encoder, os.path.join(out_dir, 'image_encoder.pkl'))
    save_model(fusion_model, os.path.join(out_dir, 'fusion_model.pkl'))

    print('Recommender built and models saved to', out_dir)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a fashion recommender system.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--out_dir', type=str, default='artifacts', help='Directory to save models')
    parser.add_argument('--max_rows', type=int, default=None, help='Limit rows for quick runs')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    build_recommender(args.data_path, args.out_dir, args.max_rows)

if __name__ == '__main__':
    main()