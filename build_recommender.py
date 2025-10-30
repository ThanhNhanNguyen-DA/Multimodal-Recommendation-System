import argparse
import gzip
import io
import os
import sys
import json
import time
import math
import pathlib
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_amazon_jsonl_gz(path: str, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
                rows.append(obj)
            except json.JSONDecodeError:
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def normalize_products(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    # Common fields in Amazon datasets
    def get_first(lst):
        if isinstance(lst, list) and len(lst) > 0:
            return lst[0]
        return None

    normalized: List[Dict[str, Any]] = []
    for r in rows:
        asin = r.get('asin') or r.get('ASIN')
        title = r.get('title') or r.get('Title')
        description = r.get('description') or r.get('Description')
        if isinstance(description, list):
            description = ' '.join([str(x) for x in description if x])

        image_urls = None
        if 'imageURLHighRes' in r and isinstance(r['imageURLHighRes'], list) and r['imageURLHighRes']:
            image_urls = r['imageURLHighRes']
        elif 'imUrl' in r and r['imUrl']:
            image_urls = [r['imUrl']]

        categories = r.get('categories')
        if isinstance(categories, list) and len(categories) > 0:
            cat = categories[0]
            if isinstance(cat, list):
                category = ' > '.join([str(c) for c in cat])
            else:
                category = str(cat)
        else:
            category = None

        normalized.append({
            'asin': asin,
            'title': title,
            'description': description,
            'image_url': get_first(image_urls) if image_urls else None,
            'category': category,
        })

    df = pd.DataFrame(normalized).dropna(subset=['asin']).reset_index(drop=True)
    return df


def build_text_corpus(df: pd.DataFrame) -> List[str]:
    corpus: List[str] = []
    for _, row in df.iterrows():
        parts = [str(row['title']) if pd.notna(row['title']) else '',
                 str(row['description']) if pd.notna(row['description']) else '']
        corpus.append(' '.join([p for p in parts if p]).strip())
    return corpus


def load_text_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device='cpu')


def load_image_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device='cpu')


def compute_text_embeddings(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Text embeddings'):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
        embeddings.append(emb)
    return np.vstack(embeddings) if embeddings else np.empty((0, 384), dtype=np.float32)


def download_image(url: str, timeout: int = 10) -> Optional[Image.Image]:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert('RGB')
        return img
    except Exception:
        return None


def compute_image_embeddings(model, df: pd.DataFrame, batch_size: int = 32) -> Tuple[np.ndarray, List[int]]:
    images: List[Image.Image] = []
    valid_indices: List[int] = []
    for idx, row in tqdm(list(df[['image_url']].itertuples(index=True)), desc='Downloading images'):
        url = row.image_url
        if isinstance(url, str) and url.startswith('http'):
            img = download_image(url)
            if img is not None:
                images.append(img)
                valid_indices.append(idx)

    embs: List[np.ndarray] = []
    for i in tqdm(range(0, len(images), batch_size), desc='Image embeddings'):
        batch_imgs = images[i:i+batch_size]
        emb = model.encode(batch_imgs, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        embs.append(emb)
    if not embs:
        return np.empty((0, 512), dtype=np.float32), []
    return np.vstack(embs), valid_indices


def build_faiss_index(vectors: np.ndarray):
    import faiss
    if vectors.size == 0:
        return None
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine if vectors are normalized
    index.add(vectors.astype(np.float32))
    return index


def save_faiss_index(index, path: str) -> None:
    if index is None:
        return
    import faiss
    faiss.write_index(index, path)


def load_faiss_index(path: str):
    import faiss
    return faiss.read_index(path)


def save_metadata(df: pd.DataFrame, out_dir: str) -> str:
    path = os.path.join(out_dir, 'metadata.csv')
    df.to_csv(path, index=False)
    return path


def cosine_top_k(index, query_vecs: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    if index is None or query_vecs.size == 0:
        return np.empty((0, top_k), dtype=np.int64), np.empty((0, top_k), dtype=np.float32)
    D, I = index.search(query_vecs.astype(np.float32), top_k)
    return I, D


def run_build(args: argparse.Namespace) -> None:
    safe_mkdir(args.out_dir)
    print('Reading data...')
    rows = read_amazon_jsonl_gz(args.data_path, max_rows=args.max_rows)
    df = normalize_products(rows)
    print(f'Total products: {len(df)}')

    # Text
    corpus = build_text_corpus(df)
    text_model = load_text_model(args.text_model)
    text_embs = compute_text_embeddings(text_model, corpus, batch_size=args.text_batch_size)
    print(f'Text embeddings shape: {text_embs.shape}')

    # Image
    image_model = load_image_model(args.image_model)
    image_embs, image_indices = compute_image_embeddings(image_model, df, batch_size=args.image_batch_size)
    print(f'Image embeddings shape: {image_embs.shape} for {len(image_indices)} images')

    # Build FAISS
    text_index = build_faiss_index(text_embs)
    image_index = build_faiss_index(image_embs)

    # Persist
    save_faiss_index(text_index, os.path.join(args.out_dir, 'faiss_text.index'))
    save_faiss_index(image_index, os.path.join(args.out_dir, 'faiss_image.index'))

    # Save metadata and alignment
    df_out = df.copy()
    # Map image row index to compact image embedding row
    df_out['image_row'] = -1
    for compact_row, df_idx in enumerate(image_indices):
        df_out.at[df_idx, 'image_row'] = compact_row

    save_metadata(df_out, args.out_dir)
    print('Done. Artifacts saved to', args.out_dir)


def load_artifacts(out_dir: str):
    meta_path = os.path.join(out_dir, 'metadata.csv')
    df = pd.read_csv(meta_path)
    text_index_path = os.path.join(out_dir, 'faiss_text.index')
    image_index_path = os.path.join(out_dir, 'faiss_image.index')

    text_index = load_faiss_index(text_index_path) if os.path.exists(text_index_path) else None
    image_index = load_faiss_index(image_index_path) if os.path.exists(image_index_path) else None
    return df, text_index, image_index


def print_results(df: pd.DataFrame, indices: np.ndarray, scores: np.ndarray, top_k: int) -> None:
    for rank in range(min(top_k, indices.shape[1] if indices.ndim == 2 else len(indices))):
        idx = indices[0, rank] if indices.ndim == 2 else indices[rank]
        score = scores[0, rank] if scores.ndim == 2 else scores[rank]
        row = df.iloc[idx]
        print(f"#{rank+1} score={score:.3f} ASIN={row.get('asin','')} | {row.get('title','')}")


def query_by_text(out_dir: str, text: str, text_model_name: str, top_k: int) -> None:
    df, text_index, _ = load_artifacts(out_dir)
    if text_index is None:
        print('Text index not found. Build first.')
        return
    model = load_text_model(text_model_name)
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    I, D = text_index.search(vec.astype(np.float32), top_k)
    print_results(df, I, D, top_k)


def query_by_asin(out_dir: str, asin: str, weight_text: float, weight_image: float, top_k: int, text_model_name: str) -> None:
    df, text_index, image_index = load_artifacts(out_dir)
    if text_index is None:
        print('Text index not found. Build first.')
        return
    # Locate product
    matches = df.index[df['asin'] == asin].tolist()
    if not matches:
        print('ASIN not found in metadata.')
        return
    idx = matches[0]

    # Prepare query vectors by re-encoding text and fetching image vector
    text = ' '.join([str(df.at[idx, 'title']) if pd.notna(df.at[idx, 'title']) else '',
                     str(df.at[idx, 'description']) if pd.notna(df.at[idx, 'description']) else '']).strip()
    t_model = load_text_model(text_model_name)
    q_text = t_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    I_text, D_text = text_index.search(q_text.astype(np.float32), top_k)

    I_img = None
    D_img = None
    if image_index is not None and weight_image > 0:
        image_row = int(df.at[idx, 'image_row']) if 'image_row' in df.columns else -1
        if image_row >= 0:
            # Query by the product's own image embedding directly
            I_img, D_img = image_index.search(np.ascontiguousarray(np.expand_dims(image_index.reconstruct(image_row), axis=0)), top_k)

    if I_img is None or D_img is None or weight_image == 0:
        print_results(df, I_text, D_text, top_k)
        return

    # Fuse scores (weighted sum of cosine similarities)
    fused_scores = {}
    for r in range(top_k):
        fused_scores[int(I_text[0, r])] = fused_scores.get(int(I_text[0, r]), 0.0) + float(weight_text * D_text[0, r])
        fused_scores[int(I_img[0, r])] = fused_scores.get(int(I_img[0, r]), 0.0) + float(weight_image * D_img[0, r])

    top = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    indices = np.array([i for i, _ in top], dtype=np.int64)
    scores = np.array([s for _, s in top], dtype=np.float32)
    print_results(df, indices, scores, top_k)


def query_by_image_path(out_dir: str, image_path: str, image_model_name: str, top_k: int) -> None:
    df, _, image_index = load_artifacts(out_dir)
    if image_index is None:
        print('Image index not found. Build first.')
        return
    model = load_image_model(image_model_name)
    img = Image.open(image_path).convert('RGB')
    vec = model.encode([img], convert_to_numpy=True, normalize_embeddings=True)
    I, D = image_index.search(vec.astype(np.float32), top_k)
    print_results(df, I, D, top_k)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build and query a multimodal recommender (Amazon data).')
    p.add_argument('--data_path', type=str, default='All_Beauty.jsonl.gz', help='Path to Amazon JSONL.GZ file')
    p.add_argument('--out_dir', type=str, default='artifacts', help='Directory to save artifacts')
    p.add_argument('--max_rows', type=int, default=None, help='Limit rows for quick runs')

    p.add_argument('--text_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--image_model', type=str, default='sentence-transformers/clip-ViT-B-32')
    p.add_argument('--text_batch_size', type=int, default=64)
    p.add_argument('--image_batch_size', type=int, default=32)

    # Query options
    p.add_argument('--query_text', type=str, default=None)
    p.add_argument('--query_asin', type=str, default=None)
    p.add_argument('--query_image_path', type=str, default=None)
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--weight_text', type=float, default=0.5)
    p.add_argument('--weight_image', type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.query_text or args.query_asin or args.query_image_path:
        if args.query_text:
            query_by_text(args.out_dir, args.query_text, args.text_model, args.top_k)
            return
        if args.query_asin:
            query_by_asin(args.out_dir, args.query_asin, args.weight_text, args.weight_image, args.top_k, args.text_model)
            return
        if args.query_image_path:
            query_by_image_path(args.out_dir, args.query_image_path, args.image_model, args.top_k)
            return
    else:
        run_build(args)


if __name__ == '__main__':
    main()



