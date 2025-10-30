## Multimodal Amazon Product Recommender (Text + Image)

This project builds a simple multimodal recommendation system using Amazon product metadata and images. It computes:

- Text embeddings from product titles/descriptions
- Image embeddings from product image URLs

Both are indexed with FAISS for fast nearest-neighbor search. You can query by text, by product ASIN, or by an image.

### Prerequisites
- Python 3.9+
- Windows or Linux/macOS
- File: `All_Beauty.jsonl.gz` in the project root (as provided)

### Install
```bash
pip install -r requirements.txt
```

### Build indexes
```bash
python build_recommender.py --data_path All_Beauty.jsonl.gz --out_dir artifacts
```

Key options:
- `--text_model` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `--image_model` (default: `sentence-transformers/clip-ViT-B-32`)
- `--max_rows` to limit rows for a quick run

### Query examples
By text:
```bash
python build_recommender.py --out_dir artifacts --query_text "hydrating facial cleanser" --top_k 10
```

By ASIN (find similar items):
```bash
python build_recommender.py --out_dir artifacts --query_asin B00EXAMPLE --top_k 10
```

By image path:
```bash
python build_recommender.py --out_dir artifacts --query_image_path path/to/image.jpg --top_k 10
```

### Outputs
Artifacts are saved in `artifacts/`:
- `faiss_text.index`, `faiss_image.index`
- `metadata.csv` (ASIN, title, description, image_url, category)
- `text_embeddings.npy` (optional), `image_embeddings.npy` (optional, if enabled)

### Notes
- If some products lack images, they are skipped for the image index but still included in the text index.
- Everything runs on CPU by default.


