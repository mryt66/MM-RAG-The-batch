"""
Standalone script to compute embeddings from the_batch_articles_min.json
and save them for fast loading in the Streamlit app.
"""

import json
import re
from math import ceil
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import requests
from io import BytesIO
from PIL import Image

# Config (must match app.py)
DATA_PATH = Path("data/the_batch_articles_min.json")
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
IMAGE_MODEL_NAME = "clip-ViT-B-32"
CHROMA_PATH = Path("chroma_db")
COLLECTION_NAME = "Article"
IMAGE_COLLECTION_NAME = "ArticleImages"
EMB_FILE = Path("embeddings/embeddings.npz")
META_FILE = Path("data/metadata.json")
IMAGE_EMB_FILE = Path("embeddings/image_embeddings.npz")
IMAGE_META_FILE = Path("data/image_metadata.json")


def is_sponsor_title(title):
    """Exclude sponsor articles like 'A MESSAGE FROM...'"""
    if not title:
        return False
    t = re.sub(r"\s+", " ", str(title)).strip().upper()
    return t.startswith("A MESSAGE FROM")


def fetch_image(url):
    """Download image from URL"""
    try:
        # Add User-Agent header to avoid 403 errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        r = requests.get(url, timeout=10, headers=headers)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print(f"    Failed to fetch {url}: {e}")
        return None


def main():
    print(f"Loading articles from {DATA_PATH}...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    articles = payload.get("articles", [])
    print(f"Found {len(articles)} articles")

    # Build index data
    texts = []
    ids = []
    metadatas = []
    for i, a in enumerate(articles):
        if is_sponsor_title(a.get("title")):
            continue
        text = (a.get("text") or "").strip()
        if not text:
            continue
        texts.append(text)
        ids.append(f"art_{i}")
        imgs = a.get("images") or []
        first_img = ""
        if isinstance(imgs, list) and imgs and isinstance(imgs[0], str):
            first_img = imgs[0]
        metadatas.append(
            {
                "title": (a.get("title") or "").strip(),
                "issue_url": a.get("issue_url") or "",
                "idx": i,
                "image0": first_img,
            }
        )

    if not texts:
        raise RuntimeError("No indexable articles")

    print(f"Indexing {len(texts)} articles (after filtering sponsors)...")

    # Load model
    print(f"Loading embedding model: {MODEL_NAME}...")
    try:
        model = SentenceTransformer(
            MODEL_NAME,
            model_kwargs={
                "attn_implementation": "flash_attention_2",
                "device_map": "auto",
            },
            tokenizer_kwargs={"padding_side": "left"},
        )
    except Exception:
        # Fallback to default init
        model = SentenceTransformer(MODEL_NAME)

    # Compute embeddings
    print("Computing embeddings...")
    batch_size = 128
    chunks = []
    total_batches = max(1, ceil(len(texts) / batch_size))
    for bi, start in enumerate(range(0, len(texts), batch_size), start=1):
        end = start + batch_size
        print(f"  Batch {bi}/{total_batches} ({start}-{min(end, len(texts))})")
        embs = model.encode(
            texts[start:end], convert_to_numpy=True, show_progress_bar=False
        )
        chunks.append(embs)
    vectors = np.vstack(chunks).astype("float32")
    print(f"✅ Computed {len(vectors)} embeddings")

    # Save to disk
    print(f"Saving embeddings to {EMB_FILE}...")
    np.savez_compressed(EMB_FILE, vectors=vectors, ids=np.array(ids, dtype="U"))

    print(f"Saving metadata to {META_FILE}...")
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "metadatas": metadatas, "model_name": MODEL_NAME}, f)

    # Build Chroma collection
    print(f"Building Chroma collection at {CHROMA_PATH}...")
    client = PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Deleted old collection")
    except Exception:
        pass
    col = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    col.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors.tolist())
    print(f"✅ Collection '{COLLECTION_NAME}' created with {len(vectors)} vectors")

    # ===== IMAGE EMBEDDINGS =====
    print("\n--- Computing Image Embeddings ---")
    img_ids = []
    img_metadatas = []
    img_data = []

    for i, a in enumerate(articles):
        if is_sponsor_title(a.get("title")):
            continue
        srcs = a.get("images") or []
        if not isinstance(srcs, list) or not srcs or not isinstance(srcs[0], str):
            continue
        img = fetch_image(srcs[0])
        if img is None:
            continue
        img_ids.append(f"img_{i}")
        img_metadatas.append(
            {
                "idx": i,
                "title": (a.get("title") or "").strip(),
                "issue_url": a.get("issue_url") or "",
                "image0": srcs[0],
            }
        )
        img_data.append(img)

    if not img_data:
        print("⚠️  No images found, skipping image index")
    else:
        print(f"Found {len(img_data)} images to index")

        # Load CLIP model on GPU
        print(f"Loading image embedding model: {IMAGE_MODEL_NAME}...")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        image_model = SentenceTransformer(IMAGE_MODEL_NAME, device=device)

        # Compute image embeddings
        print("Computing image embeddings...")
        img_batch_size = 32
        img_chunks = []
        img_total_batches = max(1, ceil(len(img_data) / img_batch_size))
        for bi, start in enumerate(range(0, len(img_data), img_batch_size), start=1):
            end = start + img_batch_size
            print(
                f"  Batch {bi}/{img_total_batches} ({start}-{min(end, len(img_data))})"
            )
            embs = image_model.encode(
                img_data[start:end], convert_to_numpy=True, show_progress_bar=False
            )
            img_chunks.append(embs)
        img_vectors = np.vstack(img_chunks).astype("float32")
        print(f"✅ Computed {len(img_vectors)} image embeddings")

        # Save to disk
        print(f"Saving image embeddings to {IMAGE_EMB_FILE}...")
        np.savez_compressed(
            IMAGE_EMB_FILE, vectors=img_vectors, ids=np.array(img_ids, dtype="U")
        )

        print(f"Saving image metadata to {IMAGE_META_FILE}...")
        with open(IMAGE_META_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ids": img_ids,
                    "metadatas": img_metadatas,
                    "model_name": IMAGE_MODEL_NAME,
                },
                f,
            )

        # Build Chroma image collection
        print(f"Building image collection '{IMAGE_COLLECTION_NAME}'...")
        try:
            client.delete_collection(IMAGE_COLLECTION_NAME)
            print("  Deleted old image collection")
        except Exception:
            pass
        img_col = client.create_collection(
            IMAGE_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        img_col.add(
            ids=img_ids, metadatas=img_metadatas, embeddings=img_vectors.tolist()
        )
        print(f"✅ Image collection created with {len(img_vectors)} vectors")

    print(
        "\n✅ Done! Both text and image indexes are ready. Run the Streamlit app to use them."
    )


if __name__ == "__main__":
    main()
