import os
import json
import re
import time
from math import ceil
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import streamlit as st
import google.generativeai as genai
import torch
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from dataclasses import dataclass
import requests
from io import BytesIO
from PIL import Image
from rag.retrieval import (
    retrieve_relevant_articles as ext_retrieve,
    retrieve_multimodal_articles as ext_retrieve_mm,
    construct_prompt as ext_construct_prompt,
)
from evaluation.database_logger import log_interaction, auto_evaluate_interaction, get_interaction_evaluation

# ---------- Config ----------
DATA_PATH = Path("data/the_batch_articles_min.json")
CHROMA_PATH = Path("chroma_db")
COLLECTION_NAME = "Article"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMB_FILE = Path("embeddings/embeddings.npz")  # stores vectors (float32) and ids (unicode)
META_FILE = Path("data/metadata.json")  # stores minimal metadata + idx mapping

# Image indexing config (CLIP for images)
IMAGE_COLLECTION_NAME = "ArticleImages"
IMAGE_MODEL_NAME = "clip-ViT-B-32"
IMAGE_EMB_FILE = Path("embeddings/image_embeddings.npz")
IMAGE_META_FILE = Path("data/image_metadata.json")


# ---------- Helpers ----------


def check_gpu_available() -> bool:
    """Check if GPU is available and display status"""
    is_available = torch.cuda.is_available()
    if is_available:
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        st.success(f"🚀 GPU Detected: {gpu_name} (CUDA {cuda_version})")
    else:
        st.info("💻 Using CPU for computations (GPU not available)")
    return is_available


@st.cache_resource(show_spinner=False)
def get_model() -> SentenceTransformer:
    # Load model with optimized settings for faster inference on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = SentenceTransformer(MODEL_NAME, device=device)
        # Set to evaluation mode and disable gradients for faster inference
        model.eval()
        return model
    except Exception:
        # Fallback to default init
        return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_image_model() -> SentenceTransformer:
    # CLIP model for image embeddings on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(IMAGE_MODEL_NAME, device=device)
    model.eval()
    return model


def load_articles() -> List[Dict[str, Any]]:
    if not DATA_PATH.exists():
        st.error(f"Missing data file: {DATA_PATH.resolve()}")
        st.stop()
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("articles", [])


def configure_gemini(api_key: Optional[str]) -> bool:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return False
    genai.configure(api_key=key)
    return True


def get_gemini_answer(prompt: str) -> tuple:
    """
    Generate answer using Gemini and return answer with token usage
    Returns: (answer_text, token_dict)
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        
        # Extract token usage
        tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if hasattr(resp, 'usage_metadata'):
            usage = resp.usage_metadata
            tokens["input_tokens"] = getattr(usage, 'prompt_token_count', 0)
            tokens["output_tokens"] = getattr(usage, 'candidates_token_count', 0)
            tokens["total_tokens"] = getattr(usage, 'total_token_count', 0)
        
        return resp.text, tokens
    except Exception as e:
        st.error(f"Gemini error: {e}")
        return None, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def redact_emails(text: str) -> str:
    return EMAIL_RE.sub("[REDACTED_EMAIL]", text or "")


def is_sponsor_title(title: Optional[str]) -> bool:
    """Heuristic: exclude sponsor/partner ads like 'A MESSAGE FROM ...'.
    Matches case-insensitively and tolerates odd spacing.
    """
    if not title:
        return False
    t = re.sub(r"\s+", " ", str(title)).strip().upper()
    return t.startswith("A MESSAGE FROM")


# ---------- RAG (Chroma) structured API ----------


@dataclass
class RAGState:
    articles: List[Dict[str, Any]]
    model: SentenceTransformer
    chroma_path: Path = CHROMA_PATH
    collection_name: str = COLLECTION_NAME


def initialize_system_chroma(
    *,
    compute_text: bool = False,
    compute_images: bool = False,
    show_progress: bool = False,
) -> Optional[RAGState]:
    """Ensure the Chroma collection is prepared; return a RAGState using Chroma."""
    arts = load_articles()

    # Only prepare text index if computing text or loading everything
    if compute_text or not compute_images:
        collection = prepare_index(arts, compute_if_missing=compute_text)
        if collection is None and not compute_images:
            return None

    # Build image index ONLY if explicitly requested
    if compute_images:
        try:
            prepare_image_index(arts, compute_if_missing=True)
        except Exception:
            pass  # Silently fail, image index is optional
    # Don't try to load image index when computing text embeddings
    elif not compute_text:
        # Only load existing image index when using "Load Index" button
        try:
            prepare_image_index(arts, compute_if_missing=False)
        except Exception:
            pass  # Silently skip if no image index exists

    result = RAGState(articles=arts, model=get_model())

    return result


def retrieve_relevant_articles(
    state: RAGState,
    query: str,
    top_k: int = 5,
    *,
    exclude_idxs: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    hits = ext_retrieve(state, query, top_k=top_k, exclude_idxs=exclude_idxs)
    # Drop the extra 'idx' before returning to UI if present
    return [{"distance": h.get("distance"), "article": h.get("article")} for h in hits]


def construct_prompt(
    state: RAGState, query: str, history_text: str = ""
) -> tuple[str, str]:
    return ext_construct_prompt(state, query, history_text)


def _get_rag_state() -> Optional[RAGState]:
    state = st.session_state.get("rag_state")
    return state if isinstance(state, RAGState) else None


def prepare_index(articles: List[Dict[str, Any]], *, compute_if_missing: bool):
    # Build rows
    texts: List[str] = []
    ids: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for i, a in enumerate(articles):
        # Skip sponsor/partner items
        if is_sponsor_title(a.get("title")):
            continue
        text = (a.get("text") or "").strip()
        if not text:
            continue
        texts.append(text)
        ids.append(f"art_{i}")
        first_img = ""
        try:
            imgs = a.get("images") or []
            if isinstance(imgs, list) and imgs and isinstance(imgs[0], str):
                first_img = imgs[0]
        except Exception:
            first_img = ""
        metadatas.append(
            {
                "title": (a.get("title") or "").strip(),
                "issue_url": a.get("issue_url") or "",
                "idx": i,
                "image0": first_img,
            }
        )

    if not texts:
        return None  # No indexable articles

    # Load embeddings from disk if present and model matches
    vectors = None
    saved_ids_emb = None
    try:
        if EMB_FILE.exists():
            with np.load(EMB_FILE, allow_pickle=False) as npz:
                saved_ids_emb = npz["ids"].astype(str).tolist()
                # probe file readability; vectors will be loaded later once validated
                _ = npz["vectors"]
    except Exception:
        pass  # Silently fail, will compute if needed

    saved_ids_meta = None
    saved_model_name = None
    if META_FILE.exists():
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                meta_payload = json.load(f)
                saved_ids_meta = meta_payload.get("ids")
                saved_model_name = meta_payload.get("model_name")
        except Exception:
            pass  # Silently fail, will compute if needed

    if (
        saved_ids_emb == ids
        and saved_ids_meta == ids
        and saved_model_name == MODEL_NAME
    ):
        try:
            with np.load(EMB_FILE, allow_pickle=False) as npz:
                vectors = npz["vectors"].astype("float32")
        except Exception:
            pass  # Silently fail, will compute if needed
    # Compute embeddings if missing and allowed
    if vectors is None:
        if not compute_if_missing:
            return None

        # Check GPU availability before starting
        check_gpu_available()

        # Clear GPU cache before starting to free any lingering memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = get_model()
        batch_size = 16  # Small batch size to avoid GPU OOM on 12GB GPU
        chunks = []
        total = len(texts)
        total_batches = max(1, ceil(total / batch_size))

        # Initialize stop button state
        if "stop_text_embedding" not in st.session_state:
            st.session_state.stop_text_embedding = False

        start_time = time.time()
        with st.status(
            f"Computing text embeddings… 0 / {total} articles", expanded=True
        ) as status:
            # Add stop button inside the status
            stop_btn_placeholder = st.empty()
            progress = st.progress(0)

            for bi, start in enumerate(range(0, total, batch_size), start=1):
                # Check if stop was requested
                if st.session_state.stop_text_embedding:
                    status.update(
                        label=f"⏹️ Stopped at {len(chunks) * batch_size} / {total} articles",
                        state="error",
                    )
                    st.session_state.stop_text_embedding = False
                    return None

                # Show stop button
                if stop_btn_placeholder.button(
                    "⏹️ Stop", key=f"stop_text_{bi}", type="secondary"
                ):
                    st.session_state.stop_text_embedding = True
                    st.rerun()

                end = min(start + batch_size, total)
                with torch.no_grad():  # Disable gradient computation to save memory
                    embs = model.encode(
                        texts[start:end],
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=batch_size,
                    )
                chunks.append(embs)

                # Clear GPU cache after each batch to prevent memory buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                pct = min(100, int(bi / total_batches * 100))
                progress.progress(pct)

                # Calculate time estimates
                elapsed = time.time() - start_time
                if bi > 1:  # After first batch, estimate remaining time
                    avg_time_per_batch = elapsed / bi
                    remaining_batches = total_batches - bi
                    eta_seconds = avg_time_per_batch * remaining_batches
                    eta_str = (
                        f" - ETA: {int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                    )
                else:
                    eta_str = ""

                status.update(
                    label=f"Computing text embeddings… {end} / {total} articles ({pct}%) - Batch {bi}/{total_batches}{eta_str}",
                    state="running",
                )

            # Clear stop button when done
            stop_btn_placeholder.empty()
            st.session_state.stop_text_embedding = False

            vectors = np.vstack(chunks).astype("float32")
            total_time = time.time() - start_time
            status.update(
                label=f"✅ Text embeddings complete - {total} articles in {int(total_time // 60)}m {int(total_time % 60)}s",
                state="complete",
            )
        # Save to disk (ids as Unicode to avoid pickle) and include model_name for compatibility checks
        np.savez_compressed(EMB_FILE, vectors=vectors, ids=np.array(ids, dtype="U"))
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump({"ids": ids, "metadatas": metadatas, "model_name": MODEL_NAME}, f)
        st.success(f"Saved embeddings to {EMB_FILE} and metadata to {META_FILE}.")

    # Upsert to Chroma
    client = PersistentClient(path=str(CHROMA_PATH))
    
    # Check if collection already exists with correct data (only when loading, not computing)
    if not compute_if_missing:
        try:
            col = client.get_collection(COLLECTION_NAME)
            # Check if collection has the right number of items
            if col.count() == len(vectors):
                return col
        except Exception:
            pass  # Collection doesn't exist, create it
    
    # Rebuild collection (when computing or collection doesn't exist/match)
    with st.status(f"📊 Building text index in ChromaDB ({len(vectors)} vectors)...", expanded=False) as status:
        # Drop entire collection to avoid stale vectors
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        # Create a fresh collection after deletion
        col = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        status.update(label="💾 Inserting vectors into ChromaDB...", state="running")
        col.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors.tolist())
        status.update(label=f"✅ Text index ready ({len(vectors)} vectors)", state="complete")
    return col


# -------- Image Index (CLIP) --------
def _fetch_image(url: str) -> Optional[Image.Image]:
    """Download and return image from URL"""
    try:
        # Add User-Agent header to avoid 403 errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        r = requests.get(url, timeout=10, headers=headers)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def prepare_image_index(articles: List[Dict[str, Any]], *, compute_if_missing: bool):
    """Build image index with CLIP embeddings, save/load from disk like text embeddings"""
    ids: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    images: List[Image.Image] = []

    # Only download images if we need to compute embeddings
    if compute_if_missing:
        with st.status("📥 Downloading images from articles...", expanded=False) as download_status:
            for i, a in enumerate(articles):
                if is_sponsor_title(a.get("title")):
                    continue
                srcs = a.get("images") or []
                if not isinstance(srcs, list) or not srcs or not isinstance(srcs[0], str):
                    continue
                img = _fetch_image(srcs[0])
                if img is None:
                    continue
                ids.append(f"img_{i}")
                metadatas.append(
                    {
                        "idx": i,
                        "title": (a.get("title") or "").strip(),
                        "issue_url": a.get("issue_url") or "",
                        "image0": srcs[0],
                    }
                )
                images.append(img)
                
                # Update progress every 50 images
                if len(images) % 50 == 0:
                    download_status.update(label=f"📥 Downloaded {len(images)} images...", state="running")
            
            download_status.update(label=f"✅ Downloaded {len(images)} images", state="complete")
    
    # When loading (not computing), try to load from disk first
    vectors = None
    if not compute_if_missing:
        try:
            if IMAGE_EMB_FILE.exists() and IMAGE_META_FILE.exists():
                with np.load(IMAGE_EMB_FILE, allow_pickle=False) as npz:
                    vectors = npz["vectors"].astype("float32")
                    ids = npz["ids"].astype(str).tolist()
                
                with open(IMAGE_META_FILE, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                    metadatas = payload.get("metadatas", [])
                    saved_model_name = payload.get("model_name")
                
                if saved_model_name != IMAGE_MODEL_NAME:
                    return None
        except Exception as e:
            return None  # No saved embeddings found, return None
    
    # If loading from disk failed or we're computing, check if we have images
    if vectors is None and not images:
        return None
    
    # Try to load saved embeddings if computing and compatible
    if compute_if_missing and vectors is None:
        saved_ids_emb = None
        try:
            if IMAGE_EMB_FILE.exists():
                with np.load(IMAGE_EMB_FILE, allow_pickle=False) as npz:
                    saved_ids_emb = npz["ids"].astype(str).tolist()
                    _ = npz["vectors"]
        except Exception:
            pass

        saved_ids_meta = None
        saved_model_name = None
        if IMAGE_META_FILE.exists():
            try:
                with open(IMAGE_META_FILE, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                    saved_ids_meta = payload.get("ids")
                    saved_model_name = payload.get("model_name")
            except Exception:
                pass

        # Load if compatible
        if (
            saved_ids_emb == ids
            and saved_ids_meta == ids
            and saved_model_name == IMAGE_MODEL_NAME
        ):
            try:
                with np.load(IMAGE_EMB_FILE, allow_pickle=False) as npz:
                    vectors = npz["vectors"].astype("float32")
            except Exception:
                pass

    # Compute if missing
    if vectors is None:
        if not compute_if_missing:
            return None

        # Check GPU availability before starting
        check_gpu_available()

        # Clear GPU cache before starting to free any lingering memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        imodel = get_image_model()
        batch = 32  # Batch size for image embeddings on 12GB GPU
        chunks = []
        total = len(images)
        total_batches = max(1, ceil(total / batch))

        # Initialize stop button state
        if "stop_image_embedding" not in st.session_state:
            st.session_state.stop_image_embedding = False

        start_time = time.time()
        with st.status(
            f"Computing image embeddings… 0 / {total} images", expanded=True
        ) as status:
            # Add stop button inside the status
            stop_btn_placeholder = st.empty()
            progress = st.progress(0)

            for bi, start in enumerate(range(0, total, batch), start=1):
                # Check if stop was requested
                if st.session_state.stop_image_embedding:
                    status.update(
                        label=f"⏹️ Stopped at {len(chunks) * batch} / {total} images",
                        state="error",
                    )
                    st.session_state.stop_image_embedding = False
                    return None

                # Show stop button
                if stop_btn_placeholder.button(
                    "⏹️ Stop", key=f"stop_image_{bi}", type="secondary"
                ):
                    st.session_state.stop_image_embedding = True
                    st.rerun()

                end = min(start + batch, total)
                with torch.no_grad():  # Disable gradient computation to save memory
                    embs = imodel.encode(
                        images[start:end],
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=batch,
                    )
                chunks.append(embs)

                # Clear GPU cache after each batch to prevent memory buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                pct = min(100, int(bi / total_batches * 100))
                progress.progress(pct)

                # Calculate time estimates
                elapsed = time.time() - start_time
                if bi > 1:  # After first batch, estimate remaining time
                    avg_time_per_batch = elapsed / bi
                    remaining_batches = total_batches - bi
                    eta_seconds = avg_time_per_batch * remaining_batches
                    eta_str = (
                        f" - ETA: {int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                    )
                else:
                    eta_str = ""

                status.update(
                    label=f"Computing image embeddings… {end} / {total} images ({pct}%) - Batch {bi}/{total_batches}{eta_str}",
                    state="running",
                )

            # Clear stop button when done
            stop_btn_placeholder.empty()
            st.session_state.stop_image_embedding = False

            vectors = np.vstack(chunks).astype("float32")
            total_time = time.time() - start_time
            status.update(
                label=f"✅ Image embeddings complete - {total} images in {int(total_time // 60)}m {int(total_time % 60)}s",
                state="complete",
            )

        # Save to disk
        np.savez_compressed(
            IMAGE_EMB_FILE, vectors=vectors, ids=np.array(ids, dtype="U")
        )
        with open(IMAGE_META_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"ids": ids, "metadatas": metadatas, "model_name": IMAGE_MODEL_NAME}, f
            )
        st.success(
            f"Saved image embeddings to {IMAGE_EMB_FILE} and metadata to {IMAGE_META_FILE}."
        )

    # Upsert to Chroma
    client = PersistentClient(path=str(CHROMA_PATH))
    
    # Check if collection already exists with correct data (only when loading, not computing)
    if not compute_if_missing:
        try:
            col = client.get_collection(IMAGE_COLLECTION_NAME)
            # Check if collection has the right number of items
            if col.count() == len(vectors):
                return col
        except Exception:
            pass  # Collection doesn't exist, create it
    
    # Rebuild collection (when computing or collection doesn't exist/match)
    with st.status(f"📊 Building image index in ChromaDB ({len(vectors)} vectors)...", expanded=False) as status:
        client = PersistentClient(path=str(CHROMA_PATH))
        try:
            client.delete_collection(IMAGE_COLLECTION_NAME)
        except Exception:
            pass
        col = client.create_collection(
            IMAGE_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        status.update(label="💾 Inserting image vectors into ChromaDB...", state="running")
        col.add(ids=ids, metadatas=metadatas, embeddings=vectors.tolist())
        status.update(label=f"✅ Image index ready ({len(vectors)} vectors)", state="complete")
    return col


# ---------- UI ----------
st.set_page_config(page_title="Chat with The Batch", page_icon="🔎", layout="wide")

# Auto-evaluate unevaluated interactions on startup (runs once per session)
if "auto_eval_done" not in st.session_state:
    try:
        from evaluation.database_logger import get_unevaluated_interactions, evaluate_all_unevaluated
        unevaluated = get_unevaluated_interactions()
        if unevaluated and len(unevaluated) > 0:
            with st.spinner(f"🔄 Auto-evaluating {len(unevaluated)} interactions..."):
                evaluate_all_unevaluated()
            st.toast(f"✅ Evaluated {len(unevaluated)} interactions!", icon="🎉")
    except Exception as e:
        pass  # Silently fail if evaluation not available
    st.session_state["auto_eval_done"] = True

# Title
st.title("Chat with The Batch")

# Controls row under title
btn_col1, btn_col2, btn_col3, _sp = st.columns([0.14, 0.20, 0.22, 0.44])
with btn_col1:
    if st.button("Load Index", use_container_width=True):
        with st.status("🔄 Loading index...", expanded=False) as status:
            status.update(label="� Loading articles...", state="running")
            state = initialize_system_chroma(show_progress=False)
        
        st.session_state["collection_ready"] = state is not None
        if state is not None:
            st.toast("✅ Index loaded successfully!", icon="🎉")
        else:
            st.toast("⚠️ Index not found. Use Compute buttons.", icon="⚠️")

with btn_col2:
    if st.button("Compute Text Embeddings", use_container_width=True):
        # Check if embeddings already exist
        if EMB_FILE.exists():
            if "confirm_text_overwrite" not in st.session_state:
                st.session_state["confirm_text_overwrite"] = False

            @st.dialog("⚠️ Overwrite Text Embeddings?")
            def confirm_text_dialog():
                st.write(
                    "Text embeddings already exist. Do you want to recompute and overwrite them?"
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "✅ Yes, Overwrite", use_container_width=True, type="primary"
                    ):
                        st.session_state["confirm_text_overwrite"] = True
                        st.rerun()
                with col2:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.session_state["confirm_text_overwrite"] = False
                        st.rerun()

            if not st.session_state.get("confirm_text_overwrite"):
                confirm_text_dialog()
            else:
                st.session_state["confirm_text_overwrite"] = False
                state = initialize_system_chroma(compute_text=True, show_progress=True)
                st.session_state["collection_ready"] = state is not None
                if state is not None:
                    st.toast("Text embeddings computed!", icon="📝")
        else:
            state = initialize_system_chroma(compute_text=True, show_progress=True)
            st.session_state["collection_ready"] = state is not None
            if state is not None:
                st.toast("Text embeddings computed!", icon="📝")

with btn_col3:
    if st.button("Compute Image Embeddings", use_container_width=True):
        # Check if embeddings already exist
        if IMAGE_EMB_FILE.exists():
            if "confirm_image_overwrite" not in st.session_state:
                st.session_state["confirm_image_overwrite"] = False

            @st.dialog("⚠️ Overwrite Image Embeddings?")
            def confirm_image_dialog():
                st.write(
                    "Image embeddings already exist. Do you want to recompute and overwrite them?"
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "✅ Yes, Overwrite", use_container_width=True, type="primary"
                    ):
                        st.session_state["confirm_image_overwrite"] = True
                        st.rerun()
                with col2:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.session_state["confirm_image_overwrite"] = False
                        st.rerun()

            if not st.session_state.get("confirm_image_overwrite"):
                confirm_image_dialog()
            else:
                st.session_state["confirm_image_overwrite"] = False
                state = initialize_system_chroma(
                    compute_images=True, show_progress=True
                )
                st.session_state["collection_ready"] = state is not None
                if state is not None:
                    st.toast("Image embeddings computed!", icon="🖼️")
        else:
            state = initialize_system_chroma(compute_images=True, show_progress=True)
            st.session_state["collection_ready"] = state is not None
            if state is not None:
                st.toast("Image embeddings computed!", icon="🖼️")

articles = load_articles()
st.caption(f"Index contains {len(articles)} total articles.")

# Initialize session state (no auto-load)
if "collection_ready" not in st.session_state:
    st.session_state["collection_ready"] = False

# Auto-load index if embeddings exist and not already loaded
if "auto_loaded" not in st.session_state:
    st.session_state["auto_loaded"] = False

if not st.session_state["auto_loaded"] and EMB_FILE.exists():
    try:
        state = initialize_system_chroma(show_progress=False)
        st.session_state["collection_ready"] = state is not None
        if state is not None:
            st.toast("✅ Index loaded automatically", icon="🚀")
        st.session_state["auto_loaded"] = True
    except Exception:
        st.session_state["auto_loaded"] = True

# ----- Chatbot UI -----
left_col, right_col = st.columns([0.62, 0.38])


def retrieve_hits(question: str, k: int = 5) -> List[Dict[str, Any]]:
    model = get_model()
    # For Qwen3 embedding models, queries benefit from the "query" prompt
    try:
        qv = model.encode([question], prompt_name="query", convert_to_numpy=True)[0]
    except Exception:
        qv = model.encode([question], convert_to_numpy=True)[0]
    client = PersistentClient(path=str(CHROMA_PATH))
    col = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    res = col.query(
        query_embeddings=[qv.tolist()],
        n_results=max(10, k * 3),
        include=["metadatas", "distances"],
    )  # noqa: E501
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    hits: List[Dict[str, Any]] = []
    seen_idxs = set()
    for meta, dist in zip(metas, dists):
        idx = meta.get("idx") if isinstance(meta, dict) else None
        if isinstance(idx, int) and 0 <= idx < len(articles) and idx not in seen_idxs:
            # Skip sponsor items defensively
            if is_sponsor_title(articles[idx].get("title")):
                continue
            seen_idxs.add(idx)
            hits.append(
                {
                    "distance": float(dist) if dist is not None else None,
                    "article": articles[idx],
                }
            )
    # Return top-k unique
    return hits[:k]


with left_col:
    if not st.session_state.get("collection_ready"):
        st.info("Build or load the index, then ask a question.")
    else:
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        if "session_id" not in st.session_state:
            import uuid
            st.session_state["session_id"] = str(uuid.uuid4())[:8]  # Short session ID
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                # Display text content (for assistant, just the answer without sources)
                if msg["role"] == "assistant":
                    # Split off sources section if present
                    content = msg["content"]
                    if "\n\n**📚 Sources:**\n" in content:
                        answer_part = content.split("\n\n**📚 Sources:**\n")[0]
                        st.write(answer_part)
                    else:
                        st.write(content)
                    
                    # Display articles with relevance scores if they exist
                    if "hits" in msg and msg["hits"]:
                        st.markdown("---")
                        st.markdown("**📚 Sources:**")
                        for i, h in enumerate(msg["hits"][:3], start=1):
                            article = h.get("article", {})
                            title = article.get("title", "(no title)")
                            url = article.get("issue_url", "")
                            distance = h.get("distance")
                            
                            st.markdown(f"{i}. [{title}]({url})")
                else:
                    # User messages - display as is
                    st.write(msg["content"])

        # Image upload option - integrates directly with chat like GitHub Copilot
        uploaded_img = st.file_uploader(
            "📎 Attach image",
            type=["png", "jpg", "jpeg"],
            help="Attach an image to your query. System automatically adjusts text/image weighting based on similarity quality.",
        )
        
        # Show status when image is attached
        if uploaded_img is not None:
            st.success("✅ Image attached! Type your question or just a space (' ') for image-only search.")
        
        user_msg = st.chat_input("Type your question (or space for image-only)…" if uploaded_img else "Attach an image or type your question…")
        
        # Process if user sends message
        if user_msg is not None:
            # Add message to chat
            actual_content = user_msg.strip()
            # Check if it's just whitespace/space for image-only search
            if actual_content == "" or actual_content == " ":
                # Image-only search (user typed just space or whitespace)
                if uploaded_img is not None:
                    st.session_state["messages"].append({"role": "user", "content": "[Image search]"})
            else:
                # Regular text message
                st.session_state["messages"].append({"role": "user", "content": user_msg})

            # Show uploaded image if provided
            pil_img = None
            if uploaded_img is not None:
                try:
                    pil_img = Image.open(uploaded_img).convert("RGB")
                    with st.chat_message("user"):
                        st.image(pil_img, caption=uploaded_img.name, width=300)
                except Exception as e:
                    st.error(f"Failed to load image: {e}")

            # Build a fresh RAG state on demand to avoid stale session issues
            state = RAGState(articles=articles, model=get_model())
            # Make right-panel results more diverse: exclude last shown idxs
            last_idxs = set()
            last_hits = st.session_state.get("last_hits") or []
            for h in last_hits:
                a = h.get("article") or {}
                try:
                    last_idx = articles.index(a)
                    last_idxs.add(last_idx)
                except ValueError:
                    pass
            # Show a thinking status with changing labels
            with st.status("Thinking…", expanded=False) as status:
                status.update(label="Thinking…", state="running")

                # Automatically determine search mode based on inputs
                # Filter out single space used for image-only search
                actual_text = user_msg.strip() if user_msg else ""
                has_text = actual_text and actual_text != " " and len(actual_text) > 0
                has_image = pil_img is not None
                
                if has_image and has_text:
                    # Both text and image provided -> Multimodal search with smart weighting
                    status.update(label="🔍 Multimodal search (smart weighting)…", state="running")
                    mm_hits = ext_retrieve_mm(
                        state,
                        text_query=user_msg,
                        image=pil_img,
                        top_k=3,
                        exclude_idxs=last_idxs,
                        alpha_text=None,  # Auto-compute based on similarity quality
                    )
                    all_hits = [
                        {"distance": h.get("distance"), "article": h.get("article")}
                        for h in mm_hits
                    ]
                elif has_image and not has_text:
                    # Only image provided -> Image-only search
                    status.update(label="🖼️ Image-only search…", state="running")
                    mm_hits = ext_retrieve_mm(
                        state,
                        text_query=None,
                        image=pil_img,
                        top_k=3,
                        exclude_idxs=last_idxs,
                        alpha_text=0.0,  # 100% image weight
                    )
                    all_hits = [
                        {"distance": h.get("distance"), "article": h.get("article")}
                        for h in mm_hits
                    ]
                elif has_text and not has_image:
                    # Only text provided -> Text-only search
                    status.update(label="📝 Text-only search…", state="running")
                    all_hits = retrieve_relevant_articles(
                        state, user_msg, top_k=3, exclude_idxs=last_idxs
                    )
                else:
                    # Neither text nor image (shouldn't happen, but handle it)
                    st.error("⚠️ Please provide text or upload an image to search.")
                    all_hits = []

                st.session_state["last_hits"] = all_hits
                status.update(label="Retrieving context…", state="running")
                
                # Build conversation history for context
                history_messages = []
                for msg in st.session_state.get("messages", [])[-6:]:  # Last 3 exchanges (6 messages)
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        history_messages.append(f"User: {content}")
                    elif role == "assistant":
                        # Extract just the answer part (without sources)
                        if "\n\n**📚 Sources:**\n" in content:
                            answer_part = content.split("\n\n**📚 Sources:**\n")[0]
                        else:
                            answer_part = content
                        history_messages.append(f"Assistant: {answer_part}")
                
                history_text = "\n".join(history_messages) if history_messages else ""
                
                # Build the prompt using the RAGState API
                # For image-only mode without text, use a default query
                query_for_prompt = user_msg if user_msg else "What can you tell me about these articles?"
                prompt, _ctx = construct_prompt(state, query_for_prompt, history_text=history_text)
                configured = configure_gemini(None)
                status.update(label="Composing answer…", state="running")
                
                # Get answer and token usage
                if configured:
                    answer, token_usage = get_gemini_answer(prompt)
                else:
                    answer = "(no LLM configured)"
                    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                
                status.update(label="Done", state="complete")
            
            # Log interaction to database
            try:
                # Prepare contexts for logging
                contexts_for_log = []
                for h in all_hits[:3]:
                    article = h.get("article", {})
                    contexts_for_log.append({
                        "title": article.get("title", ""),
                        "text": article.get("text", "")[:500]  # Truncate for storage
                    })
                
                # Log the interaction with token usage
                interaction_id = log_interaction(
                    question=query_for_prompt,
                    answer=answer or "(no response)",
                    contexts=contexts_for_log,
                    session_id=st.session_state.get("session_id", "default"),
                    token_usage=token_usage
                )
                
                # Auto-evaluate (this will compute metrics)
                status.update(label="🔍 Evaluating response quality...", state="running")
                eval_metrics = auto_evaluate_interaction(interaction_id)
                
                status.update(label="✅ Evaluation complete", state="complete")
                
                # Store interaction_id for displaying metrics
                st.session_state["last_interaction_id"] = interaction_id
                st.session_state["last_eval_metrics"] = eval_metrics
                
            except Exception as e:
                st.warning(f"Note: Failed to log to database: {e}")
                st.session_state["last_interaction_id"] = None
                st.session_state["last_eval_metrics"] = None
            
            # Format assistant response with articles and relevance scores
            response_content = answer or "(no response)"
            
            # Add articles section with relevance scores
            if all_hits:
                response_content += "\n\n**📚 Sources:**\n"
                for i, h in enumerate(all_hits[:3], start=1):
                    article = h.get("article", {})
                    title = article.get("title", "(no title)")
                    url = article.get("issue_url", "")
                    distance = h.get("distance")
                    
                    # Add article link
                    response_content += f"\n{i}. [{title}]({url})"
            
            st.session_state["messages"].append(
                {"role": "assistant", "content": response_content, "hits": all_hits}
            )
            with st.chat_message("assistant"):
                st.write(answer or "(no response)")
                
                # Display articles
                if all_hits:
                    st.markdown("---")
                    st.markdown("**📚 Sources:**")
                    for i, h in enumerate(all_hits[:3], start=1):
                        article = h.get("article", {})
                        title = article.get("title", "(no title)")
                        url = article.get("issue_url", "")
                        
                        # Display article without relevance score
                        st.markdown(f"{i}. [{title}]({url})")
                
                # Display evaluation metrics if available
                eval_metrics = st.session_state.get("last_eval_metrics")
                if eval_metrics:
                    st.markdown("---")
                    st.markdown("**📊 Evaluation Metrics:**")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        cr_score = eval_metrics.get("context_relevance", 0)
                        cr_color = "🟢" if cr_score >= 0.7 else "🟡" if cr_score >= 0.4 else "🔴"
                        st.metric(
                            label=f"{cr_color} Context Relevance",
                            value=f"{cr_score:.2f}",
                            help="How relevant are the retrieved contexts to the question?"
                        )
                    with col2:
                        g_score = eval_metrics.get("groundedness", 0)
                        g_color = "🟢" if g_score >= 0.7 else "🟡" if g_score >= 0.4 else "🔴"
                        st.metric(
                            label=f"{g_color} Groundedness",
                            value=f"{g_score:.2f}",
                            help="Is the answer supported by the contexts?"
                        )
                    with col3:
                        ar_score = eval_metrics.get("answer_relevance", 0)
                        ar_color = "🟢" if ar_score >= 0.7 else "🟡" if ar_score >= 0.4 else "🔴"
                        st.metric(
                            label=f"{ar_color} Answer Relevance",
                            value=f"{ar_score:.2f}",
                            help="How relevant is the answer to the question?"
                        )
                    
                    # Show token usage if available
                    interaction_id = st.session_state.get("last_interaction_id")
                    if interaction_id:
                        try:
                            full_eval = get_interaction_evaluation(interaction_id)
                            if full_eval and full_eval.get("total_tokens"):
                                st.caption(f"🔢 Tokens: {full_eval['total_tokens']} (input: {full_eval['input_tokens']}, output: {full_eval['output_tokens']})")
                        except:
                            pass


with right_col:
    st.subheader("Top 3 relevant articles")
    hits = st.session_state.get("last_hits") or []
    
    if not hits:
        st.caption("No results yet. Try asking a question.")
    else:
        # Style for larger right-panel images
        st.markdown(
            """
            <style>
            /* Make right-panel images roughly 2x smaller */
            .right-article-img {
                width: 50%;
                max-height: 28vh;
                object-fit: contain;
                display: block;
                margin: 0 auto; /* center the smaller image */
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        # Try a scrollable container if supported (Streamlit >= 1.32)
        try:
            panel = st.container(height=520, border=True)
            use_panel = True
        except TypeError:
            use_panel = False
        ctx = panel if use_panel else st
        with ctx if use_panel else st.container():
            for i, h in enumerate(hits[:3], start=1):
                a = h["article"]
                title = a.get("title", "(no title)")
                url = a.get("issue_url", "")
                # Pick first image if present
                img_url = None
                try:
                    imgs = a.get("images") or []
                    if isinstance(imgs, list) and imgs and isinstance(imgs[0], str):
                        img_url = imgs[0]
                except Exception:
                    img_url = None
                # Render card with image and similarity badge
                distance = h.get("distance")
                if img_url:
                    # Display image
                    st.markdown(
                        f"<img class='right-article-img' src='{img_url}' />",
                        unsafe_allow_html=True,
                    )
                
                # Display title
                st.markdown(f"**{i}. [{title}]({url})**")
                
                # Add a short text excerpt
                try:
                    raw_text = (a.get("text") or "").strip()
                except Exception:
                    raw_text = ""
                if raw_text:
                    short = raw_text[:320] + ("…" if len(raw_text) > 320 else "")
                    st.markdown(
                        f"<div style='font-size:0.9em;color:#444;line-height:1.3'>{short}</div>",
                        unsafe_allow_html=True,
                    )
                st.divider()
