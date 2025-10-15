import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from functools import lru_cache

# Defaults (keep in sync with app.py)
CHROMA_PATH = Path("chroma_db")
COLLECTION_NAME = "Article"
IMAGE_COLLECTION_NAME = "ArticleImages"
IMAGE_MODEL_NAME = "clip-ViT-B-32"

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


@dataclass
class RAGState:
    articles: List[Dict[str, Any]]
    model: SentenceTransformer
    chroma_path: Path = CHROMA_PATH
    collection_name: str = COLLECTION_NAME


def retrieve_relevant_articles(
    state: RAGState,
    query: str,
    top_k: int = 5,
    *,
    exclude_idxs: Optional[Set[int]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top_k relevant unique non-sponsor articles from Chroma using cosine distance.
    Returns list of dicts: { distance, article, idx }.
    """
    # Qwen3 embedding models provide a dedicated "query" prompt for better retrieval.
    try:
        qv = state.model.encode([query], prompt_name="query", convert_to_numpy=True)[0]
    except Exception:
        qv = state.model.encode([query], convert_to_numpy=True)[0]
    client = PersistentClient(path=str(state.chroma_path))
    col = client.get_or_create_collection(
        state.collection_name, metadata={"hnsw:space": "cosine"}
    )
    # Ask for many to allow diversity/exclusions
    res = col.query(
        query_embeddings=[qv.tolist()],
        n_results=max(25, top_k * 5),
        include=["metadatas", "distances"],
    )
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    hits: List[Dict[str, Any]] = []
    seen: Set[int] = set()
    excluded = exclude_idxs or set()
    for meta, dist in zip(metas, dists):
        idx = meta.get("idx") if isinstance(meta, dict) else None
        if not isinstance(idx, int) or idx < 0 or idx >= len(state.articles):
            continue
        if idx in seen or idx in excluded:
            continue
        if is_sponsor_title(state.articles[idx].get("title")):
            continue
        seen.add(idx)
        hits.append(
            {
                "distance": float(dist) if dist is not None else None,
                "article": state.articles[idx],
                "idx": idx,
            }
        )
        if len(hits) >= top_k:
            break
    # If we couldn't fill top_k without repeats, allow repeats to fill remaining
    if len(hits) < top_k:
        for meta, dist in zip(metas, dists):
            idx = meta.get("idx") if isinstance(meta, dict) else None
            if not isinstance(idx, int) or idx < 0 or idx >= len(state.articles):
                continue
            if idx in seen:
                continue
            if is_sponsor_title(state.articles[idx].get("title")):
                continue
            seen.add(idx)
            hits.append(
                {
                    "distance": float(dist) if dist is not None else None,
                    "article": state.articles[idx],
                    "idx": idx,
                }
            )
            if len(hits) >= top_k:
                break
    return hits


@lru_cache(maxsize=1)
def _get_clip_model() -> SentenceTransformer:
    return SentenceTransformer(IMAGE_MODEL_NAME)


def retrieve_by_image(
    state: RAGState,
    image,
    top_k: int = 5,
    *,
    exclude_idxs: Optional[Set[int]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve using an image query against the image collection (CLIP)."""
    clip = _get_clip_model()
    # encode single image (PIL or numpy array)
    qv = clip.encode([image], convert_to_numpy=True)[0]
    client = PersistentClient(path=str(state.chroma_path))
    col = client.get_or_create_collection(
        IMAGE_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    res = col.query(
        query_embeddings=[qv.tolist()],
        n_results=max(25, top_k * 5),
        include=["metadatas", "distances"],
    )
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    hits: List[Dict[str, Any]] = []
    seen: Set[int] = set()
    excluded = exclude_idxs or set()
    for meta, dist in zip(metas, dists):
        idx = meta.get("idx") if isinstance(meta, dict) else None
        if not isinstance(idx, int) or idx < 0 or idx >= len(state.articles):
            continue
        if idx in seen or idx in excluded:
            continue
        if is_sponsor_title(state.articles[idx].get("title")):
            continue
        seen.add(idx)
        hits.append(
            {
                "distance": float(dist) if dist is not None else None,
                "article": state.articles[idx],
                "idx": idx,
            }
        )
        if len(hits) >= top_k:
            break
    return hits


def _minmax(scores: Dict[int, float]) -> Dict[int, float]:
    """Min-max normalize scores to [0, 1]"""
    if not scores:
        return scores
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _compute_smart_alpha(text_scores: Dict[int, float], img_scores: Dict[int, float]) -> float:
    """
    Compute smart weighting based on the average quality of text vs image similarities.
    Returns alpha_text (weight for text), where alpha ∈ [0, 1].
    
    Strategy: Compare average top-K similarities. Higher average similarity gets more weight.
    """
    if not text_scores and not img_scores:
        return 0.5  # Default equal weight
    if not text_scores:
        return 0.0  # Only image scores available
    if not img_scores:
        return 1.0  # Only text scores available
    
    # Get top-K scores (K=5) for comparison
    top_k = 5
    text_top = sorted(text_scores.values(), reverse=True)[:top_k]
    img_top = sorted(img_scores.values(), reverse=True)[:top_k]
    
    avg_text = sum(text_top) / len(text_top) if text_top else 0.0
    avg_img = sum(img_top) / len(img_top) if img_top else 0.0
    
    # Normalize to sum to 1.0
    total = avg_text + avg_img
    if total == 0:
        return 0.5
    
    alpha_text = avg_text / total
    
    # Clamp to reasonable bounds [0.3, 0.7] to prevent extreme dominance
    alpha_text = max(0.3, min(0.7, alpha_text))
    
    return alpha_text


def retrieve_multimodal_articles(
    state: RAGState,
    *,
    text_query: Optional[str],
    image=None,
    top_k: int = 5,
    exclude_idxs: Optional[Set[int]] = None,
    alpha_text: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Unified retrieval:
    - text only -> text collection (Qwen)
    - image only -> image collection (CLIP)
    - both -> fuse normalized scores with smart weighting based on similarity quality
    
    If alpha_text is None, automatically computes optimal weight based on similarity scores.
    Returns list of dicts like retrieve_relevant_articles.
    """
    has_text = bool(text_query and text_query.strip())
    has_image = image is not None

    if has_text and not has_image:
        return retrieve_relevant_articles(
            state, text_query, top_k=top_k, exclude_idxs=exclude_idxs
        )
    if has_image and not has_text:
        return retrieve_by_image(state, image, top_k=top_k, exclude_idxs=exclude_idxs)

    # Both modalities: late fusion of scores
    text_hits = retrieve_relevant_articles(
        state, text_query or "", top_k=max(25, top_k * 5), exclude_idxs=exclude_idxs
    )
    img_hits = retrieve_by_image(
        state, image, top_k=max(25, top_k * 5), exclude_idxs=exclude_idxs
    )
    text_scores = {
        h["idx"]: 1.0 - float(h["distance"]) if h.get("distance") is not None else 0.0
        for h in text_hits
    }
    img_scores = {
        h["idx"]: 1.0 - float(h["distance"]) if h.get("distance") is not None else 0.0
        for h in img_hits
    }
    
    # Compute smart alpha if not provided
    if alpha_text is None:
        alpha_text = _compute_smart_alpha(text_scores, img_scores)
    
    tn = _minmax(text_scores)
    im = _minmax(img_scores)
    keys = set(tn) | set(im)
    fused = {
        k: alpha_text * tn.get(k, 0.0) + (1.0 - alpha_text) * im.get(k, 0.0)
        for k in keys
    }
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    out: List[Dict[str, Any]] = []
    for idx, _s in ranked:
        out.append({"article": state.articles[idx], "idx": idx})
        if len(out) >= top_k:
            break
    # Backfill with remaining text or image hits if needed
    if len(out) < top_k:
        seen = {x["idx"] for x in out}
        for h in text_hits + img_hits:
            if h["idx"] in seen:
                continue
            out.append({"article": state.articles[h["idx"]], "idx": h["idx"]})
            if len(out) >= top_k:
                break
    # Add distance=None to match UI expected keys
    return [{"distance": None, "article": h["article"], "idx": h["idx"]} for h in out]


def construct_prompt(
    state: RAGState, query: str, history_text: str = ""
) -> Tuple[str, str]:
    """Build a prompt from the top-3 retrieved articles. Returns (prompt, context)."""
    top3 = retrieve_relevant_articles(state, query, top_k=3)
    context_snippets: List[str] = []
    for h in top3:
        a = h["article"]
        excerpt = redact_emails((a.get("text") or "")[:700])
        first_img = ""
        try:
            imgs = a.get("images") or []
            if isinstance(imgs, list) and imgs and isinstance(imgs[0], str):
                first_img = imgs[0]
        except Exception:
            first_img = ""
        img_line = f"Image: {first_img}\n" if first_img else ""
        context_snippets.append(
            f"Title: {a.get('title', '')}\nURL: {a.get('issue_url', '')}\n{img_line}Excerpt: {excerpt}"
        )
    context_block = "\n\n".join(context_snippets)
    system_prompt = (
        "You are a helpful assistant answering questions strictly based on The Batch articles provided in Context. "
        "Rules: (1) Use only information in Context; do not invent facts. (2) If the answer is not in Context, say 'I don't know based on the retrieved articles.' "
        "(3) Keep answers concise. (4) When referencing details, briefly mention the article title."
    )
    full = f"System prompt:\n{system_prompt}\n\nContext:\n{context_block}\n\n"
    if history_text:
        full += f"Recent conversation:\n{history_text}\n\n"
    full += f"Question:\n{query}\n\nAnswer:"
    return full, context_block
