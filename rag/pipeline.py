import os
import glob
import re
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import streamlit as st

# Vector search libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "docs")
IMG_DIR = os.path.join(DATA_DIR, "images")

# Default Hyperparameters (can be overridden by UI)
DEFAULT_TOP_K_TEXT = 5
DEFAULT_TOP_K_IMAGES = 3
DEFAULT_TOP_K_EVIDENCE = 8
DEFAULT_ALPHA = 0.5
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------
@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    page_num: int
    text: str

@dataclass
class ImageItem:
    item_id: str
    path: str
    caption: str

# -----------------------------------------------------------------------------
# Ingestion & Processing
# -----------------------------------------------------------------------------
def clean_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_pdf_pages(pdf_path: str) -> List[TextChunk]:
    doc_id = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    out: List[TextChunk] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = clean_text(page.get_text("text"))
        if text:
            out.append(TextChunk(chunk_id=f"{doc_id}::p{i+1}", doc_id=doc_id, page_num=i+1, text=text))
    return out

def chunk_text_fixed(text: str, chunk_size: int, overlap: int, doc_id: str, page_num: int) -> List[TextChunk]:
    chunks = []
    if not text: return chunks
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append(TextChunk(chunk_id=f"{doc_id}::p{page_num}::c{start}", doc_id=doc_id, page_num=page_num, text=chunk_text))
        start += (chunk_size - overlap)
        if start >= len(text): break
    return chunks

def extract_pdf_fixed(pdf_path: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[TextChunk]:
    doc_id = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    out: List[TextChunk] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = clean_text(page.get_text("text"))
        if text:
            out.extend(chunk_text_fixed(text, chunk_size, overlap, doc_id, i+1))
    return out

def load_images(fig_dir: str) -> List[ImageItem]:
    # If the file path doesn't exist, remove it from image_items
    items: List[ImageItem] = []
    
    # Simple check for existence before globbing
    if not os.path.exists(fig_dir):
         print(f"Warning: Image directory not found at {fig_dir}")
         return items

    search_path = os.path.join(fig_dir, "*.*")
    print(f"Loading images from: {search_path}")
    
    # Force a fresh glob search and only include existing files
    for p in sorted(glob.glob(search_path)):
        if p.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            base = os.path.basename(p)
            caption = os.path.splitext(base)[0].replace("_", " ")
            items.append(ImageItem(item_id=base, path=p, caption=caption))
    return items

# -----------------------------------------------------------------------------
# Indexing
# -----------------------------------------------------------------------------
def build_tfidf_index(texts: List[str]):
    vec = TfidfVectorizer(lowercase=True, stop_words="english")
    X = vec.fit_transform(texts)
    X = normalize(X)
    return vec, X

def build_bm25_index(texts: List[str]):
    tokenized_corpus = [doc.split(" ") for doc in texts]
    return BM25Okapi(tokenized_corpus)

def build_dense_index(texts: List[str], model):
    embeddings = model.encode(texts, convert_to_numpy=True)
    normalize(embeddings, copy=False)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

# -----------------------------------------------------------------------------
# Cached Resource Loader
# -----------------------------------------------------------------------------
def load_and_index_data_core():
    """
    Core logic to load PDFs and images, build all indices (TF-IDF, BM25, Dense).
    Uncached, for use in API or direct calls.
    """
    # 1. Load Data
    # Ensure we look in the correct path relative to project root
    print(f"Loading PDFs from: {os.path.abspath(PDF_DIR)}")
    pdfs = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdfs:
        # Fallback for when running inside a subdir
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt_pdf_dir = os.path.join(base_dir, "data", "docs")
        print(f"Trying alternative path: {alt_pdf_dir}")
        pdfs = sorted(glob.glob(os.path.join(alt_pdf_dir, "*.pdf")))

    print(f"Found {len(pdfs)} PDFs")
    page_chunks = []
    for p in pdfs: page_chunks.extend(extract_pdf_pages(p))
    
    fixed_chunks = []
    for p in pdfs: fixed_chunks.extend(extract_pdf_fixed(p))
    
    image_items = load_images(IMG_DIR)
    if not image_items:
        # Fallback for images
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt_img_dir = os.path.join(base_dir, "data", "images")
        print(f"Trying alternative image path: {alt_img_dir}")
        image_items = load_images(alt_img_dir)
        
    print(f"Found {len(image_items)} images")

    # 2. Load Models
    model_st = SentenceTransformer('all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # 3. Build Indices
    
    # Page-based
    page_texts = [c.text for c in page_chunks]
    tfidf_vec_page, tfidf_X_page = build_tfidf_index(page_texts)
    bm25_page = build_bm25_index(page_texts)
    dense_index_page = build_dense_index(page_texts, model_st)

    # Fixed-size
    fixed_texts = [c.text for c in fixed_chunks]
    tfidf_vec_fixed, tfidf_X_fixed = build_tfidf_index(fixed_texts)
    bm25_fixed = build_bm25_index(fixed_texts)
    dense_index_fixed = build_dense_index(fixed_texts, model_st)

    # Images
    img_texts = [it.caption for it in image_items]
    tfidf_vec_img, tfidf_X_img = build_tfidf_index(img_texts) if img_texts else (None, None)
    dense_index_img = build_dense_index(img_texts, model_st) if img_texts else None

    return {
        "page_chunks": page_chunks,
        "fixed_chunks": fixed_chunks,
        "image_items": image_items,
        "model_st": model_st,
        "cross_encoder": cross_encoder,
        "indices": {
            "page": {
                "tfidf": (tfidf_vec_page, tfidf_X_page),
                "bm25": bm25_page,
                "dense": dense_index_page
            },
            "fixed": {
                "tfidf": (tfidf_vec_fixed, tfidf_X_fixed),
                "bm25": bm25_fixed,
                "dense": dense_index_fixed
            },
            "image": {
                "tfidf": (tfidf_vec_img, tfidf_X_img),
                "dense": dense_index_img
            }
        }
    }

@st.cache_resource
def load_and_index_data():
    """
    Streamlit cached wrapper for load_and_index_data_core.
    """
    return load_and_index_data_core()

# -----------------------------------------------------------------------------
# Retrieval Logic
# -----------------------------------------------------------------------------
def retrieve_tfidf(query: str, vec, X, top_k=5):
    if vec is None or X is None: return []
    q = vec.transform([query])
    q = normalize(q)
    scores = (X @ q.T).toarray().ravel()
    idx = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in idx]

def retrieve_bm25(query: str, bm25_obj, top_k=5):
    tokenized_query = query.split(" ")
    scores = bm25_obj.get_scores(tokenized_query)
    idx = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in idx]

def retrieve_dense(query: str, index, model, top_k=5):
    if index is None: return []
    q_emb = model.encode([query], convert_to_numpy=True)
    normalize(q_emb, copy=False)
    scores, indices = index.search(q_emb, top_k)
    return [(int(indices[0][i]), float(scores[0][i])) for i in range(top_k) if indices[0][i] != -1]

def build_context(
    query: str,
    data_pack: Dict[str, Any],
    method: str = "sparse",
    chunking: str = "page",
    top_k_text: int = DEFAULT_TOP_K_TEXT,
    top_k_images: int = DEFAULT_TOP_K_IMAGES,
    top_k_evidence: int = DEFAULT_TOP_K_EVIDENCE,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    
    indices = data_pack["indices"]
    model_st = data_pack["model_st"]
    cross_encoder = data_pack["cross_encoder"]
    image_items = data_pack["image_items"]

    # Select Corpus
    if chunking == "page":
        chunks = data_pack["page_chunks"]
        tfidf_vec, tfidf_X = indices["page"]["tfidf"]
        bm25_obj = indices["page"]["bm25"]
        dense_idx = indices["page"]["dense"]
    else:
        chunks = data_pack["fixed_chunks"]
        tfidf_vec, tfidf_X = indices["fixed"]["tfidf"]
        bm25_obj = indices["fixed"]["bm25"]
        dense_idx = indices["fixed"]["dense"]

    # 1. Text Retrieval
    text_hits = []
    if method == "sparse":
        text_hits = retrieve_tfidf(query, tfidf_vec, tfidf_X, top_k=top_k_text)
    elif method == "bm25":
        text_hits = retrieve_bm25(query, bm25_obj, top_k=top_k_text)
    elif method == "dense":
        text_hits = retrieve_dense(query, dense_idx, model_st, top_k=top_k_text)
    elif "hybrid" in method:
        h1 = retrieve_tfidf(query, tfidf_vec, tfidf_X, top_k=top_k_text * 2)
        h2 = retrieve_dense(query, dense_idx, model_st, top_k=top_k_text * 2)
        
        combined = {}
        # Weighted fusion: alpha controls text vs image, but here we fuse sparse vs dense text retrieval
        # Usually hybrid means sparse + dense fusion. Let's assume standard rank fusion here.
        # Note: The notebook uses 0.3/0.7 for hybrid. I'll stick to that or use alpha for multimodal.
        # Actually, alpha is described in notebook as "0.0 = images dominate, 1.0 = text dominate".
        # So hybrid fusion of text retrievers is fixed or another parameter.
        # I'll use Reciprocal Rank Fusion or simple score addition for sparse/dense text.
        
        # Let's stick to the notebook implementation: 0.3 * sparse + 0.7 * dense
        for i, s in h1: combined[i] = combined.get(i, 0) + 0.3 * s
        for i, s in h2: combined[i] = combined.get(i, 0) + 0.7 * s
        text_hits = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k_text]

    candidates = []
    for idx, s in text_hits:
        candidates.append({
            "modality": "text",
            "id": chunks[idx].chunk_id,
            "score": float(s),
            "text": chunks[idx].text,
            "path": None
        })

    # 2. Image Retrieval
    img_hits = []
    
    # Always attempt hybrid for images to maximize chance of finding something relevant
    # especially if filenames are poor.
    
    hits_tfidf = []
    if indices["image"]["tfidf"]:
        hits_tfidf = retrieve_tfidf(query, indices["image"]["tfidf"][0], indices["image"]["tfidf"][1], top_k=top_k_images * 2)

    hits_dense = []
    if indices["image"]["dense"]:
        hits_dense = retrieve_dense(query, indices["image"]["dense"], model_st, top_k=top_k_images * 2)

    combined_img = {}
    # Basic fusion 0.5/0.5
    for i, s in hits_tfidf: combined_img[i] = combined_img.get(i, 0) + 0.5 * s
    for i, s in hits_dense: combined_img[i] = combined_img.get(i, 0) + 0.5 * s
    
    img_hits = sorted(combined_img.items(), key=lambda x: x[1], reverse=True)[:top_k_images]

    for idx, s in img_hits:
        candidates.append({
            "modality": "image",
            "id": image_items[idx].item_id,
            "score": float(s),
            "text": image_items[idx].caption,
            "path": image_items[idx].path
        })

    # 3. Rerank (if requested)
    if "rerank" in method:
        pairs = [[query, c["text"]] for c in candidates]
        scores = cross_encoder.predict(pairs)
        for i, s in enumerate(scores): candidates[i]["score"] = float(s)
        candidates.sort(key=lambda x: x["score"], reverse=True)
    else:
        candidates.sort(key=lambda x: x["score"], reverse=True)

    # 4. Final Selection
    final_evidence = candidates[:top_k_evidence]
    ctx_lines = []
    image_paths = []
    evidence_ids = []
    
    for ev in final_evidence:
        evidence_ids.append(ev["id"])
        if ev["modality"] == "text":
            snippet = (ev["text"] or "")[:300].replace("\n", " ")
            ctx_lines.append(f"[TEXT | {ev['id']} | score={ev['score']:.3f}] {snippet}")
        else:
            ctx_lines.append(f"[IMAGE | {ev['id']} | score={ev['score']:.3f}] caption={ev['text']}")
            image_paths.append(ev["path"])

    return {
        "question": query,
        "context": "\n".join(ctx_lines),
        "image_paths": image_paths,
        "evidence": final_evidence,
        "evidence_ids": evidence_ids,
        "method": method
    }

# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------
def simple_extractive_answer(question: str, context: str, evidence: List[Dict] = None) -> str:
    MISSING_EVIDENCE_MSG = "Not enough evidence in the retrieved context."
    
    if not evidence or not context.strip():
        return MISSING_EVIDENCE_MSG
        
    # Check max score threshold
    max_score = max((e.get("score", 0.0) for e in evidence), default=0.0)
    # Cosine similarity scores are typically 0-1, but can be higher depending on impl
    # The notebook uses 0.05 threshold
    if max_score < 0.05:
        return MISSING_EVIDENCE_MSG

    lines = context.splitlines()
    if not lines:
         return MISSING_EVIDENCE_MSG
         
    # Format with citation tags
    formatted_answer = f"**Question:** {question}\n\n**Grounded Answer:**\n"
    
    for line in lines[:3]:
        # Simple regex-free parsing for robustness
        # line looks like: [TEXT | doc1.pdf::p1 | score=0.5] content...
        if "|" in line and "]" in line:
            try:
                meta, content = line.split("]", 1)
                parts = meta.split("|")
                if len(parts) >= 2:
                    doc_id = parts[1].strip()
                    # Clean up doc_id (remove ::p1)
                    if "::" in doc_id:
                        doc_id = doc_id.split("::")[0]
                    formatted_answer += f"- {content.strip()} [{doc_id}]\n"
                else:
                    formatted_answer += f"- {line}\n"
            except:
                formatted_answer += f"- {line}\n"
        else:
            formatted_answer += f"- {line}\n"
            
    return formatted_answer

def evaluate_retrieval(evidence, gold_evidence_ids, p_at_k=5, r_at_k=10):
    """Precision@P and Recall@R: gold_evidence_ids are filenames (e.g. doc1.pdf, img.png)."""
    if not gold_evidence_ids:
        return 0.0, 0.0

    gold_set = set()
    for gid in gold_evidence_ids:
        base = os.path.basename(gid)
        name = os.path.splitext(base)[0]
        gold_set.add(name.lower())

    # Precision@5: relevant in top p_at_k
    relevant_p = 0
    for ev in evidence[:p_at_k]:
        chunk_id = ev.get("id", "")
        base_filename = chunk_id.split("::")[0] if "::" in chunk_id else chunk_id
        base_name = os.path.splitext(base_filename)[0].lower()
        if base_name in gold_set:
            relevant_p += 1
    precision = relevant_p / p_at_k if p_at_k > 0 else 0.0

    # Recall@10: relevant in top r_at_k (unique gold items hit)
    seen_gold = set()
    for ev in evidence[:r_at_k]:
        chunk_id = ev.get("id", "")
        base_filename = chunk_id.split("::")[0] if "::" in chunk_id else chunk_id
        base_name = os.path.splitext(base_filename)[0].lower()
        if base_name in gold_set:
            seen_gold.add(base_name)
    recall = len(seen_gold) / len(gold_set) if gold_set else 0.0

    return precision, recall
