import json
from pathlib import Path
import pickle
import numpy as np
import faiss
import re
import torch

try:
    from sentence_transformers import SentenceTransformer
    HAVE_ST = True
except Exception:
    SentenceTransformer = None
    HAVE_ST = False
try:
    from rank_bm25 import BM25Okapi
    HAVE_BM25 = True
except Exception:
    BM25Okapi = None
    HAVE_BM25 = False

def load_faiss_components(index_dir: str = '.'):
    """Load faiss index, metadata, chunks text and any TF-IDF/SVD fallback artifacts.

    Returns a dict with keys: index, metadata, chunks_text, vectorizer, svd, embed_model_name
    """
    p = Path(index_dir)
    # Try common filenames
    idx = None
    for fn in ('faiss_index.hnsw', 'faiss_index.ivf', 'faiss_index'):
        fpath = p / fn
        if fpath.exists():
            idx = faiss.read_index(str(fpath))
            break

    # metadata
    meta_file = p / 'chunks_meta.jsonl'
    metadata = []
    if meta_file.exists():
        with open(meta_file, 'r', encoding='utf-8') as f:
            for line in f:
                metadata.append(json.loads(line))

    # chunks text
    chunks_text = {}
    chunks_file = p / 'data_books' / 'chunks.jsonl'
    if not chunks_file.exists():
        chunks_file = p / 'chunks.jsonl'
    if chunks_file.exists():
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                chunks_text[obj.get('id')] = obj.get('text', '')

    # load tfidf + svd if present
    vectorizer = None
    svd = None
    tfidf_file = p / 'tfidf_vectorizer.pkl'
    svd_file = p / 'svd_model.pkl'
    if tfidf_file.exists() and svd_file.exists():
        try:
            with open(tfidf_file, 'rb') as f:
                vectorizer = pickle.load(f)
            with open(svd_file, 'rb') as f:
                svd = pickle.load(f)
        except Exception:
            vectorizer = None
            svd = None

    # embedding model info
    embed_info = {}
    emb_info_file = p / 'embedding_model_info.json'
    if emb_info_file.exists():
        try:
            with open(emb_info_file, 'r', encoding='utf-8') as f:
                embed_info = json.load(f)
        except Exception:
            embed_info = {}

    # Optionally load/cached embedding model instance for repeated queries
    embed_model = None
    embed_model_name = embed_info.get('model', 'all-mpnet-base-v2') if embed_info else 'all-mpnet-base-v2'
    if HAVE_ST:
        try:
            embed_model = SentenceTransformer(embed_model_name)
        except Exception:
            embed_model = None

    # Build BM25 index if available for a fast lexical rerank/fallback
    bm25 = None
    chunks_tokenized = None
    if HAVE_BM25 and chunks_text:
        try:
            docs = [re.findall(r"\w+", chunks_text[k].lower()) for k in chunks_text]
            chunks_tokenized = docs
            bm25 = BM25Okapi(docs)
        except Exception:
            bm25 = None
            chunks_tokenized = None

    return {
        'index': idx,
        'metadata': metadata,
        'chunks_text': chunks_text,
        'vectorizer': vectorizer,
        'svd': svd,
        'embed_info': embed_info,
        'embed_model': embed_model,
        'embed_model_name': embed_model_name,
        'bm25': bm25,
        'chunks_tokenized': chunks_tokenized,
    }


def embed_query(rag, query, model_name: str = None):
    """Return normalized embedding for query using cached sentence-transformers model if present.

    If no embedding model is available, returns None and caller can fallback to BM25/TF-IDF.
    """
    if rag is None:
        return None
    model = rag.get('embed_model')
    mname = model_name or rag.get('embed_model_name')
    if model is None and HAVE_ST and mname:
        try:
            model = SentenceTransformer(mname)
        except Exception:
            model = None
    if model is None:
        return None
    try:
        emb = model.encode([query], convert_to_numpy=True)
        emb = emb.astype('float32')
        faiss.normalize_L2(emb)
        return emb
    except Exception:
        return None


def search_rag(rag, query: str = None, query_emb: np.ndarray = None, k: int = 3):
    """Search the FAISS index using query_emb and optionally merge with BM25 lexical scores.

    Returns list of dicts with text and meta. Will attempt the following flow:
    - If FAISS index + query_emb available: get top-N candidates from FAISS.
    - If BM25 available: get top-N candidates and merge with FAISS candidates for reranking/diversity.
    - If only BM25 available: return BM25 top-N.
    """
    idx = rag.get('index')
    bm25 = rag.get('bm25')
    chunks_text = rag.get('chunks_text', {})
    metadata = rag.get('metadata', [])

    faiss_ids = []
    faiss_scores = []
    if idx is not None and query_emb is not None:
        try:
            scores, indices = idx.search(query_emb, k)
            # faiss may return distances depending on index type; treat as score
            for score, idx_i in zip(scores[0], indices[0]):
                if idx_i < 0:
                    continue
                faiss_ids.append(idx_i)
                faiss_scores.append(float(score))
        except Exception:
            faiss_ids = []
            faiss_scores = []

    bm25_ids = []
    if bm25 is not None:
        try:
            qtok = re.findall(r"\w+", (query or "").lower())
            top_docs = bm25.get_top_n(qtok, list(chunks_text.values()), n=k)
            # map top_docs to their keys (chunk ids)
            # note: chunks_text is an insertion-ordered dict in Python 3.7+, so index lookup is stable
            all_texts = list(chunks_text.values())
            for td in top_docs:
                try:
                    idx_text = all_texts.index(td)
                    # map index to metadata index if possible
                    if idx_text < len(metadata):
                        bm25_ids.append(idx_text)
                except ValueError:
                    continue
        except Exception:
            bm25_ids = []

    # Merge ids preserving FAISS order first then BM25, remove duplicates
    merged_idx_positions = []
    for lid in faiss_ids:
        if lid not in merged_idx_positions:
            merged_idx_positions.append(lid)
    for bid in bm25_ids:
        if bid not in merged_idx_positions:
            merged_idx_positions.append(bid)

    results = []
    for idx_pos in merged_idx_positions[:k]:
        meta = metadata[idx_pos] if idx_pos < len(metadata) else {}
        chunk_id = meta.get('id')
        text = chunks_text.get(chunk_id, '')
        results.append({'score': None, 'id': chunk_id, 'meta': meta, 'text': text})

    # If no candidates but bm25 existed, return bm25 top-k by reconstructing from texts
    if not results and bm25 is not None:
        try:
            qtok = re.findall(r"\w+", (query or "").lower())
            top_docs = bm25.get_top_n(qtok, list(chunks_text.values()), n=k)
            all_texts = list(chunks_text.values())
            for td in top_docs:
                try:
                    idx_text = all_texts.index(td)
                    meta = metadata[idx_text] if idx_text < len(metadata) else {}
                    chunk_id = meta.get('id')
                    text = chunks_text.get(chunk_id, '')
                    results.append({'score': None, 'id': chunk_id, 'meta': meta, 'text': text})
                except ValueError:
                    continue
        except Exception:
            return []

    return results


def assemble_prompt(retrieved: list, query: str, max_context_chars: int = 1500) -> str:
    """Simple, effective prompt assembly to minimize corruption and improve answers.

    retrieved: list of {'score','id','meta','text'}
    """
    # Minimal, direct instruction
    instruction = "Answer the question using the context provided:\n\n"

    ctx_parts = []
    total = 0
    for i, r in enumerate(retrieved[:3], start=1):  # Limit to top 3 results
        txt = r.get('text', '').strip()
        if not txt:
            continue
            
        # Aggressive cleaning to prevent corruption
        txt_clean = re.sub(r'\(cid[:\d\[\]\(\)h\-]*\)', '', txt)  # Remove all cid variants
        txt_clean = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', txt_clean)  # Keep minimal punctuation
        txt_clean = re.sub(r'\s+', ' ', txt_clean)  # Normalize whitespace
        txt_clean = txt_clean.strip()
        
        # Skip fragments
        if len(txt_clean.split()) < 8:
            continue
            
        # Short, clean snippets
        snippet = txt_clean[:500]  # Shorter to avoid corruption
        if total + len(snippet) > max_context_chars:
            break
        ctx_parts.append(snippet)
        total += len(snippet)

    context_block = " ".join(ctx_parts)  # Simple joining

    if context_block:
        # Minimal format to reduce confusion
        prompt = f"{instruction}{context_block}\n\nQ: {query}\nA:"
    else:
        prompt = f"Q: {query}\nA:"

    return prompt


def is_bad_response(text: str) -> bool:
    """Enhanced heuristic to detect garbled or poor quality responses.

    Returns True if response looks invalid and should be retried.
    """
    if not text:
        return True
    s = text.strip()
    
    # Too short or empty
    if len(s) < 20:
        return True
        
    # Contains unicode control characters or corruption artifacts
    if re.search(r'\(cid:\d+\)', s) or re.search(r'cid[:\[\]\(\)h\-\d]+', s):
        return True
        
    # Starts with artifacts or incomplete patterns
    if re.match(r'^[:\(\)\[\]]+', s) or s.startswith('You are a technical expert'):
        return True
        
    # High ratio of non-alphanumeric characters (corruption indicator)
    alnum_chars = sum(1 for c in s if c.isalnum())
    if alnum_chars / max(1, len(s)) < 0.6:  # At least 60% should be alphanumeric
        return True
        
    # Repeated punctuation or repeated single character (corruption)
    if re.search(r'(.)\1{4,}', s):
        return True
        
    # Contains instruction text instead of answer
    instruction_markers = [
        "Instructions:", "CRITICAL INSTRUCTIONS:", "- Answer using", 
        "- Be precise", "- Do NOT", "Read the context"
    ]
    if any(marker in s for marker in instruction_markers):
        return True
        
    # Repetitive phrases (same phrase 3+ times)
    words = s.split()
    if len(words) > 15:
        phrase_counts = {}
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
            if phrase_counts[phrase] >= 3:
                return True
                
    # Starts with lowercase (likely a mid-sentence fragment)
    if s and s[0].islower() and not s.startswith(('and', 'or', 'but', 'the', 'a', 'an', 'in', 'on', 'at')):
        return True
        
    # Contains mostly incomplete sentences or fragments
    sentences = re.split(r'[.!?]+', s)
    complete_sentences = [sent.strip() for sent in sentences if len(sent.strip().split()) > 4]
    if len(complete_sentences) == 0:
        return True
        
    # Check for excessive context/source continuation patterns
    context_patterns = [
        "Reference \\d+:", "Context \\d+:", "Figure \\d+", "Page \\d+", 
        "Section \\d+", "Chapter \\d+", "Question:", "Q:"
    ]
    if any(re.search(pattern, s) for pattern in context_patterns):
        return True
        
    return False


def single_generate(model, tokenizer, prompt: str, device, max_new_tokens: int = 200, num_beams: int = 6):
    """Generate a single response deterministically for a prompt. Returns cleaned decoded text."""
    model.eval()
    toks = tokenizer([prompt], return_tensors='pt', padding=True, truncation=True, max_length=2048).to(device)
    
    with torch.no_grad():
        out = model.generate(
            toks.input_ids,
            attention_mask=toks.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic generation
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,  # Stronger penalty to avoid repetition
            length_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Correctly slice off prompt tokens - handle left padding
    input_len = toks.attention_mask.sum(dim=1).cpu().numpy()[0]
    gen_tokens = out[0][input_len:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    
    # Enhanced post-processing to clean common issues
    text = re.sub(r'\(cid:\d+\)', '', text)  # Remove unicode control chars
    text = re.sub(r'cid[:\[\]\(\)h\-\d]*', '', text)  # Remove cid variants
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\/\%\&]', ' ', text)  # Keep essential punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    # Remove instruction artifacts that might leak through
    instruction_patterns = [
        r'You are a technical expert[^.]*\.',
        r'Read the context carefully[^.]*\.',
        r'Instructions:[^:]*:',
        r'CRITICAL INSTRUCTIONS:[^:]*:',
        r'- Answer using[^.]*\.',
        r'- Be precise[^.]*\.',
        r'- Do NOT[^.]*\.',
    ]
    for pattern in instruction_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    text = text.strip()
    
    # Ensure proper sentence structure
    if text and text[0].islower():
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    
    return text
