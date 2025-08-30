"""build_faiss.py

Hybrid FAISS index builder with dense embedding support (sentence-transformers)
and robust TF-IDF+SVD fallback. Optionally builds a BM25 sparse index (rank-bm25).

Outputs:
 - faiss_index.{hnsw,ivf}  (faiss index)
 - chunks_meta.jsonl
 - embedding_model_info.json
 - tfidf_vectorizer.pkl, svd_model.pkl (when using TF-IDF fallback)

Usage: python build_faiss.py --chunks data_books/chunks.jsonl --out_dir . --use_dense True
"""

from pathlib import Path
import argparse
import json
import numpy as np
import faiss
import os
import pickle
from typing import List

try:
    from sentence_transformers import SentenceTransformer
    HAVE_ST = True
except Exception:
    SentenceTransformer = None
    HAVE_ST = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    HAVE_SK = True
except Exception:
    TfidfVectorizer = None
    TruncatedSVD = None
    HAVE_SK = False

try:
    from rank_bm25 import BM25Okapi
    HAVE_BM25 = True
except Exception:
    BM25Okapi = None
    HAVE_BM25 = False


def load_chunks(path: str) -> List[dict]:
    texts = []
    meta = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj.get('text', ''))
            meta.append({'id': obj.get('id'), 'source': obj.get('source'), 'page': obj.get('page'), 'level': obj.get('level')})
    return texts, meta


def build_dense_embeddings(texts: List[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    if not HAVE_ST:
        raise RuntimeError('sentence-transformers not available for dense embeddings')
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    # normalize for cosine
    faiss.normalize_L2(embs)
    return embs.astype('float32'), model_name


def build_tfidf_svd(texts: List[str], max_features: int = 20000, svd_dim: int = 384):
    if not HAVE_SK:
        raise RuntimeError('scikit-learn required for TF-IDF fallback')
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.95)
    X = vectorizer.fit_transform(texts)
    svd_dim = min(svd_dim, X.shape[1])
    svd = TruncatedSVD(n_components=svd_dim, random_state=42)
    Xr = svd.fit_transform(X)
    Xr = Xr.astype('float32')
    faiss.normalize_L2(Xr)
    return Xr, vectorizer, svd


def save_meta(meta: List[dict], out_path: Path):
    with open(out_path / 'chunks_meta.jsonl', 'w', encoding='utf-8') as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')


def build_faiss_index(embs: np.ndarray, out_path: Path, use_hnsw: bool = True, efSearch: int = 64, efConstruction: int = 200):
    d = embs.shape[1]
    if use_hnsw:
        # HNSW index for cosine (Inner product after normalization)
        index = faiss.IndexHNSWFlat(d, 32)
        index.hnsw.efSearch = efSearch
        index.hnsw.efConstruction = efConstruction
        index.add(embs)
        faiss.write_index(index, str(out_path / 'faiss_index.hnsw'))
        return index, 'faiss_index.hnsw'
    else:
        nlist = max(16, int(len(embs) ** 0.5))
        quant = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embs)
        index.add(embs)
        faiss.write_index(index, str(out_path / 'faiss_index.ivf'))
        return index, 'faiss_index.ivf'


def build_bm25(texts: List[str], out_path: Path):
    if not HAVE_BM25:
        print('BM25 not available, skipping')
        return None
    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(out_path / 'bm25.pkl', 'wb') as f:
        pickle.dump({'bm25': bm25}, f)
    return bm25


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--chunks', required=True, help='Path to chunks.jsonl')
    p.add_argument('--out_dir', default='.', help='Output directory')
    p.add_argument('--use_dense', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=True, help='Use sentence-transformers dense embeddings')
    p.add_argument('--model_name', default='all-mpnet-base-v2', help='Sentence-transformers model name (local or HF)')
    p.add_argument('--use_hnsw', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=True)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    texts, meta = load_chunks(args.chunks)
    print(f'Loaded {len(texts)} chunks')

    embedding_info = {}

    if args.use_dense and HAVE_ST:
        try:
            print('Building dense embeddings with', args.model_name)
            embs, model_used = build_dense_embeddings(texts, args.model_name)
            embedding_info['type'] = 'dense'
            embedding_info['model'] = model_used
        except Exception as e:
            print('Dense embedding failed:', e)
            print('Falling back to TF-IDF+SVD')
            embs, vectorizer, svd = build_tfidf_svd(texts)
            embedding_info['type'] = 'tfidf_svd'
            embedding_info['vectorizer'] = 'tfidf_vectorizer.pkl'
            embedding_info['svd'] = 'svd_model.pkl'
            with open(out / 'tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(vectorizer, f)
            with open(out / 'svd_model.pkl', 'wb') as f:
                pickle.dump(svd, f)
    else:
        print('Using TF-IDF+SVD fallback (dense disabled or unavailable)')
        embs, vectorizer, svd = build_tfidf_svd(texts)
        embedding_info['type'] = 'tfidf_svd'
        embedding_info['vectorizer'] = 'tfidf_vectorizer.pkl'
        embedding_info['svd'] = 'svd_model.pkl'
        with open(out / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(out / 'svd_model.pkl', 'wb') as f:
            pickle.dump(svd, f)

    print('Building FAISS index...')
    index, idx_name = build_faiss_index(embs, out, use_hnsw=args.use_hnsw)
    print('FAISS index written:', idx_name)

    save_meta(meta, out)
    with open(out / 'embedding_model_info.json', 'w', encoding='utf-8') as f:
        json.dump(embedding_info, f, ensure_ascii=False, indent=2)

    # Optional BM25
    build_bm25(texts, out)

    print('Done. Index contains', index.ntotal, 'vectors with dim', embs.shape[1])


if __name__ == '__main__':
    main()
