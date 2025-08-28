# build_faiss.py - Offline compatible version
import json, numpy as np, faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle

CHUNKS = "data_books/chunks.jsonl"
INDEX_OUT = "faiss_index.ivf"
META_OUT = "chunks_meta.jsonl"
VECTORIZER_OUT = "tfidf_vectorizer.pkl"
SVD_OUT = "svd_model.pkl"

print("Loading chunks...")
texts, meta = [], []
with open(CHUNKS,"r",encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        texts.append(obj["text"])
        meta.append({"id": obj["id"], "source": obj.get("source"), "page": obj.get("page")})

print(f"Loaded {len(texts)} text chunks")

# Create TF-IDF vectors (offline alternative to sentence transformers)
print("Creating TF-IDF vectors...")
vectorizer = TfidfVectorizer(
    max_features=10000,  # Limit vocabulary size
    stop_words='english',
    ngram_range=(1, 2),  # Include bigrams
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.95  # Ignore terms that appear in more than 95% of documents
)

# Fit and transform texts
tfidf_matrix = vectorizer.fit_transform(texts)
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# Reduce dimensionality using SVD for better performance
target_dim = min(384, tfidf_matrix.shape[1])  # Similar to sentence transformers dimension
print(f"Reducing dimensionality to {target_dim} using SVD...")

svd = TruncatedSVD(n_components=target_dim, random_state=42)
X = svd.fit_transform(tfidf_matrix)
X = X.astype(np.float32)

print(f"Final embedding shape: {X.shape}")

# Save vectorizer and SVD model for later use
with open(VECTORIZER_OUT, 'wb') as f:
    pickle.dump(vectorizer, f)
with open(SVD_OUT, 'wb') as f:
    pickle.dump(svd, f)

# Build FAISS index
d = X.shape[1]
nlist = max(16, int(len(X)**0.5))  # coarse centroids
print(f"Building FAISS index with {nlist} centroids...")

quant = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)

# Normalize vectors for cosine similarity
faiss.normalize_L2(X)
index.train(X)
index.add(X)

# Save index and metadata
faiss.write_index(index, INDEX_OUT)
with open(META_OUT,"w",encoding="utf-8") as f:
    for m in meta:
        f.write(json.dumps(m, ensure_ascii=False)+"\n")

print(f"Wrote: {INDEX_OUT}, {META_OUT}, {VECTORIZER_OUT}, {SVD_OUT}")
print(f"Index contains {index.ntotal} vectors with {d} dimensions")
print("FAISS index built successfully using offline TF-IDF + SVD!")
