# search_faiss.py - Search the offline FAISS index
import json, numpy as np, faiss, pickle
import sys

# File paths
INDEX_FILE = "faiss_index.ivf"
META_FILE = "chunks_meta.jsonl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"
SVD_FILE = "svd_model.pkl"

def load_search_components():
    """Load FAISS index, metadata, vectorizer, and SVD model"""
    print("Loading search components...")
    
    # Load FAISS index
    index = faiss.read_index(INDEX_FILE)
    
    # Load metadata
    metadata = []
    with open(META_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            metadata.append(json.loads(line))
    
    # Load original chunks with text content
    chunks_with_text = {}
    with open("data_books/chunks.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            chunks_with_text[chunk["id"]] = chunk["text"]
    
    # Load vectorizer and SVD
    with open(VECTORIZER_FILE, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(SVD_FILE, 'rb') as f:
        svd = pickle.load(f)
    
    print(f"Loaded index with {index.ntotal} vectors, {len(metadata)} metadata entries")
    return index, metadata, vectorizer, svd, chunks_with_text

def search_query(query, index, metadata, vectorizer, svd, chunks_with_text, k=5):
    """Search for similar chunks given a query"""
    # Transform query using the same pipeline
    query_tfidf = vectorizer.transform([query])
    query_vec = svd.transform(query_tfidf).astype(np.float32)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_vec)
    
    # Search
    scores, indices = index.search(query_vec, k)
    
    # Return results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:  # Valid index
            meta = metadata[idx]
            chunk_id = meta["id"]
            text_content = chunks_with_text.get(chunk_id, "Text not found")
            results.append({
                "score": float(score),
                "id": chunk_id,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", "unknown"),
                "text": text_content
            })
    
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python search_faiss.py 'your search query'")
        print("Example: python search_faiss.py 'remote sensing principles'")
        return
    
    query = " ".join(sys.argv[1:])
    print(f"Searching for: '{query}'")
    
    try:
        index, metadata, vectorizer, svd, chunks_with_text = load_search_components()
        results = search_query(query, index, metadata, vectorizer, svd, chunks_with_text, k=5)
        
        print(f"\nTop {len(results)} results:")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. RELEVANCE SCORE: {result['score']:.4f}")
            print(f"   SOURCE: {result['source']} (Page {result['page']})")
            print(f"   CHUNK ID: {result['id']}")
            print(f"   TEXT CONTENT:")
            print(f"   {'-' * 60}")
            
            # Format text content nicely
            text = result['text']
            if len(text) > 500:  # Truncate very long texts
                text = text[:500] + "..."
            
            # Add proper indentation
            lines = text.split('\n')
            for line in lines:
                print(f"   {line}")
            
            print(f"   {'-' * 60}")
            
    except FileNotFoundError as e:
        print(f"Error: Missing file {e}")
        print("Please run 'python build_faiss.py' first to create the index.")
    except Exception as e:
        print(f"Error during search: {e}")

if __name__ == "__main__":
    main()
