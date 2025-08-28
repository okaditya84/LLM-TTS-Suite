# prepare_data.py
"""
Offline-safe data preparation that uses local tokenizer artifacts if present:
- tries: transformers.AutoTokenizer(local_files_only=True)
- fallback: tokenizers.Tokenizer.from_file(tokenizer.json)
- fallback: sentencepiece tokenizer.model / spiece.model
- final fallback: naive char/word chunking

Outputs:
  - data_books/chunks.jsonl   (id, text, source, page, start_token)
  - hf_dataset/               (HF Dataset saved to disk: train + valid)
"""

import os
import re
import json
import random
from pathlib import Path
import pdfplumber
from datasets import Dataset, DatasetDict

# optional imports - handle missing packages gracefully
try:
    from transformers import AutoTokenizer
    HAVE_HF = True
except Exception:
    AutoTokenizer = None
    HAVE_HF = False

try:
    from tokenizers import Tokenizer as TokenizersTokenizer
    HAVE_TOKENIZERS = True
except Exception:
    TokenizersTokenizer = None
    HAVE_TOKENIZERS = False

try:
    import sentencepiece as spm
    HAVE_SP = True
except Exception:
    spm = None
    HAVE_SP = False

# ------------- CONFIG -------------
PDF_DIR = "pdfs"   # put your PDFs here
OUT_DIR = "data_books"
HF_DIR = "hf_dataset"
# point to your downloaded Sarvam-1 tokenizer directory (offline)
BASE_MODEL_DIR = "/nlsasfs/home/ledgerptf/ashsa/models/sarvam1"  # <--- correct local model folder path
CHUNK_TOKENS = 700
OVERLAP_TOKENS = 120
VAL_RATIO = 0.05
SEED = 42
# ----------------------------------

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(HF_DIR, exist_ok=True)

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\x00", " ").replace("\u200b", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def pdf_pages_to_texts(pdf_path: str):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            txt = clean_text(txt)
            if txt:
                pages.append({"page": i+1, "text": txt})
    return pages

def read_all_pdfs(pdf_dir: str):
    all_pages = []
    p = Path(pdf_dir)
    pdfs = sorted(list(p.glob("*.pdf")))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")
    for pdf_path in pdfs:
        pages = pdf_pages_to_texts(str(pdf_path))
        for pg in pages:
            all_pages.append({"source": pdf_path.name, "page": pg["page"], "text": pg["text"]})
    return all_pages

# ---------------- local tokenizer loader ----------------
def load_local_tokenizer(base_dir: str):
    base = Path(base_dir)
    # 1) Try transformers AutoTokenizer (local files)
    if HAVE_HF:
        try:
            print("Trying transformers.AutoTokenizer.from_pretrained(local_files_only=True)...")
            tok = AutoTokenizer.from_pretrained(str(base), local_files_only=True, use_fast=True)
            print("Loaded tokenizer via transformers.")
            return ("hf", tok)
        except Exception as e:
            print("AutoTokenizer failed (expected in some offline setups):", e)

        # 1b) Try loading a fast tokenizer directly from tokenizer.json using transformers
        tok_file = base / "tokenizer.json"
        if HAVE_HF and tok_file.exists():
            try:
                from transformers import PreTrainedTokenizerFast
                print("Trying PreTrainedTokenizerFast with tokenizer.json...")
                tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_file))
                print("Loaded tokenizer via PreTrainedTokenizerFast.")
                return ("hf", tok)
            except Exception as e:
                print("PreTrainedTokenizerFast load failed:", e)

    # 2) Try tokenizers' tokenizer.json
    tok_json = base / "tokenizer.json"
    if HAVE_TOKENIZERS and tok_json.exists():
        try:
            print("Trying tokenizers.Tokenizer.from_file(tokenizer.json)...")
            tk = TokenizersTokenizer.from_file(str(tok_json))
            print("Loaded tokenizer from tokenizer.json (tokenizers lib).")
            class TKWrap:
                def __init__(self, tk):
                    self._tk = tk
                def encode(self, text, add_special_tokens=False):
                    return self._tk.encode(text).ids
                def decode(self, ids):
                    return self._tk.decode(ids)
                @property
                def pad_token_id(self):
                    return None
            return ("tokenizers_json", TKWrap(tk))
        except Exception as e:
            print("tokenizer.json load failed:", e)

    # 3) Try SentencePiece model files
    for sp_name in ("tokenizer.model", "spiece.model", "sentencepiece.model"):
        spf = base / sp_name
        if HAVE_SP and spf.exists():
            try:
                print(f"Trying SentencePieceProcessor load: {sp_name}")
                sp = spm.SentencePieceProcessor(model_file=str(spf))
                class SPWrap:
                    def __init__(self, sp):
                        self._sp = sp
                    def encode(self, text, add_special_tokens=False):
                        return self._sp.encode(text, out_type=int)
                    def decode(self, ids):
                        return self._sp.decode(ids)
                    @property
                    def pad_token_id(self):
                        return None
                print("Loaded SentencePiece tokenizer.")
                return ("sentencepiece", SPWrap(sp))
            except Exception as e:
                print("SentencePiece load failed:", e)

    # 4) nothing found
    return (None, None)

# wrapper helpers to unify interface
def encode_ids(tok_tuple, text):
    ttype, tok = tok_tuple
    if ttype == "hf":
        return tok.encode(text, add_special_tokens=False)
    elif ttype == "tokenizers_json":
        return tok.encode(text)
    elif ttype == "sentencepiece":
        return tok.encode(text)
    else:
        # fallback: naive char-coded ids (not ideal)
        return [ord(c) % 256 for c in text[:5000]]

def decode_ids(tok_tuple, ids):
    ttype, tok = tok_tuple
    if ttype == "hf":
        return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    elif ttype == "tokenizers_json":
        return tok.decode(ids)
    elif ttype == "sentencepiece":
        return tok.decode(ids)
    else:
        return "".join(chr(i) for i in ids)

# ---------------- main ----------------
print("Loading tokenizer from base model dir:", BASE_MODEL_DIR)
tokenizer_type, tokenizer = load_local_tokenizer(BASE_MODEL_DIR)
if tokenizer_type is None:
    print("WARNING: No local tokenizer found. Falling back to character-based chunking.")
else:
    print("Tokenizer type:", tokenizer_type)

print("Reading PDFs from:", PDF_DIR)
pages = read_all_pdfs(PDF_DIR)
print(f"Found {len(pages)} pages across PDFs.")

# Build token id stream across pages for tokenizer-consistent chunking
all_ids = []
page_index_for_token = []
for entry in pages:
    txt = entry["text"]
    ids = encode_ids((tokenizer_type, tokenizer), txt)
    start_idx = len(all_ids)
    all_ids.extend(ids)
    end_idx = len(all_ids)
    page_index_for_token.append({
        "source": entry["source"],
        "page": entry["page"],
        "start_token": start_idx,
        "end_token": end_idx
    })

print("Total tokens/ids (approx):", len(all_ids))

# chunk windows
chunks = []
meta = []
i = 0
while i < len(all_ids):
    window = all_ids[i:i+CHUNK_TOKENS]
    if not window:
        break
    chunk_text = decode_ids((tokenizer_type, tokenizer), window).strip()
    if len(chunk_text) > 50:
        chunks.append(chunk_text)
        # find source/page
        src = None
        for pinfo in page_index_for_token:
            if pinfo["start_token"] <= i < pinfo["end_token"]:
                src = {"source": pinfo["source"], "page": pinfo["page"], "start_token": i}
                break
        meta.append(src or {})
    i += CHUNK_TOKENS - OVERLAP_TOKENS

print("Created chunks:", len(chunks))

# save chunks jsonl
chunks_file = os.path.join(OUT_DIR, "chunks.jsonl")
with open(chunks_file, "w", encoding="utf-8") as f:
    for idx, (c, m) in enumerate(zip(chunks, meta)):
        rec = {"id": idx, "text": c}
        rec.update({"source": m.get("source"), "page": m.get("page"), "start_token": m.get("start_token")})
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("Wrote chunks to:", chunks_file)

# build HF dataset train/valid
data = [{"text": c} for c in chunks]
random.shuffle(data)
n = len(data)
val_n = max(1, int(n * VAL_RATIO))
valid = data[:val_n]
train = data[val_n:]

print("Train size:", len(train), "Valid size:", len(valid))
train_ds = Dataset.from_list(train)
valid_ds = Dataset.from_list(valid)
ds = DatasetDict({"train": train_ds, "valid": valid_ds})
ds.save_to_disk(HF_DIR)
print("Saved dataset to:", HF_DIR)
print("DONE: prepare_data completed.")
