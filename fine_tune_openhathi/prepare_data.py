# prepare_data.py
"""prepare_data.py

Production-ready data preparation with tokenizer-aware, semantic chunking
and hierarchical chunks (section + paragraph) plus filtering and HF
dataset export.

Key features:
 - Local tokenizer loader (works with gpt-oss-20b downloaded locally)
 - Section-aware splitting (headings, numbered sections)
 - Paragraph-level sliding windows with overlap
 - Hierarchical chunks: 'section' and 'paragraph' levels
 - Heuristic filters to remove TOC/navigation/footer noise
 - Exports: data_books/chunks.jsonl, hf_dataset/ (DatasetDict)

Run: python prepare_data.py --pdf_dir pdfs --out_dir data_books --base_model_dir /path/to/gpt-oss-20b
"""

from pathlib import Path
import argparse
import json
import os
import re
import random


try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    HAVE_HF = True
except Exception:
    AutoTokenizer = None
    PreTrainedTokenizerFast = None
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

try:
    from datasets import Dataset, DatasetDict
except Exception:
    Dataset = None
    DatasetDict = None

DEFAULT_CHUNK_TOKENS = 700
DEFAULT_SECTION_TOKENS = 1500
DEFAULT_OVERLAP = 200
DEFAULT_VAL_RATIO = 0.05


def clean_text(s):
    if not s:
        return ""
    s = s.replace("\x00", " ").replace("\u200b", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    # remove common navigation noise
    s = re.sub(r"\b(first|previous|next|last|exit|zoom)\b", "", s, flags=re.I)
    # remove repeated page header/footer patterns like 'Page 1 of 10' or standalone numbers on lines
    s = re.sub(r"^\s*page\s*\d+.*$", "", s, flags=re.I | re.M)
    s = re.sub(r"^\s*\d{1,4}\s*$", "", s, flags=re.M)
    return s.strip()


def pdf_pages_to_texts(pdf_path):
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required to read PDFs. Install it or provide plain text inputs.")
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            txt = clean_text(txt)
            if txt:
                pages.append({"page": i + 1, "text": txt})
    return pages


def read_all_pdfs(pdf_dir):
    p = Path(pdf_dir)
    pdfs = sorted(list(p.glob("*.pdf")))
    if not pdfs:
        raise FileNotFoundError("No PDFs found in {}".format(pdf_dir))
    all_pages = []
    for pdf_path in pdfs:
        pages = pdf_pages_to_texts(str(pdf_path))
        for pg in pages:
            all_pages.append({"source": pdf_path.name, "page": pg["page"], "text": pg["text"]})
    return all_pages


def load_local_tokenizer(base_dir):
    base = Path(base_dir)
    # Try transformers AutoTokenizer first (local files only)
    if HAVE_HF:
        try:
            tok = AutoTokenizer.from_pretrained(str(base), local_files_only=True, use_fast=True)
            return ("hf", tok)
        except Exception:
            pass
        # try PreTrainedTokenizerFast with tokenizer.json
        tok_file = base / "tokenizer.json"
        if tok_file.exists() and PreTrainedTokenizerFast is not None:
            try:
                tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_file))
                return ("hf", tok)
            except Exception:
                pass

    # tokenizers lib
    tok_json = base / "tokenizer.json"
    if HAVE_TOKENIZERS and tok_json.exists():
        try:
            tk = TokenizersTokenizer.from_file(str(tok_json))
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
        except Exception:
            pass

    # SentencePiece
    for sp_name in ("tokenizer.model", "spiece.model", "sentencepiece.model"):
        spf = base / sp_name
        if HAVE_SP and spf.exists():
            try:
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
                return ("sentencepiece", SPWrap(sp))
            except Exception:
                pass

    return (None, None)


def encode_ids(tok_tuple, text):
    ttype, tok = tok_tuple
    if ttype == "hf":
        return tok.encode(text, add_special_tokens=False)
    elif ttype == "tokenizers_json":
        return tok.encode(text)
    elif ttype == "sentencepiece":
        return tok.encode(text)
    else:
        # naive whitespace tokens as fallback
        return text.split()


def decode_ids(tok_tuple, ids):
    ttype, tok = tok_tuple
    if ttype == "hf":
        return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    elif ttype == "tokenizers_json":
        return tok.decode(ids)
    elif ttype == "sentencepiece":
        return tok.decode(ids)
    else:
        return " ".join(ids)


def is_noise_chunk(text):
    t = text.strip()
    if len(t) < 30:
        return True
    # high ratio of digits/symbols
    non_alpha = sum(1 for c in t if not (c.isalnum() or c.isspace()))
    if non_alpha / max(1, len(t)) > 0.3:
        return True
    # common TOC words
    toc_words = ["contents", "index", "table of contents", "chapter contents"]
    if any(w in t.lower() for w in toc_words):
        return True
    return False


def split_into_sections(text):
    """Return list of (section_text, start_char, end_char). Heuristic split by headings/numbering."""
    lines = text.splitlines()
    sections = []
    cur = []
    start_idx = 0
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(?:[0-9]+\.|[IVXLC]+\.|[A-Z][A-Za-z\- ]{3,})\s*$", ln):
            # treat as heading boundary
            if cur:
                sec_text = "\n".join(cur).strip()
                sections.append((sec_text, start_idx, i))
                cur = []
            start_idx = i
            cur.append(ln)
        else:
            cur.append(ln)
    if cur:
        sections.append(("\n".join(cur).strip(), start_idx, len(lines)))
    return sections


def chunk_text_by_tokens(tokenizer_tuple, text, chunk_tokens, overlap):
    ids = encode_ids(tokenizer_tuple, text)
    # if fallback whitespace list
    if isinstance(ids, list) and ids and isinstance(ids[0], str):
        words = ids
        chunks = []
        i = 0
        while i < len(words):
            window = words[i:i + chunk_tokens]
            chunks.append(" ".join(window))
            i += chunk_tokens - overlap
        return chunks
    # else numeric ids
    chunks = []
    i = 0
    while i < len(ids):
        window = ids[i:i + chunk_tokens]
        text_chunk = decode_ids(tokenizer_tuple, window).strip()
        chunks.append(text_chunk)
        i += chunk_tokens - overlap
    return chunks


def prepare(pdf_dir, out_dir, base_model_dir, chunk_tokens, section_tokens, overlap, val_ratio, seed):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer from:", base_model_dir)
    tokenizer_type, tokenizer = load_local_tokenizer(base_model_dir)
    if tokenizer_type is None:
        print("WARNING: No local tokenizer found. Falling back to whitespace chunking.")
    else:
        print("Tokenizer loaded:", tokenizer_type)

    pages = read_all_pdfs(pdf_dir)
    print("Read {} pages from PDFs".format(len(pages)))

    chunks = []
    meta = []
    cid = 0

    for entry in pages:
        src = entry["source"]
        page = entry["page"]
        text = entry["text"]
        # split into reasonable sections
        sections = split_into_sections(text)
        for sec_text, sstart, send in sections:
            sec_text = clean_text(sec_text)
            if not sec_text:
                continue
            # section-level chunk (coarse)
            sec_chunks = chunk_text_by_tokens((tokenizer_type, tokenizer), sec_text, section_tokens, overlap)
            for sc in sec_chunks:
                if is_noise_chunk(sc):
                    continue
                chunks.append(sc)
                meta.append({"id": cid, "source": src, "page": page, "level": "section"})
                cid += 1
            # paragraph-level: split by double-newline then sliding window
            paras = [p.strip() for p in sec_text.split("\n\n") if p.strip()]
            for p in paras:
                p = clean_text(p)
                p_chunks = chunk_text_by_tokens((tokenizer_type, tokenizer), p, chunk_tokens, overlap)
                for pc in p_chunks:
                    if is_noise_chunk(pc):
                        continue
                    chunks.append(pc)
                    meta.append({"id": cid, "source": src, "page": page, "level": "paragraph"})
                    cid += 1

    print("Created {} chunks (after filtering)".format(len(chunks)))

    chunks_file = out / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        for text_chunk, m in zip(chunks, meta):
            rec = {"id": m["id"], "text": text_chunk, "source": m.get("source"), "page": m.get("page"), "level": m.get("level")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Wrote chunks to:", chunks_file)

    # build HF dataset
    if Dataset is not None:
        data = [{"text": c, "id": m["id"], "source": m.get("source"), "page": m.get("page"), "level": m.get("level")} for c, m in zip(chunks, meta)]
        random.Random(seed).shuffle(data)
        n = len(data)
        val_n = max(1, int(n * val_ratio))
        ds = DatasetDict({"train": Dataset.from_list(data[val_n:]), "valid": Dataset.from_list(data[:val_n])})
        ds.save_to_disk(str(out / "hf_dataset"))
        print("Saved HF dataset to:", out / "hf_dataset")
    else:
        print("datasets library not available; skipping saving HF dataset")

    return str(chunks_file)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf_dir", default="pdfs", help="Directory with PDF files")
    p.add_argument("--out_dir", default="data_books", help="Output directory for chunks and datasets")
    p.add_argument("--base_model_dir", required=True, help="Local tokenizer/model directory (e.g., gpt-oss-20b folder)")
    p.add_argument("--chunk_tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    p.add_argument("--section_tokens", type=int, default=DEFAULT_SECTION_TOKENS)
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    p.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    prepare(args.pdf_dir, args.out_dir, args.base_model_dir, args.chunk_tokens, args.section_tokens, args.overlap, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
