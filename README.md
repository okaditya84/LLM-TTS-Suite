LLM-TTS Suite — unified toolkit for LLM fine-tuning, RAG, and speech synthesis

Short description (250 chars):
LLM-TTS Suite combines LLM fine-tuning and retrieval tools with a fast, GPU-optimized TTS inference pipeline. Includes scripts for training, FAISS-based RAG, audio generation, and utilities to run and evaluate TTS models locally while keeping large checkpoints external.

Overview
--------
LLM-TTS Suite provides a compact, practical collection of tools for working with large language models and an associated TTS model. The repository contains:

- LLM fine-tuning & retrieval: `fine_tune_sarvam/` contains scripts to prepare data, build FAISS indexes, and run retrieval-augmented generation (RAG) experiments.
- TTS inference: `inference_run_version1.py` and `inference_version2.py` provide GPU-optimized, single-run and server-style TTS inference using a Parler-style TTS model.
- Utilities & artifacts: `coursework/` contains helper/downloader scripts and example PDFs used during development. `output audio/` contains example output WAVs.

Highlights
----------
- One-time model load and warmup to minimize per-request latency.
- GPU optimizations (AMP, Flash Attention checks, CUDA tuning, TorchCompile where available).
- FAISS-based search and RAG plumbing for LLM fine-tuning and retrieval experiments.
- Clear separation between model artifacts (excluded via `.gitignore`) and runnable code.

Quick start
-----------
1. Create and activate a Python virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -U pip
```

2. Install common dependencies (adjust versions as required by your GPU / CUDA):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # adjust for your CUDA
pip install transformers soundfile faiss-cpu soundfile
```

3. Do NOT commit or push large model files (e.g. `model.safetensors`). Place model artifacts in the repository root or a separate storage location and keep them listed in `.gitignore`.

TTS inference (example)
----------------------
Run the interactive CLI TTS script (one-time model load, warmup, then prompt loop):

```powershell
python inference_run_version1.py
```

Or run the server-style interface from the optimized module:

```powershell
python inference_version2.py
```

LLM fine-tuning and RAG
-----------------------
See `fine_tune_sarvam/` for:

- `prepare_data.py` — data preparation.
- `build_faiss.py` — build FAISS indices from embeddings.
- `search_faiss.py` — query the FAISS index.
- `finetune_qlora.py` — example finetuning script (QLoRA style).

Project layout
--------------
 - `.gitignore` — excludes large artifacts and local configs.
 - `inference_run_version1.py` — interactive TTS CLI with GPU optimizations.
 - `inference_version2.py` — optimized class-based TTS runner and server loop.
 - `fine_tune_sarvam/` — LLM fine-tuning, RAG, and FAISS utilities.
 - `coursework/` — helper scripts and example PDFs used during development.
 - `output audio/` — example generated audio files (small samples only).

Contributing
------------
- Open issues and PRs on GitHub. Keep PRs small and focused.
- Do not add model checkpoints or large binary artifacts to Git. Use external storage and document download/installation steps.

License & contact
-----------------
Add your preferred license file to the repository (e.g., `LICENSE`). For questions, use the repository issues or contact the maintainer.

Notes
-----
- This README is a concise guide. For function-level details, inspect script docstrings and comments in `fine_tune_sarvam/` and the inference scripts.
- If you want, I can add a `requirements.txt`, example `.env` template, or `fine_tune_sarvam/README.md` with step-by-step fine-tuning instructions.
