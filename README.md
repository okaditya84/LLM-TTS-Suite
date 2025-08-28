LLM + TTS unified toolkit

This repository contains:
- Fine-tuning scripts for an LLM (fine_tune_sarvam/)
- RAG pipeline utilities and search (fine_tune_sarvam/search_faiss.py, etc.)
- TTS inference scripts (inference_run_version1.py, inference_version2.py)
- Convenience scripts and data preparation tools

Repository goals
- Provide reproducible workflows for finetuning, retrieval, and TTS inference
- Keep code modular so LLM and TTS components can evolve independently

Quick start
1. Create a virtual environment and install dependencies (project-specific requirements are in subfolders).
2. Review `fine_tune_sarvam/README.md` or script docstrings for per-task instructions.

Contributing
- Open issues and PRs on GitHub.
- Keep large model checkpoints out of the repository; use external storage.
