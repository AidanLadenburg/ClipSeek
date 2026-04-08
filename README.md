# ClipSeek

Semantic video search for Adobe Premiere Pro: embed your library with **Cosmos-Embed1**, then query from a Premiere panel by text, image, or video.

## Components

| Piece | Role |
|--------|------|
| **`embed_app/`** | Standalone embedder GUI (`embed_app.py`). Walks a video folder recursively, writes **one `.pkl` per video** into the output folder, maintains **`processed_files.csv`**, and builds a **mmap corpus cache** (`cached_embeddings.matrix.npy` + `cached_embeddings.meta`) so the Premiere search backend can load quickly. |
| **`extension/`** | CEP panel (`index.html`, `js/`, `jsx/`, `lib/`, `CSXS/`). Spawns **`io.exe`** (or `python io.py`) and sends JSON search commands over stdin. Includes **`io.py`** and copies of shared Python modules so the panel is self-contained when copied into Premiere’s extensions directory. |
| **`extension/io.py`** | Loads Cosmos-Embed1, loads embeddings from the mmap cache (or per-video `.pkl` files), runs similarity search, prints JSON on stdout. Supports **exact batched** search and optional **FAISS** approximate candidates + exact rerank (see Settings in the panel). |

The embedder and the panel are separate: you run embedding when needed; editors only need Premiere and the search backend.

## Quick start

1. **Model weights**  
   - From the repo root:  
     `python embed_app/download.py`  
     Downloads into **`embed_app/cosmos_model/`** (gitignored).  
   - For search only, place **`cosmos_model`** next to **`extension/io.py`** (`extension/cosmos_model/`), or rely on the Hugging Face hub if online.

2. **Embed videos**  
   - Install: `pip install -r embed_app/requirements-embedder.txt`  
   - Run: `python embed_app/embed_app.py`  
   - Set **input** folder, **output** (embedding) folder, chunk size (default 10s), overlap, and **workers**.  
   - **Workers ≥ number of GPUs** loads one full model replica per GPU (data parallel).  
   - Output per video: **`<stem>_<12-char-hash>.pkl`** in a **flat** output folder (hash avoids basename collisions across subfolders).  
   - The app **updates the mmap corpus cache** while embedding (and on stop/finish). Use **Regenerate mmap cache** if you need to rebuild it manually.

3. **Premiere panel**  
   - Copy the entire **`extension/`** folder into your CEP extensions directory (keep `index.html`, `js/`, `css/`, `lib/`, `jsx/`, `CSXS/`, and Python files beside `index.html` if you use `io.py`).  
   - In settings, set **Embedding folder** to the embedder output folder (contains per-video `.pkl` files and the mmap cache).  
   - **Production:** place **`io.exe`** under **`extension/python/io.exe`** (see `extension/js/bridge.js`).  
   - **Dev:** omit `io.exe`; the panel runs **`extension/io.py`**. Set **`CLIPSEEK_PYTHON`** if `python` is not on PATH.  
   - Search deps: `pip install -r extension/requirements-search.txt` (includes **`faiss-cpu`** for optional approximate search).

4. **Search**  
   - Type a query (Enter) or use upload / drag-and-drop for image or video similarity.  
   - **Settings → Search mode:** **Exact** = full batched similarity over all chunks (accurate). **FAISS** = approximate nearest-neighbor probe to pick candidate videos, then **exact** scores on those candidates (faster on very large libraries). Annotation-assisted text queries always use the exact path.

## Corpus cache (mmap)

The search backend prefers **`cached_embeddings.matrix.npy`** + **`cached_embeddings.meta`** (float32 matrix + manifest). Startup mmap-loads this instead of unpickling a huge monolithic file. The embedder writes/updates this format; use **Regenerate mmap cache** in the embedder UI if the index is stale or corrupted.

## Building `io.exe` (PyInstaller)

Work from **`extension/`** so imports resolve like the panel:

```text
cd extension
pyinstaller --onefile io.py --add-data "cosmos_model;cosmos_model"
```

Include **`cosmos_model`** beside the executable (or ship under `extension/`). Also bundle **`embed_cache_v2.py`** (imported by `io.py`)—PyInstaller usually follows imports; add **`--hidden-import`** for **`faiss`** if the frozen build misses it. Same repo layout: the loader can fall back to **`../embed_app/cosmos_model`**.

## Shared code (duplicated on purpose)

These exist in **both** `extension/` and `embed_app/` so each tree can be copied or packaged alone:

- `clipseek_video.py` — on-disk embedding record + unpickler  
- `clipseek_cosmos_processor.py` — Cosmos processor wiring  
- `cosmos_embedder.py` — model load + text/image/video embeddings  
- `embed_cache_v2.py` — mmap matrix + meta read/write (search + embed cache)

Keep them in sync when you change embedding or on-disk formats.

## Repo layout (overview)

```text
ClipSeek/
  README.md
  embed_app/
    embed_app.py
    embeddings_cache.py       # mmap cache during / after embedding
    embed_cache_v2.py
    download.py
    clipseek_video.py
    clipseek_cosmos_processor.py
    cosmos_embedder.py
    requirements-embedder.txt
    cosmos_model/             # optional local weights (gitignored)
  extension/
    io.py
    embed_cache_v2.py
    index.html, js/, css/, lib/, jsx/, CSXS/
    cosmos_embedder.py
    clipseek_video.py
    clipseek_cosmos_processor.py
    requirements-search.txt
```

## What to commit

**`.gitignore`** excludes large or generated assets: **`cosmos_model/`**, **`*.pkl`**, **mmap cache files** (`cached_embeddings.matrix.npy`, `cached_embeddings.meta`), **logs**, **`node_modules`**, **`extension/python/io.exe`**, and common weight extensions. Run **`git status`** before pushing; do not commit downloaded model trees or customer embedding folders.

## License / third party

Cosmos-Embed1 is subject to the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). Use and redistribution of the weights are governed by that license and Hugging Face gating where applicable.
