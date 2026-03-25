# ClipSeek

Semantic video search for Adobe Premiere Pro: embed your library with **Cosmos-Embed1**, then query from a Premiere panel by text, image, or video.

## Components

| Piece | Role |
|--------|------|
| **`embed_app/`** | Standalone embedder GUI (`embed_app.py`). Walks a video folder, writes one `.pkl` per source file into your output folder (plus `processed_files.csv`). Includes its own copies of shared pickle/processor helpers. |
| **`extension/`** | CEP panel (`index.html`, `js/`, `jsx/`, `lib/`, `CSXS/`). Spawns **`io.exe`** (or `python io.py`) from this folder and sends JSON search commands over stdin. Includes **`io.py`** plus copies of **`cosmos_embedder.py`**, **`clipseek_cosmos_processor.py`**, and **`clipseek_video.py`** so the panel is self-contained when copied into Premiere’s extensions directory. |
| **`extension/io.py`** | Loads Cosmos-Embed1, loads all corpus `.pkl` files, runs similarity search, prints JSON results on stdout. |

The embedder and the panel are intentionally separate: you run embedding once (or when new media arrives); editors only need Premiere and the packaged search binary (or Python for dev).

## Quick start

1. **Model weights**  
   - Run once from the repo root:  
     `python embed_app/download.py`  
     This downloads into **`embed_app/cosmos_model/`** (gitignored).  
   - For search only, you can instead place **`cosmos_model`** next to **`extension/io.py`** (i.e. `extension/cosmos_model/`), or rely on the hub if online.

2. **Embed videos**  
   - From the repo root:  
     `python embed_app/embed_app.py`  
   - Install deps: `pip install -r embed_app/requirements-embedder.txt`  
   - Set input folder, output folder, chunk size (default 10s), overlap, workers, then start. Outputs are **`basename.pkl`** files in the output directory.

3. **Premiere panel**  
   - Copy the entire **`extension/`** folder into your CEP extensions directory (keep internal layout: `index.html`, `js/`, `lib/`, `jsx/`, `css/`, `CSXS/`, and the Python files next to `index.html` if you use `io.py`).  
   - In settings, set **Embedding folder** to the same output folder as step 2.  
   - **Production:** place **`io.exe`** under **`extension/python/io.exe`** (see `extension/js/bridge.js`).  
   - **Dev without a build:** omit `io.exe`; the panel runs **`extension/io.py`** with `python` if it sits next to `index.html`. Use **`CLIPSEEK_PYTHON`** if `python` is not on PATH.  
   - Search backend deps: `pip install -r extension/requirements-search.txt`

4. **Search**  
   - Type a query and press Enter, or use upload / drag-and-drop for image or video similarity search.

## Building `io.exe` (PyInstaller)

Work from **`extension/`** so imports resolve like the panel:

```text
cd extension
pyinstaller --onefile io.py --add-data "cosmos_model;cosmos_model"
```

Bundle **`cosmos_embedder.py`**, **`clipseek_video.py`**, **`clipseek_cosmos_processor.py`**, and the **`cosmos_model`** directory (or ship `cosmos_model` beside the exe). Paths are resolved from the directory containing `io.py` / the frozen executable; the loader also checks **`../embed_app/cosmos_model`** when both folders sit in the same repo checkout.

## Shared code (duplicated on purpose)

These files exist in **both** `extension/` and `embed_app/` so each tree can be copied or packaged alone:

- `clipseek_video.py` — pickle record + loader  
- `clipseek_cosmos_processor.py` — Cosmos processor wiring  
- `cosmos_embedder.py` — model load + text/image/video embeddings (with sibling-folder fallback for `cosmos_model`)

Keep them in sync when you change embedding or search behavior.

## Legacy notes

- An older **InternVideo2** + BERT stack (`load_model.py`, `backbones/`, `gui.py`) lived in this repo; it has been removed. Search is **Cosmos-only** and expects **`.pkl`** embeddings from **`embed_app/embed_app.py`**.  
- Old corpus files named **`{md5}.qpl`** are not used by the current `io.py`.

## Repo layout

```text
Clipseek/
  README.md
  embed_app/
    embed_app.py              # Tk embedder GUI
    download.py               # HF download → embed_app/cosmos_model/
    clipseek_video.py         # (copy) shared with extension
    clipseek_cosmos_processor.py
    cosmos_embedder.py
    requirements-embedder.txt
    cosmos_model/             # optional local weights (gitignored)
  extension/
    io.py                     # Search backend
    index.html, js/, css/, lib/, jsx/, CSXS/
    cosmos_embedder.py        # (copy) same folder as io.py for CEP
    clipseek_video.py
    clipseek_cosmos_processor.py
    requirements-search.txt
```

## What to commit (Git)

The **`.gitignore`** is set up so you mainly commit **source and small configs**: Python/JS/CEP assets, `requirements*.txt`, manifests, etc. It excludes **`cosmos_model/`** (full tree, including tokenizer/weight blobs), **embeddings (`*.pkl`)**, **logs**, **`node_modules`**, **PyInstaller output** (`extension/python/io.exe`), and common **checkpoint extensions** (`.safetensors`, `.pt`, `.pth`, …) if they appear outside an ignored folder. Run `git status` before pushing; if a large or licensed binary shows up, add a pattern or move it under an ignored path.

## License / third party

Cosmos-Embed1 is subject to the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). Use and redistribution of the weights are governed by that license and Hugging Face gating where applicable.
