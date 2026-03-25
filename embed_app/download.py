"""
Download Cosmos-Embed1 weights + processor into embed_app/cosmos_model.

The old script skipped entirely if the folder existed — that left config/tokenizer
without pytorch_model.bin / model.safetensors and caused your error.

Usage (from repo root):
  python embed_app/download.py

Gated model: log in first:
  huggingface-cli login
"""
import os
import glob
import json

from transformers import AutoConfig, AutoModel, AutoProcessor

MODEL_ID = "nvidia/Cosmos-Embed1-448p"
HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(HERE, "cosmos_model")


def has_model_weights(folder: str) -> bool:
    if not os.path.isdir(folder):
        return False
    if os.path.isfile(os.path.join(folder, "model.safetensors")):
        return True
    if os.path.isfile(os.path.join(folder, "pytorch_model.bin")):
        return True
    if glob.glob(os.path.join(folder, "model-*.safetensors")):
        return True
    return False


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if has_model_weights(SAVE_DIR):
        print(f"Model weights already present under:\n  {SAVE_DIR}\nNothing to do.")
        return

    if os.listdir(SAVE_DIR):
        print(
            "Folder exists but no weight files (model.safetensors / pytorch_model.bin / shards).\n"
            "Downloading full checkpoint into the same folder…\n"
        )

    print(f"Downloading {MODEL_ID} → {SAVE_DIR}\n(This can take a long time and needs disk space + HF access.)\n")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor.save_pretrained(SAVE_DIR)

    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    # shared tensors: avoid safetensors in some checkpoints
    model.save_pretrained(SAVE_DIR, safe_serialization=False)

    if not has_model_weights(SAVE_DIR):
        raise SystemExit(
            "Download finished but no weight files found. "
            "If the model is gated, run: huggingface-cli login"
        )

    # AutoProcessor needs auto_map in processor_config.json; otherwise HF falls back to AutoTokenizer
    # and video embedding fails with "You need to specify either `text` or `text_target`."
    proc_cfg_path = os.path.join(SAVE_DIR, "processor_config.json")
    if os.path.isfile(proc_cfg_path):
        cfg = AutoConfig.from_pretrained(SAVE_DIR, trust_remote_code=True)
        am = getattr(cfg, "auto_map", None) or {}
        if "AutoProcessor" in am:
            with open(proc_cfg_path, "r", encoding="utf-8") as f:
                pc = json.load(f)
            inner = dict(pc.get("auto_map") or {})
            if "AutoProcessor" not in inner:
                inner["AutoProcessor"] = am["AutoProcessor"]
                pc["auto_map"] = inner
                with open(proc_cfg_path, "w", encoding="utf-8") as f:
                    json.dump(pc, f, indent=2)
                print("Patched processor_config.json with auto_map for AutoProcessor.")

    print("Model downloaded successfully.")

    gui_config_path = os.path.join(HERE, "config.json")
    if not os.path.isfile(gui_config_path):
        cfg = {
            "input_dir": "",
            "output_dir": "",
            "chunk_size": "10",
            "overlap": "0",
            "workers": "2",
        }
        with open(gui_config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"Created default GUI config: {gui_config_path}")


if __name__ == "__main__":
    main()
