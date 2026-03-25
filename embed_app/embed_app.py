import os
import sys
import pickle

import threading
import gc
import csv
import json
import hashlib
import time
import tkinter as tk
import logging
import concurrent.futures
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

# AI & Processing Imports
import torch
import numpy as np
import decord
from transformers import AutoModel

from clipseek_video import ClipseekVideo as video_obj
from clipseek_cosmos_processor import assert_video_processor, load_cosmos_processor

# --- Logging Setup ---
def setup_logger():
    log_file = 'embedder.log'
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )

setup_logger()

# --- Helper Functions ---
def fix_path(path):
    return os.path.abspath(path)

def get_file_hash(path):
    return hashlib.md5(fix_path(path).encode('utf-8')).hexdigest()

# --- AI Logic ---
class VideoEmbedder:
    def __init__(self, model_id="nvidia/Cosmos-Embed1-448p"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Smart Dtype Selection
        self.dtype = torch.float32 
        if self.device == "cuda":
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16

        # Local weights next to this script (not cwd — works from any launch directory)
        _here = os.path.dirname(os.path.abspath(__file__))
        local_cosmos = os.path.join(_here, "cosmos_model")
        if os.path.isfile(os.path.join(local_cosmos, "config.json")):
            self.model_id = local_cosmos
        else:
            self.model_id = model_id
            
        self.status_callback = None
        self.total_progress_callback = None
        self.worker_progress_callback = None
        self.model = None
        self.processor = None
        
        # Concurrency Controls
        self.gpu_lock = threading.Lock() 
        self.progress_lock = threading.Lock()
        self.csv_lock = threading.Lock()
        self.stop_event = threading.Event()

    def load_model(self):
        if self.model is None:
            with self.gpu_lock:
                if self.model is None:
                    self.update_status(f"Loading Model on {self.device.upper()} ({self.dtype})...")
                    logging.info(f"Loading Model on {self.device.upper()} ({self.dtype})...")
                    try:
                        self.processor = load_cosmos_processor(self.model_id, trust_remote_code=True)
                        assert_video_processor(self.processor)
                        self.model = AutoModel.from_pretrained(
                            self.model_id, 
                            trust_remote_code=True,
                            torch_dtype=self.dtype
                        ).to(self.device)
                        self.model.eval()
                    except Exception as e:
                        logging.error(f"Error loading model: {str(e)}")
                        self.update_status(f"Error loading model: {str(e)}")
                        raise e

    def update_status(self, message):
        if self.status_callback:
            self.status_callback(message)

    def load_processed_cache(self, output_dir):
        csv_path = os.path.join(output_dir, 'processed_files.csv')
        processed = set()
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row: processed.add(row[0])
            except Exception as e:
                logging.error(f"Failed to load cache: {e}")
        return processed

    def save_processed_cache(self, output_dir, file_hash, file_path):
        csv_path = os.path.join(output_dir, 'processed_files.csv')
        with self.csv_lock:
            try:
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([file_hash, file_path])
            except Exception as e:
                logging.error(f"Failed to write cache: {e}")

    def get_vid_feat(self, video_path, chunk_size=10, overlap=0, progress_callback=None):
        try:
            vr = decord.VideoReader(video_path)
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            duration = total_frames / fps
        except Exception as e:
            logging.error(f"Failed to read video {video_path}: {e}")
            raise e
        
        embeddings = []
        
        stride = chunk_size - overlap
        if stride <= 0: stride = chunk_size
        
        start_times = []
        curr_time = 0.0
        while curr_time < duration:
            start_times.append(curr_time)
            curr_time += stride
            
        num_chunks = len(start_times)
        
        for i, start_time in enumerate(start_times):
            if self.stop_event.is_set():
                return None

            if progress_callback:
                progress_callback(i, num_chunks)

            end_time = min(start_time + chunk_size, duration)
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            if start_frame >= total_frames: break
            
            chunk_len = end_frame - start_frame
            if chunk_len < 1: continue

            actual_frames = min(8, chunk_len)
            frame_ids = np.linspace(start_frame, min(end_frame, total_frames-1), actual_frames, dtype=int).tolist()

            try:
                frames = vr.get_batch(frame_ids).asnumpy()
                if frames.shape[0] == 0: continue

                batch = np.transpose(np.expand_dims(frames, 0), (0, 1, 4, 2, 3))
                
                with self.gpu_lock:
                    video_inputs = self.processor(videos=batch)
                    video_inputs = {k: v.to(self.device) for k, v in video_inputs.items()}
                    
                    with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                        with torch.no_grad():
                            video_out = self.model.get_video_embeddings(**video_inputs)
                    
                    chunk_emb = video_out.visual_proj.float().cpu().numpy()

                embeddings.append(chunk_emb)

                del video_inputs, video_out, batch, frames
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                logging.error(f"Error chunk {i} in {os.path.basename(video_path)}: {e}")
                continue

        if progress_callback:
            progress_callback(num_chunks, num_chunks)

        del vr
        gc.collect()
        return embeddings

    def process_single_video(self, file_info):
        if self.stop_event.is_set():
            return

        idx, filename, input_dir, output_dir, chunk_size, overlap = file_info
        video_path = os.path.join(input_dir, filename)
        
        thread_id = threading.get_ident()
        
        def worker_callback(curr, total):
            if self.worker_progress_callback:
                self.worker_progress_callback(thread_id, filename, curr, total)

        try:
            logging.info(f"Start processing: {filename}")
            
            vid_obj = video_obj(
                video_path, 
                self, 
                chunk_size=chunk_size, 
                overlap=overlap,
                progress_callback=worker_callback
            )
            
            if self.stop_event.is_set():
                logging.info(f"Aborted save for {filename}")
                return

            pkl_name = os.path.splitext(filename)[0] + ".pkl"
            output_path = os.path.join(output_dir, pkl_name)
            vid_obj.save(output_path)
            
            file_hash = get_file_hash(video_path)
            self.save_processed_cache(output_dir, file_hash, video_path)

            logging.info(f"Finished: {filename}")

        except Exception as e:
            err_msg = f"Failed on {filename}: {str(e)}"
            logging.error(err_msg)
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
        finally:
            gc.collect()
            with self.progress_lock:
                if self.total_progress_callback:
                    self.total_progress_callback(1)

    def process_folder(self, input_dir, output_dir, chunk_size, overlap, max_workers=1):
        self.stop_event.clear()
        self.load_model()
        
        valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        all_files = []
        
        self.update_status("Scanning directories...")
        logging.info(f"Scanning {input_dir} recursively...")
        
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if self.stop_event.is_set(): break
                if file.lower().endswith(valid_extensions):
                    all_files.append((file, root))

        total_files = len(all_files)
        logging.info(f"Found {total_files} video files.")
        
        if total_files == 0:
            self.update_status("No video files found.")
            return

        os.makedirs(output_dir, exist_ok=True)
        
        processed_hashes = self.load_processed_cache(output_dir)
        tasks = []
        
        for i, (fname, root) in enumerate(all_files):
            fpath = os.path.join(root, fname)
            fhash = get_file_hash(fpath)
            
            if fhash in processed_hashes:
                if self.total_progress_callback:
                    self.total_progress_callback(1) 
            else:
                tasks.append((i, fname, root, output_dir, chunk_size, overlap))

        skipped_count = total_files - len(tasks)
        if skipped_count > 0:
            logging.info(f"Skipped {skipped_count} already processed files.")

        logging.info(f"Starting processing {len(tasks)} files with {max_workers} workers.")
        self.update_status(f"Queued {len(tasks)} files (Skipped {skipped_count}). Starting...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_single_video, task) for task in tasks]
            concurrent.futures.wait(futures)

        if self.stop_event.is_set():
            self.update_status("Processing Stopped.")
            logging.info("Processing Stopped by user.")
        else:
            self.update_status("Done! Processing complete.")
            logging.info("Batch processing complete.")

# --- GUI Logic ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Embedder Tool")
        self.root.geometry("600x750") # Slightly taller for extra button
        
        self.config_file = "config.json"
        
        self.embedder = VideoEmbedder()
        self.embedder.status_callback = self.update_status
        self.embedder.total_progress_callback = self.increment_total_bar
        self.embedder.worker_progress_callback = self.update_worker_bar
        
        self.total_processed_count = 0
        self.total_files_count = 0
        self.worker_ui_map = {} 
        self.worker_widgets = [] 

        # Input Path
        tk.Label(root, text="Input Video Folder (Recursive):").pack(pady=5)
        self.input_entry = tk.Entry(root, width=60)
        self.input_entry.pack(pady=2)
        tk.Button(root, text="Browse...", command=self.select_input).pack(pady=2)

        # Output Path
        tk.Label(root, text="Output Embedding Folder:").pack(pady=5)
        self.output_entry = tk.Entry(root, width=60)
        self.output_entry.pack(pady=2)
        tk.Button(root, text="Browse...", command=self.select_output).pack(pady=2)

        # Settings
        settings_frame = tk.Frame(root)
        settings_frame.pack(pady=10)

        # Chunk Size
        tk.Label(settings_frame, text="Chunk Size (s):").grid(row=0, column=0, padx=5)
        self.chunk_entry = tk.Entry(settings_frame, width=5)
        self.chunk_entry.insert(0, "10") 
        self.chunk_entry.grid(row=0, column=1, padx=5)

        # Overlap
        tk.Label(settings_frame, text="Overlap (s):").grid(row=0, column=2, padx=5)
        self.overlap_entry = tk.Entry(settings_frame, width=5)
        self.overlap_entry.insert(0, "0") 
        self.overlap_entry.grid(row=0, column=3, padx=5)

        # Workers
        tk.Label(settings_frame, text="Workers:").grid(row=0, column=4, padx=5)
        self.worker_entry = tk.Entry(settings_frame, width=5)
        self.worker_entry.insert(0, "2") 
        self.worker_entry.grid(row=0, column=5, padx=5)

        # Control Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        
        self.save_btn = tk.Button(btn_frame, text="Save Settings", command=self.save_config_gui, bg="#e0e0e0", height=2, width=15)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.start_btn = tk.Button(btn_frame, text="START PROCESSING", command=self.start_thread, bg="#dddddd", height=2, width=20)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="STOP", command=self.stop_process, bg="#ffcccc", height=2, width=10, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Total Progress
        tk.Label(root, text="Total Queue Progress:").pack(pady=(10, 2))
        self.total_progress = ttk.Progressbar(root, orient="horizontal", length=500, mode="determinate")
        self.total_progress.pack(pady=2)
        self.total_label = tk.Label(root, text="0 / 0", fg="black")
        self.total_label.pack(pady=2)

        # Worker Status Area
        tk.Label(root, text="Active Workers:").pack(pady=(15, 5))
        self.worker_frame = tk.Frame(root)
        self.worker_frame.pack(pady=5, fill="both", expand=True)

        # Status Label
        self.status_label = tk.Label(root, text="Ready", fg="blue", wraplength=580)
        self.status_label.pack(pady=10)

        # Load Config on Startup
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    
                self.input_entry.delete(0, tk.END)
                self.input_entry.insert(0, config.get("input_dir", ""))
                
                self.output_entry.delete(0, tk.END)
                self.output_entry.insert(0, config.get("output_dir", ""))
                
                self.chunk_entry.delete(0, tk.END)
                self.chunk_entry.insert(0, str(config.get("chunk_size", "10")))
                
                self.overlap_entry.delete(0, tk.END)
                self.overlap_entry.insert(0, str(config.get("overlap", "0")))
                
                self.worker_entry.delete(0, tk.END)
                self.worker_entry.insert(0, str(config.get("workers", "2")))
                
            except Exception as e:
                print(f"Failed to load config: {e}")

    def save_config_gui(self):
        self.save_config()
        messagebox.showinfo("Saved", "Settings saved successfully.")

    def save_config(self):
        config = {
            "input_dir": self.input_entry.get(),
            "output_dir": self.output_entry.get(),
            "chunk_size": self.chunk_entry.get(),
            "overlap": self.overlap_entry.get(),
            "workers": self.worker_entry.get()
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")

    def select_input(self):
        path = filedialog.askdirectory()
        if path:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, path)

    def select_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, path)

    def update_status(self, text):
        self.root.after(0, lambda: self.status_label.config(text=text))

    def setup_worker_ui(self, num_workers):
        for widget in self.worker_frame.winfo_children():
            widget.destroy()
        
        self.worker_ui_map.clear()
        self.worker_widgets.clear()
        
        for i in range(num_workers):
            frame = tk.Frame(self.worker_frame)
            frame.pack(fill="x", padx=20, pady=2)
            
            lbl = tk.Label(frame, text=f"Worker {i+1}: Idle", width=30, anchor="w", font=("Arial", 8))
            lbl.pack(side=tk.LEFT)
            
            pb = ttk.Progressbar(frame, orient="horizontal", length=300, mode="determinate")
            pb.pack(side=tk.RIGHT)
            
            self.worker_widgets.append((lbl, pb))

    def update_worker_bar(self, thread_id, filename, current, total):
        self.root.after(0, lambda: self._update_worker_ui_safe(thread_id, filename, current, total))

    def _update_worker_ui_safe(self, thread_id, filename, current, total):
        if thread_id not in self.worker_ui_map:
            slot_idx = len(self.worker_ui_map) 
            if slot_idx < len(self.worker_widgets):
                self.worker_ui_map[thread_id] = slot_idx
            else:
                return

        slot_idx = self.worker_ui_map[thread_id]
        lbl, pb = self.worker_widgets[slot_idx]
        
        short_name = (filename[:25] + '..') if len(filename) > 25 else filename
        lbl.config(text=f"Worker {slot_idx+1}: {short_name}")
        
        if total > 0:
            pct = (current / total) * 100
            pb.config(value=pct)
            
        if current >= total:
            lbl.config(text=f"Worker {slot_idx+1}: Idle")
            pb.config(value=0)

    def increment_total_bar(self, increment_val):
        self.total_processed_count += increment_val
        pct = 0
        if self.total_files_count > 0:
            pct = (self.total_processed_count / self.total_files_count) * 100
        
        self.root.after(0, lambda: self._update_total_safe(pct, self.total_processed_count, self.total_files_count))

    def _update_total_safe(self, pct, current, total):
        self.total_progress.config(value=pct)
        self.total_label.config(text=f"{current} / {total}")

    def stop_process(self):
        if self.embedder:
            self.embedder.stop_event.set()
            self.status_label.config(text="Stopping... Waiting for current chunks to finish.")
            self.stop_btn.config(state=tk.DISABLED)

    def start_thread(self):
        input_dir = self.input_entry.get()
        output_dir = self.output_entry.get()
        
        try:
            chunk_size = float(self.chunk_entry.get())
            overlap = float(self.overlap_entry.get())
            workers = int(self.worker_entry.get())
            if chunk_size <= 0 or workers <= 0: raise ValueError
            if overlap >= chunk_size:
                messagebox.showerror("Error", "Overlap must be less than Chunk Size.")
                return
            if overlap < 0: raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Check Settings (Chunk, Overlap, Workers).")
            return

        if not input_dir or not output_dir:
            messagebox.showerror("Error", "Please select both folders.")
            return
        
        count = 0
        valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    count += 1
        
        self.total_files_count = count
        self.total_processed_count = 0
        self._update_total_safe(0, 0, count)
        
        self.setup_worker_ui(workers)
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        t = threading.Thread(target=self.run_process, args=(input_dir, output_dir, chunk_size, overlap, workers))
        t.start()

    def run_process(self, input_dir, output_dir, chunk_size, overlap, workers):
        self.embedder.process_folder(input_dir, output_dir, chunk_size, overlap, max_workers=workers)
        
        self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
        if not self.embedder.stop_event.is_set():
            messagebox.showinfo("Success", "Processing Finished!")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()