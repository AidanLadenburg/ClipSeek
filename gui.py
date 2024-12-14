import sys
import os
from pathlib import Path
import cv2
import time
import hashlib
import csv
import torch
from datetime import datetime
from torch import nn
import numpy as np
import logging
import math
import pickle
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QPushButton,
    QFileDialog, QMessageBox, QProgressBar, QWidget, QVBoxLayout, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from load_model import InternVideo2_Stage2
from concurrent.futures import ThreadPoolExecutor

if getattr(sys, 'frozen', False):
    import pyi_splash


class FileScanner(QThread):
    video_found = pyqtSignal(str, int)  # Emits filename and current count
    finished = pyqtSignal(list)
    status_updated = pyqtSignal(str)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.is_running = True
        self.supported_extensions = ('.mp4', '.avi', '.mov', '.mkv')

    def run(self):
        try:
            video_files = []
            video_count = 0
            
            for root, _, files in os.walk(self.folder_path):
                if not self.is_running:
                    return
                
                for file in files:
                    if not self.is_running:
                        return
                    
                    if file.lower().endswith(self.supported_extensions):
                        video_path = os.path.join(root, file)
                        video_files.append(video_path)
                        video_count += 1
                        self.video_found.emit(file, video_count)

            self.finished.emit(video_files)
            
        except Exception as e:
            logging.error(f"Error scanning files: {str(e)}")
            self.status_updated.emit(f"Error scanning files: {str(e)}")

    def stop(self):
        self.is_running = False

class video_obj():
    def __init__(self, path, clip, num_streams=4, progress_callback=None):
        self.title = os.path.basename(path)                       
        self.path = self.fix_path(path)                                        
        self.proxy = ""                                         
        self.chunks = self.encode_chunks(path, clip, num_streams, progress_callback)            
        self.audio = self.encode_audio(path)               
        self.shot_type = []
        self.datetime = datetime.fromtimestamp(os.stat(self.fix_path(path)).st_mtime)                                  
        self.annotations = []                                   
        self.transcript = ""                                    
        self.sims = []

    def encode_chunks(self, path, clip, num_streams, progress_callback):
        chunks = clip.get_vid_feat(path, chunk_size=15, num_streams=num_streams, progress_callback=progress_callback)
        #print(len(chunks))
        return chunks

    def encode_audio(self, path):
        return None
    
    def fix_path(self, path):
        working = path.replace("\\\\", "\\")
        working = working.replace("//", "/")
        working = working.replace("\\ ", "\\")
        working = working.replace("/ ", "/")
        working = working.replace("\\", "/")
        if os.path.exists(working): return working
        if os.path.exists(working.replace('.mp4', '.mov.mp4')): return working.replace('.mp4', '.mov.mp4')
        if os.path.exists(working.replace('.mp4','.mp4.mp4')): return working.replace('.mp4','.mp4.mp4')
        if os.path.exists(working.replace('.mp4','.wav.mp4')): return working.replace('.mp4','.wav.mp4')
        if os.path.exists(working.replace('.mp4','.mp3.mp4')): return working.replace('.mp4','.mp3.mp4')
        print(f'failed to find working path {path}')
        return path

    def save(self, out):
        file = open(out, 'wb')   
        pickle.dump(self, file)                    
        file.close()

class VideoProcessingWorker(QThread):
    progress_updated = pyqtSignal(int)
    chunk_progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, clip, video_files, output_folder, num_streams=4, parallel_workers=1, log_file='processed_files.csv', cache_interval=100):
        super().__init__()
        self.clip = clip
        self.video_files = video_files
        self.output_folder = output_folder
        self.log_file = log_file = os.path.join(output_folder, log_file)
        self.num_streams = num_streams
        self.parallel_workers = parallel_workers
        self.is_running = True
        self.processed_files = self.load_processed_files()
        self.cache_interval = min(len(self.video_files)//10, cache_interval)  # Number of files before writing to cache
        self.cache_file = os.path.join(output_folder, "cached_embeddings.pkl")

    def hash_file_path(self, file_path):
        """Creates a unique hash for a file path."""
        return hashlib.md5(self.fix_path(file_path).encode()).hexdigest()

    def load_processed_files(self):
        """Loads processed files from the log file."""
        processed_files = {}
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    processed_files[row[0]] = row[1]  # {hash: full_path}
        return processed_files

    def save_processed_file(self, file_hash, file_path):
        """Logs a processed file to the log file."""
        with open(self.log_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([file_hash, file_path])

    def fix_path(self, path):
        working = path.replace("\\\\", "\\")
        working = working.replace("//", "/")
        working = working.replace("\\ ", "\\")
        working = working.replace("/ ", "/")
        working = working.replace("\\", "/")
        if os.path.exists(working): return working
        if os.path.exists(working.replace('.mp4', '.mov.mp4')): return working.replace('.mp4', '.mov.mp4')
        if os.path.exists(working.replace('.mp4','.mp4.mp4')): return working.replace('.mp4','.mp4.mp4')
        if os.path.exists(working.replace('.mp4','.wav.mp4')): return working.replace('.mp4','.wav.mp4')
        if os.path.exists(working.replace('.mp4','.mp3.mp4')): return working.replace('.mp4','.mp3.mp4')
        print(f'failed to find working path {path}')
        return path

    def update_cache_file(self, files):
        """Updates the cache file with the current processed files."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(files, f)
            self.status_updated.emit("Cache updated successfully.")
        except Exception as e:
            logging.error(f"Error updating cache: {str(e)}")
            self.error_occurred.emit(f"Error updating cache: {str(e)}")

    def run(self):
        try:
            start_time = time.time()
            total_files = len(self.video_files)
            obj_list = []

            # Load cached processed files if they exist
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    obj_list = pickle.load(f)

            # Define a single video processing function
            def process_single_video(video_path):
                if not self.is_running:  # Check cancellation before starting each video
                    return None
                
                file_hash = self.hash_file_path(video_path)
                if file_hash in self.processed_files:
                    self.status_updated.emit(f"Skipping {video_path}, already processed.")
                    return None  # Skip if already processed

                try:
                    # Calculate chunks and output path
                    output_file = os.path.join(self.output_folder, f"{file_hash}.qpl")
                    self.status_updated.emit(f"Processing {video_path}")

                    # Estimate total chunks for progress tracking
                    video = cv2.VideoCapture(video_path)
                    length = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
                    total_chunks = math.ceil(length / 15)  # Update chunk size dynamically if needed
                    video.release()

                    # Progress callback
                    class ChunkTracker:
                        def __init__(self, worker, total_chunks):
                            self.worker = worker
                            self.total_chunks = total_chunks
                            self.current_chunk = 0
                        
                        def update(self, _):
                            self.current_chunk += 1
                            progress = int(100 * (self.current_chunk / self.total_chunks))
                            self.worker.chunk_progress_updated.emit(progress)
                    
                    tracker = ChunkTracker(self, total_chunks)

                    # Process video
                    obj = video_obj(video_path, self.clip, num_streams=self.num_streams, progress_callback=tracker.update)
                    obj.save(output_file)
                    obj_list.append(obj)

                    # Log the processed file
                    self.save_processed_file(file_hash, video_path)
                    self.processed_files[file_hash] = video_path  # Update in-memory tracker
                    return obj

                except Exception as e:
                    logging.error(f"Error processing {video_path}: {e}")
                    self.status_updated.emit(f"Error processing {video_path}: {e}")
                    return None

            # Use ThreadPoolExecutor for parallel video processing
            max_workers = min(self.parallel_workers, total_files)
            print(max_workers, " workers")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_single_video, vid) for vid in self.video_files]

                # Process results and update progress
                for i, future in enumerate(futures):
                    if not self.is_running:  # Check if cancellation is triggered
                        self.status_updated.emit("Processing cancelled.")
                        break
                    result = future.result()  # Wait for processing to complete
                    if result is not None:
                        self.progress_updated.emit(int(100 * ((i + 1) / total_files)))
                    # Periodically save to cache
                    if (i + 1) % self.cache_interval == 0 and self.is_running:
                        self.update_cache_file(obj_list)

            # Finalize caching and emit finished signal
            if self.is_running:
                self.update_cache_file(obj_list)
                self.finished.emit()
                self.status_updated.emit(f"Processing complete in {time.time() - start_time:.2f} seconds.")
            else:
                self.update_cache_file(obj_list)
                self.status_updated.emit("Processing was cancelled.")
            logging.info(f"Processing finished in {time.time() - start_time:.2f} seconds.")
            print(f"Processing finished in {time.time() - start_time:.2f} seconds.")

        except Exception as e:
            logging.error(f"Error in parallel processing: {e}")
            self.error_occurred.emit(str(e))

    def stop(self):
        self.is_running = False

class VideoEmbeddingGUI(QMainWindow):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip
        self.worker = None
        self.scanner = None
        self.video_files = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Video Embedding Generator")

        # Create central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Input folder selection
        self.input_label = QLabel("Select Video Folder:")
        self.input_entry = QLineEdit(self)
        self.input_button = QPushButton("Browse")
        self.input_button.clicked.connect(self.select_input_folder)

        # Video count label
        self.video_count_label = QLabel("No videos found yet")
        self.video_count_label.setStyleSheet("color: #666666;")

        # Output folder selection
        self.output_label = QLabel("Select Output Folder:")
        self.output_entry = QLineEdit(self)
        self.output_button = QPushButton("Browse")
        self.output_button.clicked.connect(self.select_output_folder)

        # Set CUDA Streams
        self.stream_label = QLabel("CUDA Streams:")
        self.stream_spinbox = QSpinBox(self)
        self.stream_spinbox.setMinimum(1)
        self.stream_spinbox.setMaximum(32)  # Set max based on GPU memory capacity
        self.stream_spinbox.setValue(4)  # Default value

        # **Parallel Workers Spinbox**
        self.parallel_label = QLabel("Parallel Workers (Videos):")
        self.parallel_spinbox = QSpinBox(self)
        self.parallel_spinbox.setMinimum(1)
        self.parallel_spinbox.setMaximum(64)  # Maximum concurrent workers
        self.parallel_spinbox.setValue(2)  # Default value

        # Processing buttons
        self.generate_button = QPushButton("Generate Embeddings")
        self.generate_button.clicked.connect(self.generate_embeddings)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setEnabled(False)

        # Processing progress
        self.overall_progress_label = QLabel("Overall Progress:")
        self.progress = QProgressBar(self)
        self.chunk_progress_label = QLabel("Current Video Progress:")
        self.chunk_progress = QProgressBar(self)
        self.status_label = QLabel("")

        # Add widgets to layout
        widgets = [
            self.input_label, self.input_entry, self.input_button,
            self.video_count_label,
            self.output_label, self.output_entry, self.output_button,
            self.stream_label, self.stream_spinbox,
            self.parallel_label, self.parallel_spinbox,  # **Add spinbox for parallel workers**
            self.generate_button, self.cancel_button,
            self.overall_progress_label, self.progress,
            self.chunk_progress_label, self.chunk_progress,
            self.status_label
        ]
        for widget in widgets:
            layout.addWidget(widget)

        self.setCentralWidget(central_widget)
        self.resize(400, 450)  # Adjust window size for added widget

    def select_input_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder with Videos")
        if folder_path:
            self.input_entry.setText(folder_path)
            self.scan_directory(folder_path)

    def scan_directory(self, folder_path):
        # Reset and disable controls during scanning
        self.toggle_controls(False)
        self.video_count_label.setText("Scanning for videos...")
        self.status_label.setText("Scanning directory...")
        self.video_files = []
        
        # Create and start scanner thread
        self.scanner = FileScanner(folder_path)
        self.scanner.video_found.connect(self.update_video_count)
        self.scanner.status_updated.connect(self.status_label.setText)
        self.scanner.finished.connect(self.scanning_finished)
        self.scanner.start()

    def update_video_count(self, filename, count):
        self.video_count_label.setText(f"{count} Videos found")
        self.status_label.setText(f"Found: {filename}")

    def scanning_finished(self, video_files):
        self.video_files = video_files
        self.toggle_controls(True)
        self.status_label.setText("Ready to process videos")

    def select_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder for Output Embeddings")
        if folder_path:
            self.output_entry.setText(folder_path)

    def generate_embeddings(self):
        if not self.video_files:
            QMessageBox.warning(self, "Warning", "No video files found in the selected directory.")
            return

        output_folder = self.output_entry.text()
        if not os.path.isdir(output_folder):
            QMessageBox.critical(self, "Error", "Please select a valid output folder.")
            return

        setup_logger()

        num_streams = self.stream_spinbox.value()
        parallel_workers = self.parallel_spinbox.value()  # Get value from the spinbox

        # Disable input controls during processing
        self.toggle_controls(False)

        # Create and start worker thread
        self.worker = VideoProcessingWorker(self.clip, self.video_files, output_folder, num_streams, parallel_workers)
        self.worker.progress_updated.connect(self.progress.setValue)
        self.worker.chunk_progress_updated.connect(self.chunk_progress.setValue)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.finished.connect(self.processing_finished)
        self.worker.error_occurred.connect(self.handle_error)

        # Pass the number of parallel workers to the `run` method
        self.worker.start()

    def processing_finished(self):
        self.toggle_controls(True)
        self.progress.setValue(0)
        self.chunk_progress.setValue(0)
        QMessageBox.information(self, "Completed", "Embeddings generation completed!")

    def handle_error(self, error_message):
        self.toggle_controls(True)
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")

    def cancel_processing(self):
        if self.scanner and self.scanner.isRunning():
            self.scanner.stop()
            self.scanner.wait()
            self.scanning_finished([])
            self.status_label.setText("Scanning cancelled")
        
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.processing_finished()
            self.status_label.setText("Processing cancelled")

    def toggle_controls(self, enabled):
        controls = [
            self.input_entry, self.input_button,
            self.output_entry, self.output_button,
            self.generate_button
        ]
        for control in controls:
            control.setEnabled(enabled)
        self.cancel_button.setEnabled(not enabled)

    def closeEvent(self, event):
        self.cancel_processing()
        event.accept()

def setup_logger():
    log_file = './output.log'
    
    # Clear existing handlers to prevent duplicate logging
    logger = logging.getLogger()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    
    # Clear the log file
    with open(log_file, 'w', encoding='utf-8'):
        pass  # Just open and close to clear the file
    
    # Configure logging settings
    logging.basicConfig(
        filename=log_file,
        filemode='w',  # Overwrite the log file each time
        level=logging.INFO,  # Log INFO level and above
        format='%(asctime)s - %(levelname)s - %(message)s',
    )


if __name__ == '__main__':
    import win32com.client

    app = QApplication(sys.argv)
    try:
        clip = torch.load("model.pt")
    except Exception as e:
        print(e)
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut('model.pt.lnk')
        target = shortcut.TargetPath
        print(target)
        clip = torch.load(target)

    gui = VideoEmbeddingGUI(clip)
    gui.show()
    if getattr(sys, 'frozen', False):
        pyi_splash.close()
    sys.exit(app.exec_())