import sys
import json
import pickle
import os
import torch
import time
import math
import hashlib
from datetime import datetime
import gc
from concurrent.futures import ThreadPoolExecutor
from numpy.linalg import norm
import warnings
warnings.filterwarnings("ignore")

from load_model import InternVideo2_Stage2

class video_obj():
    def __init__(self, path, clip):
        self.title = os.path.basename(path)                       
        self.path = path                                        
        self.proxy = ""                                         
        self.chunks = self.encode_chunks(path, clip)            
        self.audio = self.encode_audio(path)                    
        self.shot_type = []
        self.datetime = None                                    
        self.annotations = []                                   
        self.transcript = ""                                    
        self.sims = []

    def encode_chunks(self, path, clip):
        chunks = clip.get_vid_feat(path, chunk_size=15)
        return chunks

    def encode_audio(self, path):
        return None

    def save(self, out):
        with open(out, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

class annotation_obj():
    def __init__(self, key):
        self.key = key
        self.len = 0
        self.values = {"imgs": [], "text": []}
        self.mean = 0
        #ADD SOMETHING TO SAVE EMBEDDINGS SIM

    def add_value(self, value, t, clip):
        if t == "text":
            self.values["text"].append(value)
        if t == "image" or "video":
            encoding = clip.get_img_feat(value)
            self.values["imgs"].append(encoding)

    def save(self, out):
        file = open(out, 'wb')
        pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)                    
        file.close()

class PersistentSearch:
    def __init__(self, video_folder=None, embedding_folder=None):
        print("Startup...", flush=True)
        with open(os.devnull, "w") as f:
            sys.stdout = f
            base = os.path.dirname(__file__).replace('\\', '/')
            print("loading model", flush=True)
            try:
                self.model = torch.load(f"{base}/model.pt")
            except Exception as e:
                import win32com.client
                shell = win32com.client.Dispatch("WScript.Shell")
                shortcut = shell.CreateShortCut(f"{base}/model.pt.lnk")
                target = shortcut.TargetPath
                self.model = torch.load(target)
        sys.stdout = sys.__stdout__
        #sys.stdout.reconfigure(encoding='utf-8')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cached_objects = None
        self.current_embedding_folder = embedding_folder
        self.current_video_folder = video_folder
        if embedding_folder:
            print("loading startup embedding folder", flush=True)
            self.cached_objects = self.load_objs(video_folder, embedding_folder)
        print("READY", flush=True)

    def cos(self, A, B, normalize=True):
        A, B = A.cpu(), B.cpu()
        if normalize:
            return (A @ B.T)/(norm(A)*norm(B, axis=1))
        return A @ B.T

    def update_embedding_folder(self, video_folder, new_folder):
        self.current_embedding_folder = new_folder
        self.current_video_folder = video_folder
        self.cached_objects = self.load_objs(video_folder, new_folder)
        print("Embedding folder updated and embeddings reloaded.", flush=True)

    def create_annotation(self, folder, key, annotation_type, value):
        """
        Create and store an annotation based on provided key, type, and value.
        """
        if not os.path.exists(folder): 
            print("NO ANNOTATION FOLDER FOUND", flush=True)
            return
        if os.path.exists(os.path.join(folder, f"{key}.anno")):
            print(f"adding {value} to {key} annotation", flush=True)
            with open(os.path.join(folder, f"{key}.anno"), 'rb') as f:
                current = pickle.load(f)
        else:
            print(f"creating new annotation: {key}", flush=True)
            current = annotation_obj(key)

        if isinstance(value, list): 
            for img_path in value: current.add_value(img_path, annotation_type, self.model)
        else: current.add_value(value, annotation_type, self.model)
        if len(current.values['imgs']) > 0: current.mean = torch.stack(current.values['imgs']).mean(dim=0)
        current.len+=1
        current.save(f"{folder}\\{key}.anno")
        
        print(f"Annotation created: {json.dumps(f"{key}, {annotation_type}, {value}")}", flush=True)

    def search_file(self, file_path, query_type, video_folder, embedding_folder, annotation_folder):
        try:
            # Load video objects
            vid_objs = self.load_objs(video_folder, embedding_folder)

            # Embed the file based on its type
            if query_type == 'image':
                query_feat = self.model.get_img_feat(file_path).to(self.device)
            elif query_type == 'video':
                temp_obj = video_obj(file_path, self.model)
                query_feat = temp_obj.chunks.to(self.device)
            else:
                return {"error": "Unsupported query type"}

            # Prepare all chunks for similarity computation
            all_chunks = [vid_obj.chunks.to(self.device) for vid_obj in vid_objs]
            stacked_chunks = torch.cat(all_chunks, dim=0)

            # Calculate cosine similarity
            with torch.no_grad():
                similarities = (query_feat @ stacked_chunks.T) / (
                    torch.norm(query_feat) * torch.norm(stacked_chunks, dim=1)
                )

            # Split similarities by each video's chunks
            split_sizes = [len(vid_obj.chunks) for vid_obj in vid_objs]
            sims_split = torch.split(similarities, split_sizes, dim=1)

            # Calculate mean similarity for each video
            final_sims = torch.stack([torch.mean(sim) for sim in sims_split])

            # Get top results
            top_values, top_indices = torch.topk(final_sims, k=min(len(vid_objs), 500), largest=True, sorted=True)
            top_vids = [vid_objs[i] for i in top_indices]

            # Return results as a list of file paths and timestamps
            return [(i.path, (torch.argmax(sims_split[idx]) * 15).item()) for idx, i in enumerate(top_vids)]

        except Exception as e:
            print(f"error with image search {e}", flush=True)
            return []
    
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

    def filter_metadata(self, vid_objs, date_from=None, date_to=None, embeddings_folder=None):
        """
        Filter video objects by date and ensure metadata consistency.
        Fixes broken paths, removes old hashed files for corrected objects, and updates the cache accordingly.
        """
        if not vid_objs:
            return []

        changed = False
        filtered_objects = []
        updated_cache = []

        # Convert date strings to datetime objects
        date_from = datetime.strptime(date_from, '%Y-%m-%d') if date_from else None
        date_to = datetime.strptime(date_to, '%Y-%m-%d') if date_to else None

        for obj in vid_objs:
            try:
                # Check if 'datetime' attribute exists
                if not hasattr(obj, 'datetime'):
                    print(f"Object {obj.path} missing datetime: please re-encode/fix files", flush=True)
                    continue
                    """obj_path = self.fix_path(obj.path)

                    if os.path.exists(obj_path):
                        # Compute the old hash based on the original (broken) path
                        old_hash = hashlib.md5(obj.path.encode()).hexdigest()
                        old_file = os.path.join(embeddings_folder, f"{old_hash}.qpl")

                        # Update path and metadata
                        obj.path = obj_path
                        obj.datetime = datetime.fromtimestamp(os.stat(obj_path).st_mtime)

                        # Save the updated object with the new hash
                        new_hash = hashlib.md5(obj.path.encode()).hexdigest()
                        new_file = os.path.join(embeddings_folder, f"{new_hash}.qpl")
                        obj.save(new_file)

                        # Delete the old file if it exists and is different from the new one
                        if os.path.exists(old_file) and old_file != new_file:
                            os.remove(old_file)
                            print(f"Removed old file: {old_file}", flush=True)

                        changed = True
                    else:
                        print(f"Path not found for object {obj.path}; skipping.", flush=True)
                        continue"""

                # Apply date filters if specified
                if ((not date_from or obj.datetime >= date_from) and 
                    (not date_to or obj.datetime <= date_to)):
                    filtered_objects.append(obj)

            except Exception as e:
                print(f"Error processing object {e}", flush=True)
                #print(f"Error processing object {obj.path}: {e}", flush=True) #this caused errors I think
                continue

        # If metadata was updated, update the cache file
        """if changed:
            print("Metadata updated; saving new cache.", flush=True)
            try:
                cache_file = os.path.join(embeddings_folder, "cached_embeddings.pkl")
                with open(cache_file, 'wb') as f:
                    pickle.dump(updated_cache, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f"Error saving updated cache file: {e}", flush=True)"""
        
        print(f"Filtered {len(vid_objs)} videos down to {len(filtered_objects)}", flush=True)
        return filtered_objects

    def load_objs(self, video_folder, embeddings_path, chunk_size=1000, save_interval=50000):
        """
        Load objects using a cache file if available and process missing files incrementally,
        with iterative saving every `save_interval` files.
        """
        if (embeddings_path == self.current_embedding_folder and 
            self.cached_objects is not None):
            print('Using pre-cached objs', flush=True)
            return self.cached_objects

        cache_file = os.path.join(embeddings_path, "cached_embeddings.pkl")
        cached_objects, processed_files = [], set()

        # Load existing cache if available
        if os.path.exists(cache_file):
            try:
                print("Using Cache File..", flush=True)
                with open(cache_file, 'rb') as f:
                    cached_objects = pickle.load(f)
                processed_files = {f"{os.path.join(embeddings_path, hashlib.md5(obj.path.encode()).hexdigest())}.qpl" for obj in cached_objects}
                print(f"Cache file loaded with {len(cached_objects)} videos.", flush=True)
            except Exception as e:
                print(f"Error loading cache file: {e}. Fallback to processing files. (This may take some time)", flush=True)

        # Identify all .qpl files in the embedding path
        curr_files = {self.fix_path(f.path) for f in os.scandir(embeddings_path) if f.name.endswith(".qpl")}
        unprocessed_files = curr_files - processed_files

        # If no unprocessed files, return cached objects
        if not unprocessed_files:
            self.cached_objects = cached_objects
            self.current_video_folder = video_folder
            self.current_embedding_folder = embeddings_path
            return cached_objects
        
        print(f"Caching {len(unprocessed_files)} uncached files, {unprocessed_files}", flush=True)

        # Process unprocessed files in chunks
        new_objects = []
        total_processed = 0

        for i in range(0, len(unprocessed_files), chunk_size):
            chunk_start = time.time()
            chunk = list(unprocessed_files)[i:i + chunk_size]

            # Load chunk using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=8) as executor:
                chunk_objs = list(filter(None, executor.map(self.load_single_obj, chunk)))

            chunk_load_time = time.time() - chunk_start
            objects_per_second = len(chunk) / chunk_load_time if chunk_load_time > 0 else 0

            print(f"Chunk {i}-{i+len(chunk)}: {chunk_load_time:.2f}s ({objects_per_second:.1f} obj/s)", flush=True)

            new_objects.extend(chunk_objs)
            total_processed += len(chunk_objs)

            # Iteratively save every `save_interval` files
            if total_processed >= save_interval:
                try:
                    print(f"Saving intermediate cache after {total_processed} objects.", flush=True)
                    intermediate_objects = cached_objects + new_objects
                    with open(cache_file, 'wb') as f:
                        pickle.dump(intermediate_objects, f, pickle.HIGHEST_PROTOCOL)
                    total_processed = 0  # Reset counter after saving
                except Exception as e:
                    print(f"Error during intermediate saving: {e}", flush=True)

            # Clear memory
            gc.collect()

        # Final merge of new objects with cached objects
        all_objects = cached_objects + new_objects
    
        # Save final updated cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(all_objects, f, pickle.HIGHEST_PROTOCOL)
            print("Final cache file saved.", flush=True)
        except Exception as e:
            print(f"Error saving final cache file: {e}", flush=True)

        self.cached_objects = all_objects
        self.current_video_folder = video_folder
        self.current_embedding_folder = embeddings_path

        return all_objects

    def load_single_obj(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError, FileNotFoundError) as e:
            print(f"Error loading {file_path}: {e}. Removing malformed file.", flush=True)
            try:
                os.remove(file_path)
                #NEED TO ADD CODE THAT REMOVES THIS FROM THE CSV FILE AS WELL!!!!
                print(f"Deleted malformed file (if this is a mistake, please re-embed the video): {file_path}", flush=True)
            except Exception as delete_error:
                print(f"Failed to delete malformed file {file_path}: {delete_error}", flush=True)
            return None
    
    def sigmoid(self, x:list, r=5, c=0.5): 
        return [0 if i==0 else 1 / (1 + math.exp(-r*(i-c))) for i in x]

    def retrieve_vids(self, query, annotation_folder, vid_objs, isMean, query_type):
        has_anno = False
        has_query = True
        if query_type == "text":
            query = query.lower()
            annotations = []
            annotation_list = []
            if os.path.exists(annotation_folder):
                annotations = [i for i in os.listdir(annotation_folder) if i.endswith(".anno")]
                annotation_list = [i for i in annotations if i[:-5] in query]
            if len(annotation_list) >0:
                query = [i for i in query.split(annotation_list[0][:-5]) if i != '']
                has_anno=True
                print("adding annos", annotation_list, flush=True)
            
            # Load annotation vectors only if they exist
            annotations_vector = []
            for anno in annotation_list:
                with open(os.path.join(annotation_folder, anno), 'rb') as a:
                    anno_obj = pickle.load(a)
                annotations_vector.append(anno_obj.mean.to(self.device))

            if len(query) > 0:
                _, text_feat = self.model.get_txt_feat(query)
                query_feat = text_feat.to(self.device)
            else: has_query = False
        elif query_type == "video":
            output_file = os.path.join(self.current_embedding_folder, f"{hashlib.md5(self.fix_path(query).encode()).hexdigest()}.qpl")
            with open(output_file, 'rb') as f:    
                obj = pickle.load(f)
            query_feat = obj.chunks.to(self.device)

        start = time.time()
        
        # Stack all chunks for batch processing
        all_chunks = [vid_obj.chunks.to(self.device) for vid_obj in vid_objs]
        stacked_chunks = torch.cat(all_chunks, dim=0)
        #print("stack", time.time()-start)

        # Calculate similarities for all chunks at once
        with torch.no_grad():
            if has_query: similarities = (query_feat @ stacked_chunks.T) / (torch.norm(query_feat) * torch.norm(stacked_chunks, dim=1))
            if has_anno: anno_sims = [(i @ stacked_chunks.T) / (torch.norm(i) * torch.norm(stacked_chunks, dim=1)) for i in annotations_vector]
            #print("sims", time.time()-start)
            # Split similarities by each video's chunks
            split_sizes = [len(vid_obj.chunks) for vid_obj in vid_objs]
            #print("chunk lens", time.time()-start)
            if has_query: sims_split = torch.split(similarities, split_sizes, dim=1)
            if has_anno: anno_sims_split = torch.split(anno_sims[0], split_sizes, dim=1)
            #print("split", time.time()-start)
            # Vectorized calculation of mean or max similarity
            # Using `cat` and then splitting by split_sizes
            #start2 = time.time()
            
            if has_query: final_sims = torch.stack([torch.mean(sim) if isMean else torch.max(sim) for sim in sims_split])
            if has_anno: final_anno_sims = torch.stack([torch.mean(sim) if isMean else torch.max(sim) for sim in anno_sims_split])
            #print("v2: ", time.time()-start2)
            #print("sims meanmax", time.time()-start)
        #sims1, sims2 = self.sigmoid(final_sims, 10, 0.5), self.sigmoid(final_anno_sims, 10, 0.5)
        #print("sigmoid", time.time()-start)
        
        #try multiply?
        if has_anno and has_query: 
            sims_tensor = final_sims * final_anno_sims
            for i in range(len(vid_objs)): vid_objs[i].sims = sims_split[i] * anno_sims_split[i]
        elif not has_anno and has_query: 
            sims_tensor = final_sims
            for i in range(len(vid_objs)): vid_objs[i].sims = sims_split[i]
        elif has_anno and not has_query: 
            sims_tensor = final_anno_sims
            for i in range(len(vid_objs)): vid_objs[i].sims = anno_sims_split[i]
        
        print("Search (cos sim) in: ", time.time()-start, flush=True)
        return sims_tensor
   
    def search_videos(self, video_folder, embedding_folder, annotation_folder, query, isMean, query_type, date_from=None, date_to=None):
        try:
            start = time.time()
            vid_objs = self.load_objs(video_folder, embedding_folder)
            print("Objs loaded in: ", time.time()-start, flush=True)
            if date_from or date_to: vid_objs = self.filter_metadata(vid_objs, date_from, date_to, embedding_folder)
            sims_tensor = self.retrieve_vids(query, annotation_folder, vid_objs, isMean, query_type)
            top_values, top_indices = torch.topk(sims_tensor, k=min(len(vid_objs), 500), largest=True, sorted=True) #only get the top 500 videos (should be plenty)
            top_vids = [vid_objs[i] for i in top_indices]
            del vid_objs

            return top_vids
        
        except Exception as e:
            print(f"Error in search: {e}", flush=True)
            return []

    def process_commands(self):
        while True:
            try:
                line = input()
                if line.strip().lower() == 'exit':
                    break
                
                try:
                    command = json.loads(line)
                    if command.get('command') == 'create_annotation':
                        folder = command.get('annotation_folder')
                        key = command.get('key')
                        annotation_type = command.get('type')
                        value = command.get('value')
                        self.create_annotation(folder, key, annotation_type, value)
                    elif command.get('command') == 'update_embedding_folder':
                        new_folder = command.get('embedding_folder')
                        video_folder = command.get('video_folder')
                        self.update_embedding_folder(video_folder, new_folder)
                    elif command.get('command') == 'search_file':
                        # Pass inputs to the search_file method
                        file_path = command.get('file_path')
                        query_type = command.get('query_type')
                        video_folder = command.get('video_folder')
                        embedding_folder = command.get('embedding_folder')
                        annotation_folder = command.get('annotation_folder')

                        result = self.search_file(file_path, query_type, video_folder, embedding_folder, annotation_folder)
                        print(json.dumps(result), flush=True)
                    else:
                        date_from = command.get('date_from')
                        date_to = command.get('date_to')
                        vids = self.search_videos(
                            command['video_folder'],
                            command['embedding_folder'],
                            command['annotation_folder'],
                            command['query'],
                            command['is_mean'],
                            command['query_type'],
                            date_from,
                            date_to
                        )
                        #print("Done")
                        print(json.dumps([ (i.path, (torch.argmax(i.sims)*15).item())  for i in vids ]), flush=True)
                except json.JSONDecodeError:
                    print(json.dumps({"error": "Invalid JSON"}), flush=True)
                except Exception as e:
                    print(json.dumps({"error": str(e)}), flush=True)
                    
            except EOFError:
                break
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    search = PersistentSearch()
    search.process_commands()
    
    #create anno
    #{"command": "create_annotation", "annotation_folder": "C:\\Users\\Aidan Ladenburg\\Desktop\\anno","key": "jensen", "type": "img", "value": "C:\\Users\\Aidan Ladenburg\\Desktop\\crying.png"} 
        

    #small w dates
    #{"command": "search_videos", "video_folder": "", "embedding_folder": "C:\\Users\\Aidan Ladenburg\\Desktop\\test_e2","annotation_folder": "", "query": "example", "is_mean": "true", "query_type": "text", "date_from": "2024-11-01", "date_to": "2024-12-31"}
    
    #large w dates
    #{"command": "search_videos", "video_folder": "", "embedding_folder": "L:\\DoNotDelete - Embeddings2","annotation_folder": "", "query": "example", "is_mean": "true", "query_type": "text", "date_from": "2024-11-19", "date_to": "2024-12-31"}
    
    #large no dates
    #{"command": "search_videos", "video_folder": "", "embedding_folder": "L:\\DoNotDelete - Embeddings2","annotation_folder": "", "query": "example", "is_mean": "true", "query_type": "text"}
    