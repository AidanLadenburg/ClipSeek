import torch
from torch import nn
import cv2
import numpy as np
import math

from easydict import EasyDict as edict

from backbones.bert.tokenization_bert import BertTokenizer
from backbones.bert.xbert import BertForMaskedLM, BertConfig
from backbones.internvideo2.intern_video2 import pretrain_internvideo2_1b_patch14_224
from backbones.internvideo2.pos_embed import interpolate_pos_embed_internvideo2_new

from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings("ignore")

def setup_internvideo2(path):
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    model = InternVideo2_Stage2(tokenizer=tokenizer, is_pretrain=True)
    
    model = model.to(torch.device("cuda"))
    model_without_ddp = model

    checkpoint = torch.load(path, map_location="cuda")
    state_dict = checkpoint["module"] # This is a deepspeed stage 1 model

    interpolate_pos_embed_internvideo2_new(state_dict, model_without_ddp.vision_encoder, orig_t_size=4)

    model_without_ddp.load_state_dict(state_dict, strict=False)
    model_without_ddp = model_without_ddp.to(torch.float32)
    model_without_ddp.eval()
    return (model_without_ddp, tokenizer,)

class InternVideo2_Stage2(nn.Module):

    def __init__(self, tokenizer, is_pretrain: bool=True):
        super(InternVideo2_Stage2, self).__init__()

        self.tokenizer = tokenizer

        self.is_pretrain = is_pretrain
        self.vision_width = 768
        self.text_width = 1024
        self.embed_dim = 512

        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.freeze_vision()

        self.text_encoder = self.build_text_encoder()
        self.freeze_text()

        self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)

    def freeze_vision(self):
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

    def freeze_text(self):
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def _frame_from_video(self, path, fps, total_frames, chunk_len=-1, chunk=-1, start_frame=None, end_frame=None):
        video = cv2.VideoCapture(path)
        if not (start_frame and end_frame):
            if chunk_len == -1: 
                print("encode whole video")
                start_frame = 0
                end_frame = int(total_frames)
            else: 
                start_frame = int(int(fps)*chunk_len*(chunk-1))
                end_frame = int(int(fps)*chunk_len*chunk)
                if end_frame > total_frames - 100: end_frame = int(total_frames)
        interval = max(1, (end_frame - start_frame) // 4)
        frames = []
        for i in range(4):
            target_frame = start_frame + i * interval
            if target_frame >= end_frame:
                break  # Stop if we've reached the end of the video

            # Set video position to the target frame
            video.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
            # Read and store the frame
            success, frame = video.read()
            if not success:
                print("Error Reading frames")
                break  # Stop if the frame couldn't be read
            
            frames.append(cv2.resize(frame, (224, 224)))  # Resize immediately to reduce memory load
        video.release()

        if len(frames) < 4:  # Ensure minimum frame length for the tensor
                frames = [frames[0]] * (4 - len(frames)) + frames

        return frames

    def frames2tensor(self, vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):        
        # stack frames into a single numpy array for batch processing
        vid_array = np.array([frame[:, :, ::-1] for frame in vid_list])

        # Normalize all frames at once (batch normalization)
        v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        vid_array = (vid_array / 255.0 - v_mean) / v_std

        # Convert to tensor and adjust dimensions
        vid_tensor = torch.from_numpy(vid_array).permute(0, 3, 1, 2).unsqueeze(0)
        vid_tensor = vid_tensor.to(device, non_blocking=True).float()

        return vid_tensor

    @property
    def dtype(self):
        return self.vision_encoder.patch_embed.proj.weight.dtype

    def encode_vision(self, image: torch.Tensor, test: bool=False):
        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4).to(self.dtype) # [B,T,C,H,W] -> [B,C,T,H,W]
        if test:
            vision_embeds, pooled_vision_embeds, _, _ = self.vision_encoder(image, None, use_image)
            return vision_embeds, pooled_vision_embeds
        else:
            mask, targets_clip_middle_vis, targets_clip_final_vis = self.encode_teacher(image) 
            vision_embeds, pooled_vision_embeds, student_output, student_output_final = self.vision_encoder(
                    image, mask, use_image)
            return vision_embeds, pooled_vision_embeds, student_output, student_output_final, targets_clip_middle_vis, targets_clip_final_vis

    def encode_text(self, text: dict):
        text_output = self.get_text_encoder()(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        pooled_text_embeds = text_embeds[:, 0]
        return text_output, pooled_text_embeds

    def build_vision_encoder(self):
        model_config = {'model_cls': 'InternVideo2_Stage2', 'vision_encoder': {'name': 'pretrain_internvideo2_1b_patch14_224', 'img_size': 224, 'num_frames': 4, 'tubelet_size': 1, 'patch_size': 14, 'd_model': 1408, 'clip_embed_dim': 768, 'clip_teacher_embed_dim': 3200, 'clip_teacher_final_dim': 768, 'clip_norm_type': 'l2', 'clip_return_layer': 6, 'clip_student_return_interval': 1, 'pretrained': './InternVideo2-stage2_1b-224p-f4.pt', 'use_checkpoint': True, 'checkpoint_num': 40, 'use_flash_attn': False, 'use_fused_rmsnorm': False, 'use_fused_mlp': False, 'clip_teacher': None, 'clip_input_resolution': 224, 'clip_teacher_return_interval': 1, 'video_mask_type': 'random', 'video_mask_ratio': 0.8, 'image_mask_type': 'random', 'image_mask_ratio': 0.5, 'sep_image_video_pos_embed': True, 'keep_temporal': False, 'only_mask': True}, 'text_encoder': {'name': 'bert_large', 'pretrained': 'bert-large-uncased', 'config': 'configs/config_bert_large.json', 'd_model': 1024, 'fusion_layer': 19}, 'multimodal': {'enable': True}, 'embed_dim': 512, 'temp': 0.07, 'find_unused_parameters': False}
        model_config = edict(model_config)
        vision_encoder = pretrain_internvideo2_1b_patch14_224(model_config)

        # parameters for mask
        img_size = 224
        num_frames = 4
        tublet_size = 1
        patch_size = 14
        self.clip_img_size = 224
        self.video_mask_type = "random"
        self.video_window_size = (num_frames // tublet_size, img_size // patch_size, img_size // patch_size)
        self.video_mask_ratio = 0.8
        self.image_mask_type = "random"
        self.image_window_size = (1, img_size // patch_size, img_size // patch_size)
        self.image_mask_ratio = 0.5
        
        return vision_encoder

    def build_text_encoder(self):
        bert_config = BertConfig.from_json_file("./config_bert_large.json")
        bert_config.encoder_width = 1408
        bert_config.gradient_checkpointing = True
        bert_config.fusion_layer = 19
        text_encoder, _ = BertForMaskedLM.from_pretrained("bert-large-uncased", config=bert_config, output_loading_info=True, local_files_only=True)
        return text_encoder

    def get_text_encoder(self):
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder
    
    def process_chunk(self, path, fps, total_frames, chunk_size, chunk, device, stream):
        # Load frames for the current chunk
        frames = self._frame_from_video(path, fps, total_frames, chunk_size, chunk)
        if len(frames) < 4:  # Ensure minimum frame length for the tensor
            frames = [frames[0]] * (4 - len(frames)) + frames

        # Use the provided stream for asynchronous execution
        with torch.cuda.stream(stream):
            frames_tensor = self.frames2tensor(frames, fnum=4, target_size=(224, 224), device=device)

            with torch.no_grad():
                _, vfeat = self.encode_vision(frames_tensor, test=True)
                vfeat = self.vision_proj(vfeat)
                vfeat /= vfeat.norm(dim=-1, keepdim=True)

        # Clean up memory
        #del frames_tensor, frames
        #torch.cuda.empty_cache()  # Free up unused memory on the GPU

        return vfeat

    def _frame_from_video_async(self, path, fps, total_frames, chunk_len, chunk, executor):
        """
        Asynchronously loads frames for a video chunk.
        """
        start_frame = int(fps * chunk_len * (chunk - 1))
        end_frame = int(fps * chunk_len * chunk)
        end_frame = min(end_frame, total_frames)  # Ensure end doesn't exceed total frames

        # Submit the frame loading task to the executor
        future = executor.submit(self._frame_from_video, path, fps, total_frames, chunk_len, chunk, start_frame, end_frame)
        return future  # Return the future to track completion

    def get_vid_feat(self, path, chunk_size=15, num_streams=1, device=torch.device('cuda'), progress_callback=None):
        video = cv2.VideoCapture(path)
        length = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        total_chunks = math.ceil(length / chunk_size)
        video.release()
        chunks = []
        streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]  # CUDA streams
        
        # Use ThreadPoolExecutor for asynchronous frame loading
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(total_chunks):
                # Asynchronously load frames for each chunk
                future = self._frame_from_video_async(path, fps, total_frames, chunk_size, i+1, executor)
                futures.append((future, i))

            for future, i in futures:
                frames = future.result()  # Wait for frames to be loaded
                stream = streams[i % num_streams]  # Choose a stream for this chunk
                
                # Process chunk on GPU
                with torch.cuda.stream(stream):
                    frames_tensor = self.frames2tensor(frames, fnum=4, target_size=(224, 224), device=device)
                    with torch.no_grad():
                        _, vfeat = self.encode_vision(frames_tensor, test=True)
                        vfeat = self.vision_proj(vfeat)
                        vfeat /= vfeat.norm(dim=-1, keepdim=True)

                    chunks.append(vfeat)
                    if progress_callback:
                        progress_callback((i + 1) / total_chunks * 100)

        # Synchronize streams after processing
        for stream in streams:
            stream.synchronize()

        torch.cuda.empty_cache()  # Clear unused GPU memory
        return torch.cat(chunks)
            
    def get_vid_feat_simple(self):
        frames = self._frame_from_video(path)
        frames_tensor = self.frames2tensor(frames, fnum=4, target_size=(224, 224), device=torch.device('cuda'))
        with torch.no_grad():
            _, vfeat = self.encode_vision(frames_tensor, test=True)
            vfeat = self.vision_proj(vfeat)
            vfeat /= vfeat.norm(dim=-1, keepdim=True)
        return vfeat
    

    def get_img_feat(self, img, device=torch.device('cuda')):
        img = cv2.imread(img)
        img = cv2.resize(img, (224, 224))

        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        img = (img - mean) / std

        img = np.expand_dims(img, axis=(0, 1))
        img = np.transpose(img, (0, 1, 4, 2, 3))
        img = torch.from_numpy(img).to(device, non_blocking=True).float()

        with torch.no_grad():  
            a, vfeat = self.encode_vision(img, test=True)
            img_feat = self.vision_proj(vfeat)
            img_feat /= vfeat.norm(dim=-1, keepdim=True)

        return img_feat

    def get_txt_feat(self, text: str):
        with torch.no_grad():
            text = self.tokenizer(
                text, 
                padding="max_length", 
                truncation=True, 
                max_length=40, 
                return_tensors="pt",).to("cuda")
            t_nop, tfeat = self.encode_text(text)
            tfeat = self.text_proj(tfeat)
            tfeat /= tfeat.norm(dim=-1, keepdim=True)
        return t_nop, tfeat


if __name__ == '__main__':
    import time
    start = time.time()
    #odel_pth = './InternVideo2-stage2_1b-224p-f4.pt'
    #intern_model, tokenizer = setup_internvideo2(model_pth)
    #model = intern_model.to(torch.device('cuda'))

    #import pickle
    #pickle.dump(model, open(f"./model2.pkl", 'wb'))
    #model = pickle.load(open(f"./model.pkl", 'rb'))
    
    #model = torch.load("model.pt")
    #model = model.to('cpu')
    #torch.save(model, "model_cpu.pt")
    """model = torch.load("model.pt")

    print("model loaded: ", time.time()-start)

    #EXAMPLE TEXT ENCODING
    _ , text_encoded = model.get_txt_feat("woman sitting on couch looking at tablet")
    _ , text_encoded2 = model.get_txt_feat("a cow playing tennis")

    print("text encoded: ", time.time()-start)

    path = "E:/adhoc_search/nvidia2/04_Die-to_Datacenter_GB200_v30_Textless.mov.mp4"
    video_encoded = model.get_vid_feat(path)
    print(video_encoded)
    print("vid encoded: ", time.time()-start)

    #print("SIM: ", video_encoded@text_encoded.T)
    #print("SIM2: ", video_encoded@text_encoded2.T)
    print(f"exec time: {time.time()-start}")"""