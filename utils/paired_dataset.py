import torch
from torch.utils.data import Dataset
import random
import os
import numpy as np
from pathlib import Path
import cv2


class PairedContrastiveDataset(Dataset):
    def __init__(self, index_data, T=32, transform=None, frames_root='./dataset_frames'):
        self.pairs = self._organize_pairs(index_data)
        self.T = T
        self.transform = transform
        self.frames_root = frames_root
        
        
    def _organize_pairs(self, index_data):
        groups = {}
        for item in index_data:
            gid = item['group_id']
            if gid not in groups:
                groups[gid] = {'real': None, 'fake': None}
            
            if item['label'] == 1:
                groups[gid]['real'] = item
            else:
                groups[gid]['fake'] = item
        
        pairs = []
        for gid, pair in groups.items():
            if pair['real'] is not None and pair['fake'] is not None:
                pairs.append({
                    'group_id': gid,
                    'real_path': pair['real']['video_path'],
                    'fake_path': pair['fake']['video_path'],
                    'generation_method': pair['fake']['generation_method']
                })
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def _get_frames_path(self, video_path):
        video_name = Path(video_path).stem

        if 'original_videos' in video_path:
            parts = Path(video_path).parts
            group_id = parts[-2]
            frames_path = os.path.join(self.frames_root, 'original_videos', group_id, video_name)
        else:
            parts = Path(video_path).parts
            method_folder = parts[-2]
            frames_path = os.path.join(self.frames_root, 'generated_videos', method_folder, video_name)
        
        return frames_path
    
    def _sample_frames(self, num_frames):
        if num_frames <= self.T:
            idx = list(range(num_frames)) + [num_frames - 1] * (self.T - num_frames)
            return idx
        
        return list(range(self.T))
    
    def _read_frames_from_folder(self, frames_folder):
        try:
            frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
            num_frames = len(frame_files)
            
            if num_frames == 0:
                raise ValueError(f"Frame folder is empty: {frames_folder}")
            
            frame_indices = self._sample_frames(num_frames)
            
            frames = []
            for idx in frame_indices:
                frame_path = os.path.join(frames_folder, frame_files[idx])
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            return frames
            
        except Exception as e:
            print(f"Failed to read frames from {frames_folder}, Error: {e}")
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.T)]
    
    def _process_video(self, video_path):
        frames_folder = self._get_frames_path(video_path)
        
        frames = self._read_frames_from_folder(frames_folder)
        
        processed_frames = []
        for frame in frames:
            if frame.shape[:2] != (224, 224):
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            if self.transform:
                frame_tensor = self.transform(frame_tensor)
            processed_frames.append(frame_tensor)
        
        return torch.stack(processed_frames)  # [T, C, H, W]
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        real_video = self._process_video(pair['real_path'])
        fake_video = self._process_video(pair['fake_path'])
        
        return real_video, fake_video, pair['group_id']


def collate_fn(items):
    batch_data = []
    
    for pair_idx, (real_v, fake_v, gid) in enumerate(items):
        real_v = real_v.permute(1, 0, 2, 3)
        fake_v = fake_v.permute(1, 0, 2, 3)
        
        batch_data.append({
            'video': real_v,
            'label': 1,
            'pair_index': pair_idx
        })
        
        batch_data.append({
            'video': fake_v,
            'label': 0,
            'pair_index': pair_idx
        })
    
    random.shuffle(batch_data)
    
    return batch_data
