"""
Streaming STCN Inference Core
Supports real-time incremental inference without pre-loading all frames
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from inference_memory_bank import MemoryBank
from model.eval_network import STCN
from model.aggregate import aggregate
from util.tensor_util import pad_divide_by, unpad
from dataset.range_transform import im_normalization
from dataset.util import all_to_onehot


class StreamingSTCN:
    """
    Streaming STCN inference processor for real-time video segmentation

    Key features:
    - No need to pre-load all frames
    - Step-by-step processing with step() method
    - Automatic memory bank management
    - Memory-efficient (only keeps recent frames in memory)
    """

    def __init__(self, prop_net, first_frame, first_mask, resolution=480,
                 top_k=20, mem_every=5, max_mem_frames=20):
        """
        Initialize streaming STCN processor

        Args:
            prop_net: STCN model
            first_frame: First frame image (BGR, numpy array)
            first_mask: First frame mask (binary uint8, 255 for object)
            resolution: Processing resolution
            top_k: Top-k for memory bank matching
            mem_every: Add to memory every N frames
            max_mem_frames: Maximum memory frames to keep (for memory efficiency)
        """
        self.prop_net = prop_net
        self.resolution = resolution
        self.top_k = top_k
        self.mem_every = mem_every
        self.max_mem_frames = max_mem_frames

        self.device = 'cuda'
        self.current_frame_idx = 0
        self.last_mem_frame_idx = 0

        # Setup image transform
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
            transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC, antialias=True),
        ])

        # Store original size for output
        self.orig_size = first_frame.shape[:2]

        # Process first frame to get dimensions
        first_frame_tensor = self._preprocess_frame(first_frame)
        _, h, w = first_frame_tensor.shape

        # Pad to multiple of 16
        first_frame_padded, self.pad = pad_divide_by(first_frame_tensor.unsqueeze(0).unsqueeze(0), 16)
        _, _, _, self.nh, self.nw = first_frame_padded.shape

        # Initialize mask
        labels = np.unique(first_mask)
        labels = labels[labels != 0]

        if len(labels) == 0:
            raise ValueError("First frame mask is empty!")

        self.num_objects = len(labels)
        self.labels = labels

        # Convert mask to tensor
        mask_transform = transforms.Resize(resolution, interpolation=InterpolationMode.NEAREST)
        mask_onehot = all_to_onehot(first_mask, labels)
        mask_tensor = torch.from_numpy(mask_onehot).float()
        mask_tensor = mask_transform(mask_tensor).unsqueeze(1).cuda()  # (k, 1, H, W)

        # Pad mask
        mask_padded, _ = pad_divide_by(mask_tensor, 16)

        # Add background
        with_bg_msk = torch.cat([
            1 - torch.sum(mask_padded, dim=0, keepdim=True),
            mask_padded,
        ], 0)  # (k+1, 1, H, W)

        # Initialize memory banks
        self.enabled_obj = list(range(1, self.num_objects + 1))
        self.mem_banks = {}

        for oi in self.enabled_obj:
            self.mem_banks[oi] = MemoryBank(k=1, top_k=self.top_k)

        # Current probability (for this frame only)
        self.prob = torch.zeros((self.num_objects + 1, 1, self.nh, self.nw),
                                dtype=torch.float32, device=self.device)

        # Set first frame mask
        # mask_regions shape: (1, H, W), prob shape: (k+1, 1, H, W)
        mask_regions = (with_bg_msk[1:].sum(0) > 0.5)  # (1, H, W)
        # Expand mask_regions to match prob dimensions
        self.prob[:, mask_regions] = 0
        for i, oi in enumerate(self.enabled_obj):
            self.prob[oi] = with_bg_msk[oi]

        self.prob = aggregate(self.prob[1:], keep_bg=True)

        # Encode first frame and add to memory
        key_k, _, qf16, _, _ = self._encode_key(first_frame_padded.squeeze(0))
        key_v = self.prop_net.encode_value(
            first_frame_padded.squeeze(0).cuda(),
            qf16,
            self.prob[self.enabled_obj].cuda()
        )
        key_k = key_k.unsqueeze(2)

        for i, oi in enumerate(self.enabled_obj):
            self.mem_banks[oi].add_memory(key_k, key_v[i:i+1])

        self.current_frame_idx = 1  # Next frame to process
        self.last_mem_frame_idx = 0

        print(f"StreamingSTCN initialized: {self.num_objects} object(s), resolution={resolution}")

    def _preprocess_frame(self, frame):
        """Convert BGR numpy frame to tensor"""
        frame_rgb = frame[..., ::-1].copy()  # BGR to RGB
        frame_pil = Image.fromarray(frame_rgb)
        frame_tensor = self.im_transform(frame_pil)
        return frame_tensor

    def _encode_key(self, frame_tensor):
        """Encode frame for key extraction"""
        return self.prop_net.encode_key(frame_tensor.cuda())

    def step(self, frame):
        """
        Process one frame and return segmentation mask

        Args:
            frame: Frame image (BGR, numpy array)

        Returns:
            mask: Binary mask (H, W) uint8, resized to original size
        """
        # Preprocess frame
        frame_tensor = self._preprocess_frame(frame)
        frame_padded, _ = pad_divide_by(frame_tensor.unsqueeze(0).unsqueeze(0), 16)
        frame_padded = frame_padded.squeeze(0)  # (1, C, H, W)

        # Encode current frame
        with torch.amp.autocast('cuda', enabled=True):
            k16, qv16, qf16, qf8, qf4 = self._encode_key(frame_padded)

            # Segment using memory banks
            out_mask = torch.cat([
                self.prop_net.segment_with_query(self.mem_banks[oi], qf8, qf4, k16, qv16)
                for oi in self.enabled_obj
            ], 0)

            out_mask = aggregate(out_mask, keep_bg=True)

            # Update current prob
            # Check if out_mask already has spatial singleton dimension
            if out_mask.dim() == 3:  # (k+1, H, W)
                self.prob = out_mask.unsqueeze(1)  # (k+1, 1, H, W)
                out_mask_for_value = out_mask[1:]  # (k, H, W)
            else:  # Already 4D (k+1, 1, H, W)
                self.prob = out_mask
                out_mask_for_value = out_mask[1:].squeeze(1)  # (k, H, W)

            # Update memory bank if needed
            is_mem_frame = (self.current_frame_idx - self.last_mem_frame_idx) >= self.mem_every

            if is_mem_frame:
                # encode_value expects mask without background, shape (k, H, W) or (k, 1, H, W)
                prev_value = self.prop_net.encode_value(
                    frame_padded.cuda(),
                    qf16,
                    out_mask_for_value.unsqueeze(1) if out_mask_for_value.dim() == 3 else out_mask_for_value
                )
                prev_key = k16.unsqueeze(2)

                for i, oi in enumerate(self.enabled_obj):
                    self.mem_banks[oi].add_memory(prev_key, prev_value[i:i+1], is_temp=False)

                    # Limit memory size for efficiency
                    if self.mem_banks[oi].mem_k is not None:
                        mem_size = self.mem_banks[oi].mem_k.shape[2]
                        if mem_size > self.max_mem_frames:
                            # Keep only recent frames
                            start_idx = mem_size - self.max_mem_frames
                            self.mem_banks[oi].mem_k = self.mem_banks[oi].mem_k[:, :, start_idx:]
                            self.mem_banks[oi].mem_v = self.mem_banks[oi].mem_v[:, :, start_idx:]

                self.last_mem_frame_idx = self.current_frame_idx

        self.current_frame_idx += 1

        # Extract and resize mask to original size
        # self.prob shape: (k+1, 1, H, W)
        # Add batch dimension and squeeze the spatial singleton
        prob_5d = self.prob.unsqueeze(0)  # (1, k+1, 1, H, W)
        prob_4d = prob_5d.squeeze(2)  # (1, k+1, H, W)

        # Unpad (expects 4D: B, C, H, W)
        prob_unpadded = unpad(prob_4d, self.pad)  # (1, k+1, H_unpad, W_unpad)

        # Resize to original size
        prob_resized = F.interpolate(prob_unpadded, size=self.orig_size, mode='bilinear', align_corners=False)

        # Get mask index
        mask_idx = torch.argmax(prob_resized, dim=1).squeeze(0)  # (H, W)

        # Convert to binary mask
        mask = (mask_idx > 0).cpu().numpy().astype(np.uint8) * 255

        return mask

    def get_stats(self):
        """Get current statistics"""
        mem_sizes = [self.mem_banks[oi].mem_k.shape[2] if self.mem_banks[oi].mem_k is not None else 0
                     for oi in self.enabled_obj]

        return {
            'frame_idx': self.current_frame_idx,
            'num_objects': self.num_objects,
            'memory_frames': max(mem_sizes) if mem_sizes else 0,
            'last_mem_update': self.last_mem_frame_idx,
        }
