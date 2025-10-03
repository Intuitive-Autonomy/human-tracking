"""
Video inference script with performance benchmarking
Directly processes video files without pre-extracting frames

Usage:
    python inference_video.py --video <path_to_video.mp4> --mask <path_to_first_frame_mask.png> --output <output_dir>

Features:
    - Direct video processing with OpenCV
    - Detailed performance metrics (FPS, timing breakdown)
    - Optional visualization output
    - GPU memory monitoring
"""

import os
import time
import json
from argparse import ArgumentParser
from collections import defaultdict

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from model.eval_network import STCN
from util.tensor_util import unpad, pad_divide_by
from inference_core_yv import InferenceCore
from dataset.range_transform import im_normalization
from dataset.util import all_to_onehot


class PerformanceMonitor:
    """Monitor and record performance metrics"""
    def __init__(self):
        self.timings = defaultdict(list)
        self.gpu_events = []

    def start_timer(self, name):
        return time.perf_counter()

    def end_timer(self, name, start_time):
        elapsed = time.perf_counter() - start_time
        self.timings[name].append(elapsed)
        return elapsed

    def add_gpu_event(self, name, start_event, end_event):
        self.gpu_events.append((name, start_event, end_event))

    def get_summary(self):
        summary = {}
        for name, times in self.timings.items():
            summary[name] = {
                'total': sum(times),
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'count': len(times)
            }
        return summary

    def print_report(self, video_name, num_frames, total_time):
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        print(f"Video: {video_name}")
        print(f"Total frames: {num_frames}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {num_frames/total_time:.2f}")
        print(f"Time per frame: {total_time/num_frames*1000:.1f}ms")
        print("-"*60)

        summary = self.get_summary()
        if summary:
            print("Timing breakdown (per frame):")
            for name, stats in summary.items():
                print(f"  {name:20s}: {stats['mean']*1000:6.1f}ms (±{stats['std']*1000:4.1f}ms)")

        # GPU memory
        if torch.cuda.is_available():
            print("-"*60)
            print(f"GPU Memory Peak: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
            print(f"GPU Memory Current: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        print("="*60 + "\n")


def load_video_and_mask(video_path, mask_path, resolution=480):
    """Load video frames and first frame mask"""

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {orig_width}x{orig_height}, {fps:.2f}fps, {total_frames} frames")

    # Setup transforms
    if resolution != -1:
        im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
            transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC, antialias=True),
        ])
        mask_transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=InterpolationMode.NEAREST),
        ])
    else:
        im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])
        mask_transform = None

    # Read all frames
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Apply transform
        frame_tensor = im_transform(frame_pil)
        frames.append(frame_tensor)

        frame_idx += 1

    cap.release()

    # Stack frames
    frames = torch.stack(frames, 0)  # (T, C, H, W)
    frames = frames.unsqueeze(0)  # (1, T, C, H, W)

    print(f"Loaded {frames.shape[1]} frames, shape: {frames.shape}")

    # Load mask
    mask_pil = Image.open(mask_path).convert('P')
    palette = mask_pil.getpalette()
    mask_np = np.array(mask_pil, dtype=np.uint8)

    # Get unique labels
    labels = np.unique(mask_np)
    labels = labels[labels != 0]  # Remove background
    num_objects = len(labels)

    print(f"Mask info: {mask_np.shape}, objects: {num_objects}, labels: {labels}")

    # Convert to one-hot
    mask_onehot = all_to_onehot(mask_np, labels)
    mask_tensor = torch.from_numpy(mask_onehot).float()

    # Apply mask transform
    if mask_transform is not None:
        mask_tensor = mask_transform(mask_tensor)

    # mask_tensor is now (num_objects, H, W), no need to add extra dimension
    # It will be handled correctly in the interact function

    info = {
        'labels': labels,
        'num_objects': num_objects,
        'size': (orig_height, orig_width),
        'palette': palette,
        'fps': fps,
        'total_frames': total_frames
    }

    return frames, mask_tensor, info


def run_inference(video_path, mask_path, model_path, output_path,
                 resolution=480, top_k=20, mem_every=5, amp=True,
                 save_visualization=False):
    """Run inference on video with performance monitoring"""

    perf_monitor = PerformanceMonitor()

    # Disable gradient computation
    torch.autograd.set_grad_enabled(False)

    # Reset GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Load model
    print("\nLoading model...")
    t_start = perf_monitor.start_timer('model_loading')

    prop_model = STCN().cuda().eval()
    prop_saved = torch.load(model_path, weights_only=False)

    # Performs input mapping for stage 0 models
    for k in list(prop_saved.keys()):
        if k == 'value_encoder.conv1.weight':
            if prop_saved[k].shape[1] == 4:
                pads = torch.zeros((64,1,7,7), device=prop_saved[k].device)
                prop_saved[k] = torch.cat([prop_saved[k], pads], 1)

    prop_model.load_state_dict(prop_saved)
    perf_monitor.end_timer('model_loading', t_start)
    print(f"Model loaded in {perf_monitor.timings['model_loading'][0]:.2f}s")

    # Load video and mask
    print("\nLoading video and mask...")
    t_start = perf_monitor.start_timer('data_loading')
    frames, mask_tensor, info = load_video_and_mask(video_path, mask_path, resolution)
    perf_monitor.end_timer('data_loading', t_start)
    print(f"Data loaded in {perf_monitor.timings['data_loading'][0]:.2f}s")

    # Prepare output directory
    os.makedirs(output_path, exist_ok=True)

    # Start inference
    print("\nStarting inference...")
    total_start = time.perf_counter()

    with torch.amp.autocast('cuda', enabled=amp):
        num_objects = info['num_objects']
        labels = info['labels']
        size = info['size']
        palette = info['palette']

        # Initialize processor
        t_start = perf_monitor.start_timer('processor_init')
        processor = InferenceCore(
            prop_model,
            frames,
            num_objects=num_objects,
            top_k=top_k,
            mem_every=mem_every,
            include_last=False
        )
        perf_monitor.end_timer('processor_init', t_start)

        # Process first frame with mask
        t_start = perf_monitor.start_timer('first_frame_encode')

        # Prepare mask - mask_tensor should be (num_objects, H, W)
        # Add spatial singleton dimension to match expected format: (num_objects, 1, H, W)
        mask_4d = mask_tensor.unsqueeze(1).cuda()

        # Add background channel - result: (num_objects+1, 1, H, W)
        # This matches the format from eval_generic.py where msk[:,frame_idx] has shape (num_objects, 1, H, W)
        with_bg_msk = torch.cat([
            1 - torch.sum(mask_4d, dim=0, keepdim=True),  # Background: (1, 1, H, W)
            mask_4d,  # Objects: (num_objects, 1, H, W)
        ], 0)  # Result: (num_objects+1, 1, H, W)

        # Object indices (1-indexed for the processor)
        obj_idx = list(range(1, num_objects + 1))

        # Interact with first frame (index 0) and propagate to end
        # The interact function will handle padding internally
        processor.interact(with_bg_msk, 0, frames.shape[1], obj_idx)

        perf_monitor.end_timer('first_frame_encode', t_start)

        # Extract results
        print("Extracting results...")
        t_start = perf_monitor.start_timer('postprocessing')

        out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')

        for ti in range(processor.t):
            t_frame_start = perf_monitor.start_timer(f'postprocess_frame')

            prob = unpad(processor.prob[:,ti], processor.pad)
            prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
            out_masks[ti] = torch.argmax(prob, dim=0)

            perf_monitor.end_timer(f'postprocess_frame', t_frame_start)

        out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

        perf_monitor.end_timer('postprocessing', t_start)

    total_time = time.perf_counter() - total_start

    # Remap indices to original labels
    t_start = perf_monitor.start_timer('save_results')
    idx_masks = np.zeros_like(out_masks)
    for i in range(1, num_objects + 1):
        idx_masks[out_masks == i] = labels[i-1]

    # Save masks
    print(f"Saving results to {output_path}...")
    for f in range(idx_masks.shape[0]):
        img_E = Image.fromarray(idx_masks[f])
        img_E.putpalette(palette)
        img_E.save(os.path.join(output_path, f'{f:05d}.png'))

    perf_monitor.end_timer('save_results', t_start)

    # Save visualization video if requested
    if save_visualization:
        print("Creating visualization video...")
        create_visualization_video(
            video_path, idx_masks, palette,
            os.path.join(output_path, 'visualization.mp4'),
            info['fps']
        )

    # Print performance report
    video_name = os.path.basename(video_path)
    perf_monitor.print_report(video_name, processor.t, total_time)

    # Save performance report as JSON
    report = {
        'video': video_name,
        'num_frames': int(processor.t),
        'total_time': float(total_time),
        'fps': float(processor.t / total_time),
        'time_per_frame': float(total_time / processor.t),
        'timings': {k: {kk: float(vv) for kk, vv in v.items()}
                   for k, v in perf_monitor.get_summary().items()},
        'gpu_memory_peak_gb': float(torch.cuda.max_memory_allocated()/1024**3) if torch.cuda.is_available() else 0,
    }

    with open(os.path.join(output_path, 'performance_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Performance report saved to {os.path.join(output_path, 'performance_report.json')}")

    return out_masks, report


def create_visualization_video(video_path, masks, palette, output_path, fps):
    """Create a visualization video with overlay"""

    # Convert palette to RGB colormap
    palette_np = np.array(palette).reshape(-1, 3)

    # Open original video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for idx in range(len(masks)):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize mask to original size
        mask = cv2.resize(masks[idx], (width, height), interpolation=cv2.INTER_NEAREST)

        # Create colored mask
        colored_mask = palette_np[mask]

        # Blend with original frame
        alpha = 0.5
        blended = cv2.addWeighted(frame, 1-alpha, colored_mask.astype(np.uint8), alpha, 0)

        out.write(blended)

    cap.release()
    out.release()

    print(f"Visualization saved to {output_path}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--mask', required=True, help='Path to first frame mask (PNG)')
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument('--model', default='saves/stcn.pth', help='Path to model checkpoint')
    parser.add_argument('--resolution', type=int, default=480, help='Processing resolution (-1 for original)')
    parser.add_argument('--top', type=int, default=20, help='Top-k memory frames')
    parser.add_argument('--mem_every', type=int, default=5, help='Add memory frame every N frames')
    parser.add_argument('--amp_off', action='store_true', help='Disable automatic mixed precision')
    parser.add_argument('--save_viz', action='store_true', help='Save visualization video')

    args = parser.parse_args()

    run_inference(
        video_path=args.video,
        mask_path=args.mask,
        model_path=args.model,
        output_path=args.output,
        resolution=args.resolution,
        top_k=args.top,
        mem_every=args.mem_every,
        amp=not args.amp_off,
        save_visualization=args.save_viz
    )
