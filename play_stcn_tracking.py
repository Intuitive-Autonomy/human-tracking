#!/usr/bin/env python3
"""
Real-time STCN Tracking with automatic first frame mask extraction (keep_middle)
Combines YOLO mask extraction + STCN tracking + real-time visualization

Changes:
- Use keep_middle: pick the person mask whose centroid is closest to the vertical midline (x = W/2)
- If frame 0 has no mask, automatically scan subsequent frames until a mask is found, then start tracking from that frame
- STCN is initialized with the annotation at mask_frame_idx to avoid timeline mismatch
"""

import csv
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from ultralytics import YOLO
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from model.eval_network import STCN
from util.tensor_util import unpad, pad_divide_by
from inference_core_yv import InferenceCore
from dataset.range_transform import im_normalization
from dataset.util import all_to_onehot

# ----------------- Config -----------------
CONF = 0.3  # YOLO confidence threshold

# Load YOLO segmentation model
print("Loading YOLO segmentation model...")
yolo_model = YOLO("yolov8n-seg.pt")

# Load STCN model
print("Loading STCN model...")
torch.autograd.set_grad_enabled(False)
stcn_model = STCN().cuda().eval()
model_path = "saves/stcn.pth"
prop_saved = torch.load(model_path, weights_only=False)

# Handle stage 0 model compatibility
for k in list(prop_saved.keys()):
    if k == 'value_encoder.conv1.weight':
        if prop_saved[k].shape[1] == 4:
            pads = torch.zeros((64, 1, 7, 7), device=prop_saved[k].device)
            prop_saved[k] = torch.cat([prop_saved[k], pads], 1)

stcn_model.load_state_dict(prop_saved)
print("Models loaded successfully!")


# ----------------- Perspective Correction -----------------
def correct_camera_perspective(image, camera_id, angle_degrees=25):
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    offset = int(w * angle_degrees / 90)

    if camera_id == 0:  # Top camera, narrow bottom
        dst = np.float32([[0, 0], [w, 0], [w - offset, h], [offset, h]])
    else:  # Bottom camera, narrow top
        dst = np.float32([[offset, 0], [w - offset, 0], [w, h], [0, h]])

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h))


def stitch_images(img_top, img_bottom, angle=25):
    img_top = correct_camera_perspective(img_top, 0, angle)
    img_bottom = correct_camera_perspective(img_bottom, 1, angle)

    # Align width
    w = min(img_top.shape[1], img_bottom.shape[1])
    img_top = cv2.resize(img_top, (w, int(img_top.shape[0] * w / img_top.shape[1])))
    img_bottom = cv2.resize(img_bottom, (w, int(img_bottom.shape[0] * w / img_bottom.shape[1])))

    stitched = np.vstack([img_top, img_bottom])
    return stitched


# ----------------- keep_middle Mask Extraction -----------------
def extract_mask_from_frame_keep_middle(frame_bgr):
    """
    Extract person segmentation from a single frame using YOLO, and keep the mask
    whose centroid is closest to the vertical midline (x = W/2).

    Args:
        frame_bgr: BGR image (H, W, 3)

    Returns:
        final_mask: uint8 binary mask (0/255) of the most "middle" person, or None if no person
        num_detections: total number of person masks detected in the frame
    """
    h, w = frame_bgr.shape[:2]
    cx_mid = w * 0.5

    # Run YOLO segmentation (expects RGB)
    results = yolo_model.predict(
        frame_bgr[..., ::-1], classes=[0], conf=CONF, verbose=False, device="cpu"
    )

    candidates = []  # (abs_dx, mask_uint8)
    total = 0

    for r in results:
        if r.masks is None:
            continue

        for m in r.masks.data:
            total += 1
            arr = m.detach().cpu().numpy()
            if arr.shape[-2:] != (h, w):
                arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)

            mask = (arr > 0.5).astype(np.uint8) * 255

            # Compute centroid (cx)
            M = cv2.moments(mask)
            if M["m00"] <= 1e-6:
                continue
            cx = M["m10"] / M["m00"]
            abs_dx = abs(cx - cx_mid)

            candidates.append((abs_dx, mask))

    if len(candidates) == 0:
        return None, 0

    # Keep the mask closest to the vertical midline
    candidates.sort(key=lambda x: x[0])
    final_mask = candidates[0][1].astype(np.uint8)
    return final_mask, total


def extract_first_frame_mask_keep_middle(session_frames, base_path):
    """
    Scan from frame 0 forward until a person mask is detected (keep_middle).
    Returns:
        mask (uint8), frame_idx (int), stitched_bgr (H, W, 3)
    """
    for frame_idx in range(len(session_frames)):
        row = session_frames[frame_idx]
        img0 = cv2.imread(str(base_path / row['camera_0_rgb_filename']))
        img1 = cv2.imread(str(base_path / row['camera_1_rgb_filename']))
        if img0 is None or img1 is None:
            continue

        stitched_bgr = stitch_images(img0, img1)
        mask, num_det = extract_mask_from_frame_keep_middle(stitched_bgr)
        if mask is not None:
            print(f"✓ keep_middle: Found mask at frame {frame_idx} "
                  f"({num_det} detection{'s' if num_det!=1 else ''})")
            return mask, frame_idx, stitched_bgr

    print("✗ No person mask found in the entire session!")
    return None, -1, None


# ----------------- STCN Tracking Setup -----------------
def prepare_stcn_frames(session_frames, base_path, resolution=480):
    """
    Load and prepare all frames for STCN tracking

    Returns:
        frames: Tensor (1, T, C, H, W)
        orig_size: Original (H, W)
    """
    im_transform = transforms.Compose([
        transforms.ToTensor(),
        im_normalization,
        transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC, antialias=True),
    ])

    frame_tensors = []
    orig_size = None

    for row in session_frames:
        # Load and stitch
        img0 = cv2.imread(str(base_path / row['camera_0_rgb_filename']))
        img1 = cv2.imread(str(base_path / row['camera_1_rgb_filename']))
        stitched = stitch_images(img0, img1)
        if stitched is None:
            continue

        if orig_size is None:
            orig_size = stitched.shape[:2]  # (H, W)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)

        # Transform
        from PIL import Image
        frame_pil = Image.fromarray(frame_rgb)
        frame_tensor = im_transform(frame_pil)
        frame_tensors.append(frame_tensor)

    frames = torch.stack(frame_tensors, 0).unsqueeze(0)  # (1, T, C, H, W)
    return frames, orig_size


def setup_stcn_tracking(frames, first_mask, mask_frame_idx, resolution=480):
    """
    Initialize STCN tracking with first frame mask at frame index = mask_frame_idx

    Args:
        frames: Tensor (1, T, C, H, W)
        first_mask: Binary mask (H, W) uint8, in the ORIGINAL resolution
        mask_frame_idx: The index where the first annotation is given
        resolution: Processing resolution

    Returns:
        processor: InferenceCore instance
    """
    # Prepare mask -> one-hot at processing resolution
    labels = np.unique(first_mask)
    labels = labels[labels != 0]
    if len(labels) == 0:
        raise ValueError("Mask is empty!")

    mask_onehot = all_to_onehot(first_mask, labels)  # (K, H, W)
    mask_tensor = torch.from_numpy(mask_onehot).float()  # (K, H, W)
    mask_tensor = transforms.Resize(
        resolution, interpolation=InterpolationMode.NEAREST
    )(mask_tensor).unsqueeze(1).cuda()  # (K, 1, H', W')

    # with background
    with_bg_msk = torch.cat([1 - torch.sum(mask_tensor, dim=0, keepdim=True), mask_tensor], 0)

    num_objects = len(labels)
    processor = InferenceCore(
        stcn_model,
        frames,
        num_objects=num_objects,
        top_k=20,
        mem_every=5,
        include_last=False
    )

    # IMPORTANT: start propagation with annotation at mask_frame_idx (not 0)
    with torch.amp.autocast('cuda', enabled=True):
        processor.interact(with_bg_msk, mask_frame_idx, frames.shape[1], list(range(1, num_objects + 1)))

    return processor


def extract_all_masks(processor, orig_size):
    """
    Extract all masks from STCN processor (to original size)

    Returns:
        masks: Array (T, H, W) uint8 in {0,255}
    """
    T = processor.t
    H, W = orig_size
    masks = torch.zeros((T, 1, H, W), dtype=torch.uint8, device='cuda')

    for ti in range(T):
        prob = unpad(processor.prob[:, ti], processor.pad)                  # (C, h, w)
        prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=False)
        masks[ti] = torch.argmax(prob, dim=0)

    masks = (masks.detach().cpu().numpy()[:, 0]).astype(np.uint8)
    masks = (masks > 0).astype(np.uint8) * 255
    return masks


# ----------------- Main Visualization -----------------
def play_stcn_tracking(csv_file, recordings_dir, session_id=None):
    """
    Play STCN tracking visualization for a session

    Args:
        csv_file: Path to camera_matches.csv
        recordings_dir: Path to recordings directory
        session_id: Specific session ID, or None for random selection
    """
    # Read CSV
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by session
    sessions = {}
    for row in rows:
        sid = row['session_id']
        if sid not in sessions:
            sessions[sid] = []
        sessions[sid].append(row)

    # Select session
    if session_id is None:
        session_id = random.choice(list(sessions.keys()))

    session_frames = sessions[session_id]
    base_path = Path(recordings_dir) / session_id

    print(f"\n{'='*60}")
    print(f"Session: {session_id}")
    print(f"Total frames: {len(session_frames)}")
    print(f"{'='*60}\n")

    # Step 1: Extract first frame mask with keep_middle, scanning forward until found
    print("Step 1: keep_middle - scanning forward for the first valid mask...")
    first_mask, mask_frame_idx, first_frame = extract_first_frame_mask_keep_middle(
        session_frames, base_path
    )
    if first_mask is None:
        print("Failed to extract a person mask in the session. Aborting.")
        return

    # Step 2: Prepare frames for STCN
    print("\nStep 2: Loading and preparing all frames...")
    frames, orig_size = prepare_stcn_frames(session_frames, base_path)
    print(f"Loaded {frames.shape[1]} frames, tensor shape: {frames.shape}, orig_size={orig_size}")

    # Step 3: Run STCN tracking (annotation at mask_frame_idx)
    print("\nStep 3: Running STCN tracking (start index = mask_frame_idx)...")
    processor = setup_stcn_tracking(frames, first_mask, mask_frame_idx)

    # Step 4: Extract all masks
    print("\nStep 4: Extracting all masks...")
    all_masks = extract_all_masks(processor, orig_size)
    print(f"Extracted {len(all_masks)} masks")

    print(f"\n{'='*60}")
    print("Setup complete! Starting visualization...")
    print(f"Annotation frame index (start): {mask_frame_idx}")
    print(f"{'='*60}\n")

    # Step 5: Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax in axes:
        ax.axis('off')

    # First display frame: we’ll show frame 0 initially, masks are for all frames already
    row0 = session_frames[0]
    img0 = cv2.imread(str(base_path / row0['camera_0_rgb_filename']))
    img1 = cv2.imread(str(base_path / row0['camera_1_rgb_filename']))
    stitched0 = stitch_images(img0, img1)
    rgb0 = cv2.cvtColor(stitched0, cv2.COLOR_BGR2RGB)

    im_rgb = axes[0].imshow(rgb0)
    axes[0].set_title('Original RGB')

    im_mask = axes[1].imshow(all_masks[0], cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('STCN Segmentation Mask')

    overlay0 = rgb0.copy()
    overlay0[all_masks[0] > 128] = [255, 0, 0]
    blend0 = cv2.addWeighted(rgb0, 0.7, overlay0, 0.3, 0)
    im_overlay = axes[2].imshow(blend0)
    axes[2].set_title('Overlay')

    title = plt.suptitle(
        f"Session: {session_id} | Frame: 0/{len(session_frames)} | STCN Tracking (start idx: {mask_frame_idx})"
    )
    plt.tight_layout()

    def update_frame(frame_idx):
        row = session_frames[frame_idx]
        img0 = cv2.imread(str(base_path / row['camera_0_rgb_filename']))
        img1 = cv2.imread(str(base_path / row['camera_1_rgb_filename']))
        stitched = stitch_images(img0, img1)
        rgb = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)

        mask = all_masks[frame_idx]

        im_rgb.set_array(rgb)
        im_mask.set_array(mask)

        overlay = rgb.copy()
        overlay[mask > 128] = [255, 0, 0]
        blend = cv2.addWeighted(rgb, 0.7, overlay, 0.3, 0)
        im_overlay.set_array(blend)

        title.set_text(
            f"Session: {session_id} | Frame: {frame_idx}/{len(session_frames)} | STCN Tracking (start idx: {mask_frame_idx})"
        )
        return im_rgb, im_mask, im_overlay, title

    anim = animation.FuncAnimation(
        fig, update_frame, frames=len(session_frames),
        interval=33, blit=False, repeat=True
    )

    print("Playing STCN tracking animation...")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Play STCN tracking with keep_middle mask selection")
    parser.add_argument("--csv", default="/media/oliver/8068753A68752FD0/0929/camera_matches.csv", help="Path to CSV file")
    parser.add_argument("--recordings", default="/media/oliver/8068753A68752FD0/0929/recordings", help="Path to recordings directory")
    parser.add_argument("--session", default=None, help="Specific session ID (optional)")

    args = parser.parse_args()

    play_stcn_tracking(args.csv, args.recordings, args.session)
