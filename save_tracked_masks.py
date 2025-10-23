#!/usr/bin/env python3
"""
Real-time STCN Tracking with automatic first frame mask extraction (keep_middle)
Batch mode: run on ALL sessions and save masks with perspective unwarped.

What this script does now:
1) 扫描 camera_matches.csv 的所有 session（每个 session 一次 STCN 跟踪）
2) 对每一帧保存二值 mask 到 masks/<session_id>/<frame_id>.png
3) 保存前对 mask 做【透视变换还原】(inverse warp)，再按公共宽度拼接成上下两段输出
4) 初始标注帧仍然用 keep_middle 自动扫描获得；STCN 从该帧开始

Notes:
- 需要 CSV 中有 'session_id', 'camera_0_rgb_filename', 'camera_1_rgb_filename', 'frame_id' 四列
- STCN 权重路径为 saves/stcn.pth；YOLO 使用 yolov8n-seg.pt
"""

import os
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
SAVE_ROOT = "masks"  # 保存根目录，每个 session 一个子目录
BATCH_SIZE = 200  # 每批处理的帧数，防止内存溢出

# ----------------- Load models -----------------
print("Loading YOLO segmentation model...")
yolo_model = YOLO("yolov8n-seg.pt")

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


# ----------------- Perspective helpers -----------------
def correct_camera_perspective_with_meta(image, camera_id, angle_degrees=25):
    """
    Warp a single camera image with perspective transform and also return metadata
    so we can later invert the warp on masks.

    Returns:
        warped, meta
        meta keys:
          - 'M': forward 3x3 homography (orig -> warped)
          - 'orig_size': (h, w)
          - 'warped_size': (h, w)  # same as orig here
          - 'camera_id': 0 or 1
    """
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    offset = int(w * angle_degrees / 90)

    if camera_id == 0:  # Top camera, narrow bottom
        dst = np.float32([[0, 0], [w, 0], [w - offset, h], [offset, h]])
    else:  # Bottom camera, narrow top
        dst = np.float32([[offset, 0], [w - offset, 0], [w, h], [0, h]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (w, h))

    meta = {
        'M': M.astype(np.float32),
        'orig_size': (h, w),
        'warped_size': (h, w),
        'camera_id': camera_id
    }
    return warped, meta


def stitch_images_with_meta(img_top, img_bottom, angle=25):
    """
    Warp both cameras, then resize by common width and vertically stack.
    Return stitched image and full metadata to enable unwarping later.

    Returns:
        stitched (Hc, Wc, 3), meta dict:
          - top: {M, orig_size, warped_size, scale, resized_size}
          - bottom: {M, orig_size, warped_size, scale, resized_size}
          - resize_w: common width
          - split_h: resized height of top half in the stitched image
    """
    warped_top, meta_top = correct_camera_perspective_with_meta(img_top, 0, angle)
    warped_bottom, meta_bottom = correct_camera_perspective_with_meta(img_bottom, 1, angle)

    w_common = min(warped_top.shape[1], warped_bottom.shape[1])
    scale_top = w_common / warped_top.shape[1]
    scale_bottom = w_common / warped_bottom.shape[1]

    top_resized_h = int(round(warped_top.shape[0] * scale_top))
    bot_resized_h = int(round(warped_bottom.shape[0] * scale_bottom))

    top_resized = cv2.resize(warped_top, (w_common, top_resized_h), interpolation=cv2.INTER_LINEAR)
    bot_resized = cv2.resize(warped_bottom, (w_common, bot_resized_h), interpolation=cv2.INTER_LINEAR)

    stitched = np.vstack([top_resized, bot_resized])

    meta = {
        'top': {
            **meta_top,
            'scale': scale_top,
            'resized_size': (top_resized_h, w_common),
        },
        'bottom': {
            **meta_bottom,
            'scale': scale_bottom,
            'resized_size': (bot_resized_h, w_common),
        },
        'resize_w': w_common,
        'split_h': top_resized_h
    }
    return stitched, meta


def unwarp_stitched_mask_to_original_pair(mask_stitched, meta):
    """
    Given a stitched binary mask (uint8 0/255) and the warp/resize metadata,
    return a single vertically-stacked mask whose upper part corresponds to cam0 (unwarped),
    lower part corresponds to cam1 (unwarped). Both parts are resized to the same common width
    (meta['resize_w']) so they can be vertically stacked.

    Steps:
      1) Split stitched mask into top/bottom by split_h
      2) Resize each split back to warped size (undo "resize to common width")
      3) Apply inverse perspective transform to original camera plane
      4) Resize each unwarped mask to common width and vstack -> final image
    """
    Hc, Wc = mask_stitched.shape[:2]
    split_h = meta['split_h']
    assert Wc == meta['resize_w'], "Mask width and meta.resize_w mismatch"

    top_mask_resized = mask_stitched[:split_h, :]
    bot_mask_resized = mask_stitched[split_h:, :]

    # --- Cam0 (top) ---
    scale0 = meta['top']['scale']
    h0_warp, w0_warp = meta['top']['warped_size']
    # resize back to warped size
    top_mask_warp = cv2.resize(
        top_mask_resized, (w0_warp, h0_warp), interpolation=cv2.INTER_NEAREST
    )

    # inverse perspective: warped -> original
    M0 = meta['top']['M']
    Minv0 = np.linalg.inv(M0)
    h0_orig, w0_orig = meta['top']['orig_size']
    top_mask_unwarped = cv2.warpPerspective(
        top_mask_warp, Minv0.astype(np.float32), (w0_orig, h0_orig), flags=cv2.INTER_NEAREST
    )

    # --- Cam1 (bottom) ---
    scale1 = meta['bottom']['scale']
    h1_warp, w1_warp = meta['bottom']['warped_size']
    bot_mask_warp = cv2.resize(
        bot_mask_resized, (w1_warp, h1_warp), interpolation=cv2.INTER_NEAREST
    )
    M1 = meta['bottom']['M']
    Minv1 = np.linalg.inv(M1)
    h1_orig, w1_orig = meta['bottom']['orig_size']
    bot_mask_unwarped = cv2.warpPerspective(
        bot_mask_warp, Minv1.astype(np.float32), (w1_orig, h1_orig), flags=cv2.INTER_NEAREST
    )

    # 为了能上下拼接，统一一个公共宽度（沿用 stitched 时的 resize_w）
    w_common = meta['resize_w']
    top_out_h = int(round(h0_orig * (w_common / w0_orig)))
    bot_out_h = int(round(h1_orig * (w_common / w1_orig)))

    top_out = cv2.resize(top_mask_unwarped, (w_common, top_out_h), interpolation=cv2.INTER_NEAREST)
    bot_out = cv2.resize(bot_mask_unwarped, (w_common, bot_out_h), interpolation=cv2.INTER_NEAREST)

    final = np.vstack([top_out, bot_out]).astype(np.uint8)
    return final


# ----------------- keep_middle Mask Extraction -----------------
def extract_mask_from_frame_keep_middle(frame_bgr):
    """
    Extract person segmentation from a single frame using YOLO,
    keep only detections with >=1/3 of bbox in bottom half,
    then choose the mask whose centroid is closest to midline.
    """
    h, w = frame_bgr.shape[:2]
    cx_mid = w * 0.5

    results = yolo_model.predict(
        frame_bgr[..., ::-1], classes=[0], conf=CONF, verbose=False, device="cpu"
    )

    candidates = []  # (abs_dx, mask_uint8)
    total = 0

    for r in results:
        if r.masks is None:
            continue

        for m, box in zip(r.masks.data, r.boxes.xyxy):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            box_h = y2 - y1
            if box_h <= 0:
                continue

            # intersection of bbox with lower half
            inter_h = max(0, min(y2, h) - max(y1, h // 2))
            ratio_in_bottom = inter_h / box_h

            # ✅ 至少 1/3 在下半部分
            if ratio_in_bottom < 1/3:
                continue

            total += 1
            arr = m.detach().cpu().numpy()
            if arr.shape[-2:] != (h, w):
                arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)

            mask = (arr > 0.5).astype(np.uint8) * 255

            # 计算质心
            M = cv2.moments(mask)
            if M["m00"] <= 1e-6:
                continue
            cx = M["m10"] / M["m00"]
            abs_dx = abs(cx - cx_mid)

            candidates.append((abs_dx, mask))

    if len(candidates) == 0:
        return None, 0

    # 选最靠近垂直中线的
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

        stitched_bgr, _ = stitch_images_with_meta(img0, img1)
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
        orig_size: Original (H, W) of stitched frames
        stitch_metas: list of per-frame meta for unwarping saves
        stitched_rgb_list: list of stitched RGB arrays (for visualization, optional)
    """
    im_transform = transforms.Compose([
        transforms.ToTensor(),
        im_normalization,
        transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC, antialias=True),
    ])

    frame_tensors = []
    stitch_metas = []
    stitched_rgb_list = []
    orig_size = None

    for row in session_frames:
        # Load and stitch (with meta)
        img0 = cv2.imread(str(base_path / row['camera_0_rgb_filename']))
        img1 = cv2.imread(str(base_path / row['camera_1_rgb_filename']))
        if img0 is None or img1 is None:
            continue

        stitched_bgr, meta = stitch_images_with_meta(img0, img1)
        stitch_metas.append(meta)

        if orig_size is None:
            orig_size = stitched_bgr.shape[:2]  # (H, W)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(stitched_bgr, cv2.COLOR_BGR2RGB)
        stitched_rgb_list.append(frame_rgb.copy())

        # Transform -> tensor
        from PIL import Image
        frame_pil = Image.fromarray(frame_rgb)
        frame_tensor = im_transform(frame_pil)
        frame_tensors.append(frame_tensor)

    frames = torch.stack(frame_tensors, 0).unsqueeze(0)  # (1, T, C, H, W)
    return frames, orig_size, stitch_metas, stitched_rgb_list


def setup_stcn_tracking(frames, first_mask, mask_frame_idx, resolution=480):
    # 1) 先确保 first_mask 真是二值(0/255 -> 0/1)
    first_mask_bin = (first_mask > 0).astype(np.uint8) * 1

    labels = np.unique(first_mask_bin)
    labels = labels[labels != 0]
    if len(labels) == 0:
        raise ValueError("Mask is empty!")

    mask_onehot = all_to_onehot(first_mask_bin, labels)  # (K, H, W)
    mask_tensor = torch.from_numpy(mask_onehot).float()  # (K, H, W)

    # 🎯 关键修复：把掩码缩放到和 frames 完全相同的空间尺寸 (H′, W′)
    Hp, Wp = int(frames.shape[-2]), int(frames.shape[-1])  # 540, 480
    mask_tensor = transforms.Resize(
        (Hp, Wp), interpolation=InterpolationMode.NEAREST
    )(mask_tensor).unsqueeze(1).cuda()  # (K, 1, H′, W′)

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

    with torch.amp.autocast('cuda', enabled=True):
        processor.interact(with_bg_msk, mask_frame_idx, frames.shape[1], list(range(1, num_objects + 1)))

    return processor



def extract_all_masks(processor, orig_size):
    """
    Extract all masks from STCN processor (to original stitched size)

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


# ----------------- Batch run on ALL sessions + save -----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def run_all_sessions_and_save(csv_file, recordings_dir, save_root=SAVE_ROOT, batch_size=BATCH_SIZE):
    """
    For each session in CSV:
      - find first keep_middle mask frame
      - run STCN in batches to avoid memory overflow
      - for each frame: unwarp mask back to original camera planes and save as masks/<session>/<frame_id>.png
    """
    # Read CSV
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by session
    sessions = {}
    for row in rows:
        sid = row['session_id']
        sessions.setdefault(sid, []).append(row)

    total_sessions = len(sessions)
    print(f"\nFound {total_sessions} session(s). Starting batch processing...\n")

    save_root_path = Path(save_root)
    ensure_dir(save_root_path)

    # Iterate sessions
    for si, (session_id, session_frames) in enumerate(sessions.items(), 1):
        print(f"\n{'='*70}")
        print(f"[{si}/{total_sessions}] Session: {session_id} | Total frames: {len(session_frames)}")
        print(f"{'='*70}\n")

        base_path = Path(recordings_dir) / session_id
        if not base_path.exists():
            print(f"!! Skip: base path not found: {base_path}")
            continue

        # Step 1: Get initial mask with keep_middle
        print("Step 1: keep_middle - scanning forward for the first valid mask...")
        first_mask, mask_frame_idx, _ = extract_first_frame_mask_keep_middle(session_frames, base_path)
        if first_mask is None:
            print("!! No valid initial mask in this session. Skipping...")
            continue

        # Create session save directory
        session_save_dir = save_root_path / session_id
        ensure_dir(session_save_dir)

        # 分批处理以避免内存溢出
        total_frames = len(session_frames)
        num_batches = (total_frames + batch_size - 1) // batch_size
        print(f"\nProcessing {total_frames} frames in {num_batches} batch(es) of up to {batch_size} frames each...")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_frames)
            batch_frames = session_frames[start_idx:end_idx]

            print(f"\n--- Batch {batch_idx + 1}/{num_batches}: frames {start_idx}-{end_idx-1} ---")

            # Step 2: Prepare frames + metas (for unwarp later)
            print(f"Loading {len(batch_frames)} frames...")
            frames, orig_size, stitch_metas, _ = prepare_stcn_frames(batch_frames, base_path)
            print(f"Loaded {frames.shape[1]} frames, tensor shape: {frames.shape}, orig_size={orig_size}")

            # Adjust mask_frame_idx for this batch
            batch_mask_idx = mask_frame_idx - start_idx
            if batch_mask_idx < 0:
                # 初始 mask 在之前的 batch 中，使用第一帧
                batch_mask_idx = 0
            elif batch_mask_idx >= len(batch_frames):
                # 初始 mask 在之后的 batch 中，使用第一帧
                batch_mask_idx = 0

            # 如果是第一个 batch 且包含初始 mask，使用原始 first_mask
            # 否则从第一帧重新提取 mask（或从上一批次的最后一帧继续）
            if batch_idx == 0 and mask_frame_idx >= start_idx and mask_frame_idx < end_idx:
                current_mask = first_mask
                current_mask_idx = batch_mask_idx
            else:
                # 从当前 batch 的第一帧重新提取 mask
                img0 = cv2.imread(str(base_path / batch_frames[0]['camera_0_rgb_filename']))
                img1 = cv2.imread(str(base_path / batch_frames[0]['camera_1_rgb_filename']))
                stitched_bgr, _ = stitch_images_with_meta(img0, img1)
                current_mask, _ = extract_mask_from_frame_keep_middle(stitched_bgr)
                if current_mask is None:
                    print(f"⚠ Warning: Could not extract mask for batch {batch_idx + 1}, skipping...")
                    continue
                current_mask_idx = 0

            # Step 3: STCN
            print(f"Running STCN tracking (start index = {current_mask_idx})...")
            processor = setup_stcn_tracking(frames, current_mask, current_mask_idx)

            # Step 4: Collect masks (stitched space)
            print("Extracting masks...")
            all_masks = extract_all_masks(processor, orig_size)
            print(f"Extracted {len(all_masks)} masks")

            # Step 5: Save masks per frame (unwarp -> save)
            print("Saving unwarped masks...")
            for local_idx, row in enumerate(batch_frames):
                # stitched-space mask for this frame
                m_stitched = all_masks[local_idx]

                # unwarp back to original camera planes and vertically stack to common width
                meta = stitch_metas[local_idx]
                m_unwarped = unwarp_stitched_mask_to_original_pair(m_stitched, meta)  # uint8 0/255

                # filename by frame_id from CSV (fallback to index if missing)
                frame_id = row.get('frame_id', None)
                if frame_id is None or len(str(frame_id)) == 0:
                    frame_id = str(start_idx + local_idx).zfill(6)

                out_path = session_save_dir / f"{frame_id}.png"
                cv2.imwrite(str(out_path), m_unwarped)

            print(f"✓ Batch {batch_idx + 1}/{num_batches} saved")

            # 清理内存
            del frames, processor, all_masks, stitch_metas
            torch.cuda.empty_cache()

        print(f"\n✓ Session complete. All masks saved to: {session_save_dir}")

    print("\nAll sessions processed. Done!")


# ----------------- (Optional) Original single-session visualize -----------------
def play_stcn_tracking(csv_file, recordings_dir, session_id=None):
    """
    (Unchanged) Optional visualization on a single session.
    """
    # Read CSV
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by session
    sessions = {}
    for row in rows:
        sid = row['session_id']
        sessions.setdefault(sid, []).append(row)

    # Select session
    if session_id is None:
        session_id = random.choice(list(sessions.keys()))

    session_frames = sessions[session_id]
    base_path = Path(recordings_dir) / session_id

    print(f"\n{'='*60}")
    print(f"Session: {session_id}")
    print(f"Total frames: {len(session_frames)}")
    print(f"{'='*60}\n")

    # Step 1
    print("Step 1: keep_middle - scanning forward for the first valid mask...")
    first_mask, mask_frame_idx, first_frame = extract_first_frame_mask_keep_middle(session_frames, base_path)
    if first_mask is None:
        print("Failed to extract a person mask in the session. Aborting.")
        return

    # Step 2
    print("\nStep 2: Loading and preparing all frames...")
    frames, orig_size, _, stitched_rgb_list = prepare_stcn_frames(session_frames, base_path)
    print(f"Loaded {frames.shape[1]} frames, tensor shape: {frames.shape}, orig_size={orig_size}")

    # Step 3
    print("\nStep 3: Running STCN tracking (start index = mask_frame_idx)...")
    processor = setup_stcn_tracking(frames, first_mask, mask_frame_idx)

    # Step 4
    print("\nStep 4: Extracting all masks...")
    all_masks = extract_all_masks(processor, orig_size)
    print(f"Extracted {len(all_masks)} masks")

    print(f"\n{'='*60}")
    print("Setup complete! Starting visualization...")
    print(f"Annotation frame index (start): {mask_frame_idx}")
    print(f"{'='*60}\n")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax in axes:
        ax.axis('off')

    rgb0 = stitched_rgb_list[0]
    im_rgb = axes[0].imshow(rgb0)
    axes[0].set_title('Original RGB (stitched)')

    im_mask = axes[1].imshow(all_masks[0], cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('STCN Segmentation Mask (stitched)')

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
        rgb = stitched_rgb_list[frame_idx]
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


# ----------------- CLI -----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch STCN tracking on ALL sessions and save unwarped masks")
    parser.add_argument("--subject", type=str, default='', help="Subject name for path formatting (e.g., recordings_{subject})")
    parser.add_argument("--csv", default=None, help="Path to CSV file (default: ../camera_matches_{subject}.csv)")
    parser.add_argument("--recordings", default=None, help="Path to recordings directory (default: ../recordings_{subject})")
    parser.add_argument("--save_root", default=None, help="Where to save masks (default: ../masks_{subject})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help=f"Number of frames per batch (default: {BATCH_SIZE})")
    parser.add_argument("--viz_one", default=None, help="Optional: visualize one session id instead of batch saving")

    args = parser.parse_args()

    # Build paths with subject suffix (in parent directory)
    subject_suffix = f"_{args.subject}" if args.subject else ""

    csv_file = args.csv if args.csv else f"../camera_matches{subject_suffix}.csv"
    recordings_dir = args.recordings if args.recordings else f"../recordings{subject_suffix}"
    save_root = args.save_root if args.save_root else f"../masks{subject_suffix}"

    if args.viz_one is not None:
        play_stcn_tracking(csv_file, recordings_dir, args.viz_one)
    else:
        run_all_sessions_and_save(csv_file, recordings_dir, save_root, args.batch_size)
