#!/usr/bin/env python3
"""
独立的YOLO检测进程
通过stdin/stdout与主程序通信，完全隔离CUDA上下文
"""

import sys
import cv2
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO

class YOLODetector:
    def __init__(self, conf_threshold=0.3):
        self.conf_threshold = conf_threshold
        self.model = YOLO("yolov8n-seg.pt")

        # Send ready signal immediately (warmup on first actual prediction)
        sys.stdout.buffer.write(b"READY\n")
        sys.stdout.buffer.flush()

        # Flag to track first prediction warmup
        self.first_run = True

    def detect_keep_middle(self, frame_bgr):
        """使用YOLO提取最靠近中线的人体mask"""
        h, w = frame_bgr.shape[:2]
        cx_mid = w * 0.5

        # Downsample to 512 width for faster YOLO processing
        # Input is already 640x360 from main tracker, downsample to ~512x288
        target_w = 512
        scale = target_w / w
        ds_h = int(h * scale)
        ds_w = target_w

        frame_ds = cv2.resize(frame_bgr, (ds_w, ds_h), interpolation=cv2.INTER_AREA)

        # Use FP16 and optimized parameters for faster inference on Jetson
        # Note: First run may take longer due to CUDA initialization
        results = self.model.predict(
            frame_ds[..., ::-1],
            classes=[0],           # Person class only
            conf=self.conf_threshold,
            verbose=False,
            device="cuda",
            half=True,             # FP16 for ~2x speedup on Jetson
            imgsz=512,             # Match downsampled size
            max_det=10             # Limit max detections for speed
        )

        self.first_run = False

        candidates = []
        total = 0

        for r in results:
            if r.masks is None:
                continue

            for m, box in zip(r.masks.data, r.boxes.xyxy):
                y1, y2 = box.cpu().numpy().astype(int)[[1, 3]]
                box_h = y2 - y1
                if box_h <= 0:
                    continue

                # Check if at least 1/3 in bottom half (in downsampled coordinates)
                inter_h = max(0, min(y2, ds_h) - max(y1, ds_h // 2))
                ratio_in_bottom = inter_h / box_h

                if ratio_in_bottom < 1/3:
                    continue

                total += 1
                arr = m.detach().cpu().numpy()

                # Resize mask to downsampled size first
                if arr.shape[-2:] != (ds_h, ds_w):
                    arr = cv2.resize(arr, (ds_w, ds_h), interpolation=cv2.INTER_NEAREST)

                mask_ds = (arr > 0.5).astype(np.uint8) * 255

                # Upscale mask back to original size
                mask = cv2.resize(mask_ds, (w, h), interpolation=cv2.INTER_NEAREST)

                # Calculate centroid in original coordinates
                M = cv2.moments(mask)
                if M["m00"] <= 1e-6:
                    continue
                cx = M["m10"] / M["m00"]
                abs_dx = abs(cx - cx_mid)

                candidates.append((abs_dx, mask))

        if len(candidates) == 0:
            return None, 0

        # Select closest to midline
        candidates.sort(key=lambda x: x[0])
        final_mask = candidates[0][1].astype(np.uint8)
        return final_mask, total

    def run(self):
        """主循环：接收图像，返回mask"""
        while True:
            try:
                # Read size (4 bytes)
                size_bytes = sys.stdin.buffer.read(4)
                if len(size_bytes) != 4:
                    break

                size = int.from_bytes(size_bytes, 'little')

                # Read image data - read in chunks to avoid truncation
                data = b''
                remaining = size
                while remaining > 0:
                    chunk = sys.stdin.buffer.read(remaining)
                    if not chunk:
                        break
                    data += chunk
                    remaining -= len(chunk)

                if len(data) != size:
                    continue

                frame_bgr = pickle.loads(data)

                # Detect
                mask, num_det = self.detect_keep_middle(frame_bgr)

                # Send result
                result = {'mask': mask, 'num_det': num_det}
                result_data = pickle.dumps(result)
                result_size = len(result_data).to_bytes(4, 'little')

                sys.stdout.buffer.write(result_size)
                sys.stdout.buffer.write(result_data)
                sys.stdout.buffer.flush()

            except Exception as e:
                print(f"ERROR: {e}", file=sys.stderr, flush=True)
                break

if __name__ == "__main__":
    detector = YOLODetector()
    detector.run()
