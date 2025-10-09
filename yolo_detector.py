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

        # Send ready signal via stdout (binary protocol)
        sys.stdout.buffer.write(b"READY\n")
        sys.stdout.buffer.flush()

    def detect_keep_middle(self, frame_bgr):
        """使用YOLO提取最靠近中线的人体mask"""
        h, w = frame_bgr.shape[:2]
        cx_mid = w * 0.5

        results = self.model.predict(
            frame_bgr[..., ::-1],
            classes=[0],
            conf=self.conf_threshold,
            verbose=False,
            device="cpu",
            half=False,
            amp=False
        )

        candidates = []
        total = 0

        for r in results:
            if r.masks is None:
                continue

            for m, box in zip(r.masks.data, r.boxes.xyxy):
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                box_h = y2 - y1
                if box_h <= 0:
                    continue

                # Check if at least 1/3 in bottom half
                inter_h = max(0, min(y2, h) - max(y1, h // 2))
                ratio_in_bottom = inter_h / box_h

                if ratio_in_bottom < 1/3:
                    continue

                total += 1
                arr = m.detach().cpu().numpy()
                if arr.shape[-2:] != (h, w):
                    arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)

                mask = (arr > 0.5).astype(np.uint8) * 255

                # Calculate centroid
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
