import os
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


class FaceDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.session = None
        if HAS_ORT:
            # Check model in modulo directory
            candidates = [
                model_path,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '../modulo/modulo.onnx')),
                os.path.abspath(os.path.join(os.path.dirname(__file__), '../modulo/yoloface_8n.onnx')),
                os.path.abspath(os.path.join(os.path.dirname(__file__), '../../modulo/modulo.onnx')),
                os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/yoloface_8n.onnx')),
            ]
            for cand in candidates:
                if cand and os.path.exists(cand):
                    try:
                        self.session = ort.InferenceSession(cand, providers=['CPUExecutionProvider'])
                        break
                    except Exception:
                        pass

    def detect_faces(self, image_bgr: np.ndarray, min_score: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect faces in image.
        Returns list of dicts: [{'bbox': [x1, y1, x2, y2], 'score': float}]
        """
        if self.session is None:
            return self._heuristic_fallback(image_bgr)

        try:
            h, w = image_bgr.shape[:2]
            target_size = (640, 640)
            
            # Letterbox / resize for YOLO
            resized = cv2.resize(image_bgr, target_size)
            input_tensor = resized.astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC to CHW
            input_tensor = np.expand_dims(input_tensor, axis=0)    # NCHW

            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})[0]
            
            # Output shape is (1, 20, 8400) or (1, 16, 8400)
            detections = np.squeeze(outputs)
            if detections.shape[0] < detections.shape[1]:
                detections = detections.T  # -> (8400, channels)

            ratio_x = w / float(target_size[0])
            ratio_y = h / float(target_size[1])

            boxes = []
            scores = []

            for row in detections:
                score = float(row[4])
                if score >= min_score:
                    cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                    x1 = int(max(0, (cx - bw / 2.0) * ratio_x))
                    y1 = int(max(0, (cy - bh / 2.0) * ratio_y))
                    x2 = int(min(w, (cx + bw / 2.0) * ratio_x))
                    y2 = int(min(h, (cy + bh / 2.0) * ratio_y))
                    
                    if (x2 - x1) > 20 and (y2 - y1) > 20:
                        boxes.append([x1, y1, x2 - x1, y2 - y1])
                        scores.append(score)

            # Apply Non-Maximum Suppression (NMS)
            if boxes:
                indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=min_score, nms_threshold=0.45)
                results = []
                if len(indices) > 0:
                    for idx in indices.flatten():
                        bx, by, bw, bh = boxes[idx]
                        results.append({
                            'bbox': [bx, by, bx + bw, by + bh],
                            'score': round(float(scores[idx]), 3)
                        })
                return results
            return []
        except Exception:
            return self._heuristic_fallback(image_bgr)

    def _heuristic_fallback(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Fallback: If no deep model loaded, approximate the central region as primary subject.
        """
        h, w = image_bgr.shape[:2]
        pad_x = int(w * 0.25)
        pad_y = int(h * 0.20)
        return [{
            'bbox': [pad_x, pad_y, w - pad_x, int(h * 0.80)],
            'score': 0.5,
            'is_fallback': True
        }]
