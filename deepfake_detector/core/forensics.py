import base64
import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
from PIL.ExifTags import TAGS

from deepfake_detector.core.face_engine import FaceDetector


class DeepfakeDetector:
    def __init__(self):
        self.face_detector = FaceDetector()

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Comprehensive forensic analysis on an image file.
        Returns detailed report and base64-encoded visual maps.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Load image via OpenCV and PIL
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not decode image at {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = image_bgr.shape[:2]

        # 1. Face Detection
        faces = self.face_detector.detect_faces(image_bgr)
        primary_face = faces[0] if faces else None

        # 2. Error Level Analysis (ELA)
        ela_data = self._compute_ela(image_rgb, primary_face)

        # 3. Noise Residual (PRNU Inconsistency)
        noise_data = self._compute_noise_residual(gray, primary_face)

        # 4. Frequency Domain (2D FFT Spectrum)
        fft_data = self._compute_fft_spectrum(gray, primary_face)

        # 5. Face Boundary Seam Discontinuity
        boundary_data = self._compute_boundary_gradient(gray, primary_face)

        # 6. Metadata & EXIF Analysis
        metadata_data = self._analyze_metadata(image_path)

        # 7. Compute Unified Ensemble Score
        ensemble = self._compute_ensemble_score(
            ela_data=ela_data,
            noise_data=noise_data,
            fft_data=fft_data,
            boundary_data=boundary_data,
            metadata_data=metadata_data,
            has_face=(primary_face is not None)
        )

        # 8. Generate Visual Maps
        annotated_image = self._draw_face_boxes(image_bgr.copy(), faces)
        maps = {
            'annotated': self._image_to_base64(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)),
            'ela_map': self._image_to_base64(ela_data['visual_map']),
            'noise_map': self._image_to_base64(noise_data['visual_map']),
            'fft_map': self._image_to_base64(fft_data['visual_map']),
            'boundary_map': self._image_to_base64(boundary_data['visual_map']),
        }

        return {
            'summary': {
                'verdict': ensemble['verdict'],
                'verdict_code': ensemble['verdict_code'],  # 'genuine', 'suspicious', 'deepfake'
                'authenticity_score': ensemble['authenticity_score'],
                'manipulation_probability': ensemble['manipulation_probability'],
                'confidence': ensemble['confidence'],
                'face_count': len(faces),
                'resolution': f"{w}x{h}",
            },
            'indicators': ensemble['indicators'],
            'metrics': {
                'ela': {
                    'discrepancy_ratio': round(ela_data['discrepancy_ratio'], 3),
                    'global_energy': round(ela_data['global_mean'], 2),
                    'face_energy': round(ela_data.get('face_mean', 0.0), 2),
                    'risk_level': ela_data['risk_level'],
                },
                'noise': {
                    'noise_variance_ratio': round(noise_data['variance_ratio'], 3),
                    'face_noise_std': round(noise_data.get('face_std', 0.0), 2),
                    'bg_noise_std': round(noise_data.get('bg_std', 0.0), 2),
                    'risk_level': noise_data['risk_level'],
                },
                'frequency': {
                    'spectral_grid_anomaly': round(fft_data['grid_anomaly_score'], 3),
                    'high_frequency_energy': round(fft_data['hf_ratio'], 3),
                    'risk_level': fft_data['risk_level'],
                },
                'boundary': {
                    'seam_gradient_step': round(float(boundary_data['gradient_step']), 3),
                    'risk_level': boundary_data['risk_level'],
                },
                'metadata': metadata_data,
            },
            'visual_maps': maps
        }

    def _compute_ela(self, image_rgb: np.ndarray, primary_face: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Error Level Analysis: re-compress image to JPEG 90 and calculate pixel difference.
        """
        # Re-encode to JPEG quality 90
        success, encoded = cv2.imencode('.jpg', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            return {'discrepancy_ratio': 0.0, 'global_mean': 0.0, 'risk_level': 'Low', 'visual_map': image_rgb}

        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        recompressed_rgb = cv2.cvtColor(recompressed, cv2.COLOR_BGR2RGB)

        # Difference
        diff = cv2.absdiff(image_rgb, recompressed_rgb).astype(np.float32)
        diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        global_mean = float(np.mean(diff_gray))
        global_std = float(np.std(diff_gray))

        discrepancy_ratio = 0.0
        face_mean = 0.0
        bg_mean = global_mean

        if primary_face:
            x1, y1, x2, y2 = primary_face['bbox']
            h, w = diff_gray.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            face_region = diff_gray[y1:y2, x1:x2]
            if face_region.size > 0:
                face_mean = float(np.mean(face_region))
                
                # Mask out face to get background
                mask = np.ones_like(diff_gray, dtype=bool)
                mask[y1:y2, x1:x2] = False
                bg_region = diff_gray[mask]
                bg_mean = float(np.mean(bg_region)) if bg_region.size > 0 else global_mean

                # Ratio of face error to background error
                discrepancy_ratio = abs(face_mean - bg_mean) / (bg_mean + 1e-4)

        # Colorized Heatmap for visualization
        scaled_diff = np.clip(diff_gray * 12.0, 0, 255).astype(np.uint8)
        colorized_map = cv2.applyColorMap(scaled_diff, cv2.COLORMAP_TURBO)
        colorized_rgb = cv2.cvtColor(colorized_map, cv2.COLOR_BGR2RGB)

        # Assess risk
        if discrepancy_ratio > 0.45 or global_mean > 14.0:
            risk = 'High'
        elif discrepancy_ratio > 0.22 or global_mean > 9.0:
            risk = 'Moderate'
        else:
            risk = 'Low'

        return {
            'discrepancy_ratio': discrepancy_ratio,
            'global_mean': global_mean,
            'global_std': global_std,
            'face_mean': face_mean,
            'bg_mean': bg_mean,
            'risk_level': risk,
            'visual_map': colorized_rgb
        }

    def _compute_noise_residual(self, gray: np.ndarray, primary_face: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Sensor noise residual (PRNU proxy): high-pass filter by subtracting median blur.
        """
        denoised = cv2.medianBlur(gray, 3)
        residual = cv2.absdiff(gray, denoised).astype(np.float32)

        global_std = float(np.std(residual))
        face_std = global_std
        bg_std = global_std
        variance_ratio = 0.0

        if primary_face:
            x1, y1, x2, y2 = primary_face['bbox']
            h, w = gray.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_res = residual[y1:y2, x1:x2]
            if face_res.size > 0:
                face_std = float(np.std(face_res))
                mask = np.ones_like(residual, dtype=bool)
                mask[y1:y2, x1:x2] = False
                bg_res = residual[mask]
                bg_std = float(np.std(bg_res)) if bg_res.size > 0 else global_std
                variance_ratio = abs(face_std - bg_std) / (bg_std + 1e-4)

        # Visual map: normalized noise residual
        norm_res = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm_color = cv2.cvtColor(norm_res, cv2.COLOR_GRAY2RGB)

        if variance_ratio > 0.40:
            risk = 'High'
        elif variance_ratio > 0.20:
            risk = 'Moderate'
        else:
            risk = 'Low'

        return {
            'variance_ratio': variance_ratio,
            'face_std': face_std,
            'bg_std': bg_std,
            'risk_level': risk,
            'visual_map': norm_color
        }

    def _compute_fft_spectrum(self, gray: np.ndarray, primary_face: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        2D FFT Spectral Analysis: detects periodic grid artifacts typical of GAN/Diffusion upsamplers.
        """
        # Focus on face if present, else center crop
        h, w = gray.shape
        if primary_face:
            x1, y1, x2, y2 = primary_face['bbox']
            patch = gray[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        else:
            patch = gray

        if patch.size < 64 * 64:
            patch = gray

        # Resize patch to power of 2 for clean FFT
        patch_resized = cv2.resize(patch, (256, 256)).astype(np.float32)
        f = np.fft.fft2(patch_resized)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift) + 1e-8
        log_mag = 20 * np.log(magnitude)

        # High-frequency analysis: check for anomalous spikes in mid-high frequencies
        center = 128
        y, x = np.ogrid[:256, :256]
        dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
        
        # High frequency ring
        hf_mask = (dist_from_center > 40) & (dist_from_center < 120)
        hf_values = log_mag[hf_mask]
        
        # Outlier spike detection (GAN checkerboard artifact test)
        q75, q25 = np.percentile(hf_values, [75, 25])
        iqr = q75 - q25
        upper_bound = q75 + (1.8 * iqr)
        spike_ratio = float(np.sum(hf_values > upper_bound) / (hf_values.size + 1e-5))
        grid_anomaly_score = min(1.0, spike_ratio * 25.0)

        hf_ratio = float(np.mean(hf_values) / (np.mean(log_mag) + 1e-4))

        # Visual spectrum map (normalized 0-255)
        spectrum_norm = cv2.normalize(log_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        spectrum_color = cv2.applyColorMap(spectrum_norm, cv2.COLORMAP_VIRIDIS)
        spectrum_rgb = cv2.cvtColor(spectrum_color, cv2.COLOR_BGR2RGB)

        if grid_anomaly_score > 0.45:
            risk = 'High'
        elif grid_anomaly_score > 0.20:
            risk = 'Moderate'
        else:
            risk = 'Low'

        return {
            'grid_anomaly_score': grid_anomaly_score,
            'hf_ratio': hf_ratio,
            'risk_level': risk,
            'visual_map': spectrum_rgb
        }

    def _compute_boundary_gradient(self, gray: np.ndarray, primary_face: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Face boundary gradient discontinuity: detects blending mask seam / feathering.
        """
        h, w = gray.shape
        gradient = cv2.Laplacian(gray, cv2.CV_32F)
        abs_gradient = np.abs(gradient)

        gradient_step = 0.0
        if primary_face:
            x1, y1, x2, y2 = primary_face['bbox']
            x1, y1 = max(10, x1), max(10, y1)
            x2, y2 = min(w - 10, x2), min(h - 10, y2)

            # Boundary strip (inner 5px vs outer 5px)
            if (x2 - x1) > 30 and (y2 - y1) > 30:
                inner_mask = np.zeros_like(gray, dtype=np.uint8)
                cv2.rectangle(inner_mask, (x1 + 6, y1 + 6), (x2 - 6, y2 - 6), 255, -1)
                
                border_mask = np.zeros_like(gray, dtype=np.uint8)
                cv2.rectangle(border_mask, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), 255, -1)
                cv2.rectangle(border_mask, (x1 + 6, y1 + 6), (x2 - 6, y2 - 6), 0, -1)

                inner_grad = float(np.mean(abs_gradient[inner_mask > 0]))
                border_grad = float(np.mean(abs_gradient[border_mask > 0]))
                
                gradient_step = float(abs(inner_grad - border_grad) / (border_grad + 1e-4))

        # Visual gradient map
        grad_norm = np.clip(abs_gradient * 3.0, 0, 255).astype(np.uint8)
        grad_color = cv2.applyColorMap(grad_norm, cv2.COLORMAP_INFERNO)
        grad_rgb = cv2.cvtColor(grad_color, cv2.COLOR_BGR2RGB)

        if gradient_step > 0.45:
            risk = 'High'
        elif gradient_step > 0.22:
            risk = 'Moderate'
        else:
            risk = 'Low'

        return {
            'gradient_step': float(gradient_step),
            'risk_level': risk,
            'visual_map': grad_rgb
        }

    def _analyze_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        Inspect EXIF metadata, camera hardware presence, and AI signatures.
        """
        info = {
            'has_exif': False,
            'camera_make': None,
            'camera_model': None,
            'software': None,
            'ai_marker_detected': False,
            'details': []
        }

        try:
            with Image.open(image_path) as img:
                exif = img.getexif()
                if exif:
                    info['has_exif'] = True
                    for tag_id, val in exif.items():
                        tag = TAGS.get(tag_id, str(tag_id))
                        val_str = str(val).strip()
                        if tag == 'Make':
                            info['camera_make'] = val_str
                        elif tag == 'Model':
                            info['camera_model'] = val_str
                        elif tag == 'Software':
                            info['software'] = val_str
                        
                        # Check text fields for AI generator keywords
                        val_lower = val_str.lower()
                        for marker in ['midjourney', 'stable diffusion', 'comfyui', 'dall-e', 'flux', 'firefly', 'novelai', 'facefusion', 'roop']:
                            if marker in val_lower:
                                info['ai_marker_detected'] = True
                                info['details'].append(f"AI signature '{marker}' found in metadata tag '{tag}'")

                # Check raw PNG / JPEG text comments
                if hasattr(img, 'text') and isinstance(img.text, dict):
                    for k, v in img.text.items():
                        v_lower = str(v).lower()
                        for marker in ['parameters', 'prompt', 'steps:', 'sampler:', 'model_hash', 'sd_model']:
                            if marker in v_lower:
                                info['ai_marker_detected'] = True
                                info['details'].append(f"Generative AI workflow key '{k}' detected in image headers")

        except Exception as e:
            info['details'].append(f"Metadata read error: {str(e)}")

        return info

    def _compute_ensemble_score(
        self,
        ela_data: Dict[str, Any],
        noise_data: Dict[str, Any],
        fft_data: Dict[str, Any],
        boundary_data: Dict[str, Any],
        metadata_data: Dict[str, Any],
        has_face: bool
    ) -> Dict[str, Any]:
        """
        Weighted ensemble to compute final authenticity score and manipulation probability.
        """
        score_manipulation = 0.0
        indicators = []

        # 1. ELA Score (Weight 30%)
        ela_ratio = ela_data['discrepancy_ratio']
        if ela_ratio > 0.45:
            score_manipulation += 30.0
            indicators.append("Severe JPEG compression discrepancy detected between face and background (typical of face swaps).")
        elif ela_ratio > 0.22:
            score_manipulation += 15.0
            indicators.append("Moderate compression level variance across image regions.")
        else:
            indicators.append("Compression error levels are uniform across regions.")

        # 2. Noise Residual PRNU (Weight 25%)
        noise_ratio = noise_data['variance_ratio']
        if noise_ratio > 0.40:
            score_manipulation += 25.0
            indicators.append("High sensor noise inconsistency: subject area exhibits artificial smoothing or mismatched noise pattern.")
        elif noise_ratio > 0.20:
            score_manipulation += 12.0
            indicators.append("Mild noise pattern variance between central subject and surroundings.")
        else:
            indicators.append("Camera sensor noise distribution is consistent across the frame.")

        # 3. FFT Frequency Anomaly (Weight 20%)
        grid_anomaly = fft_data['grid_anomaly_score']
        if grid_anomaly > 0.45:
            score_manipulation += 20.0
            indicators.append("Frequency domain shows periodic grid/checkerboard upsampling peaks (characteristic of GANs/Diffusion).")
        elif grid_anomaly > 0.20:
            score_manipulation += 10.0
            indicators.append("Subtle high-frequency spectral deviations observed.")
        else:
            indicators.append("Natural 1/f spectral power falloff without synthetic periodic spikes.")

        # 4. Boundary Seam Discontinuity (Weight 15%)
        if has_face:
            bound_step = boundary_data['gradient_step']
            if bound_step > 0.45:
                score_manipulation += 15.0
                indicators.append("Sharp gradient discontinuity at facial contour indicating blending mask seams.")
            elif bound_step > 0.20:
                score_manipulation += 8.0
                indicators.append("Minor gradient transition variance near face boundary.")

        # 5. Metadata Factors (Weight 10%)
        if metadata_data['ai_marker_detected']:
            score_manipulation += 25.0
            indicators.append("Definitive AI generation markers identified in image metadata.")
        elif metadata_data['camera_make'] and metadata_data['camera_model']:
            score_manipulation = max(0.0, score_manipulation - 15.0)
            indicators.append(f"Authentic camera hardware tags verified: {metadata_data['camera_make']} {metadata_data['camera_model']}.")
        elif not metadata_data['has_exif']:
            indicators.append("EXIF metadata stripped (common for social media & generated files).")

        manipulation_probability = min(99.0, max(1.0, round(score_manipulation, 1)))
        authenticity_score = round(100.0 - manipulation_probability, 1)

        # Verdict assignment
        if manipulation_probability >= 60.0:
            verdict = "High Probability Deepfake / Manipulated"
            verdict_code = "deepfake"
            confidence = "High" if manipulation_probability >= 75 else "Medium"
        elif manipulation_probability >= 30.0:
            verdict = "Suspicious / Inconclusive"
            verdict_code = "suspicious"
            confidence = "Medium"
        else:
            verdict = "Likely Genuine / Authentic Photo"
            verdict_code = "genuine"
            confidence = "High" if authenticity_score >= 80 else "Medium"

        return {
            'verdict': verdict,
            'verdict_code': verdict_code,
            'authenticity_score': authenticity_score,
            'manipulation_probability': manipulation_probability,
            'confidence': confidence,
            'indicators': indicators
        }

    def _draw_face_boxes(self, image_bgr: np.ndarray, faces: List[Dict[str, Any]]) -> np.ndarray:
        for idx, f in enumerate(faces):
            x1, y1, x2, y2 = f['bbox']
            score = f.get('score', 1.0)
            # Draw sleek glowing rectangle
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 240, 255), 2)
            label = f"Face #{idx+1} ({int(score * 100)}%)"
            cv2.putText(image_bgr, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 255), 2)
        return image_bgr

    def _image_to_base64(self, img_rgb: np.ndarray) -> str:
        """Convert RGB numpy array to base64 JPEG string."""
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if success:
            b64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{b64}"
        return ""
