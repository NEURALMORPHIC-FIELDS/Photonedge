# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.

"""PhotonEdge Web Processing Pipeline.

Wraps all core modules into a configurable real-time processing pipeline.
Supports live camera frames, uploaded images, and synthetic demo shapes.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# Bootstrap core imports
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from core.edges import edges_strict_zero_cross, robust_sigma_mad, adaptive_threshold
from core.fusion import fuse_v2, fuse_or, fuse_backbone_with_fine
from core.metrics import edge_metrics_symmetric, gt_edges_from_binary
from core.shapes import SHAPES, IN_BAND_SHAPES, NYQUIST_SHAPES

# Optional heavy dependencies
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Application mode presets
# ---------------------------------------------------------------------------

MODE_PRESETS: Dict[str, dict] = {
    "edge_detection": {
        "name": "Edge Detection",
        "icon": "scan",
        "description": "Pure multi-scale DoG edge detection with Fusion v2",
        "scales": [[0.8, 1.6], [1.2, 2.4], [2.0, 4.0]],
        "edge_t": 2.5,
        "smooth_sigma": 0.9,
        "fusion_mode": "v2",
        "coverage_radius": 3,
        "denoise": False,
        "contrast": False,
        "pseudocolor": False,
        "detect_shapes": False,
        "detect_objects": False,
        "depth": False,
        "tracking": False,
    },
    "night_enhancement": {
        "name": "Night Enhancement",
        "icon": "moon",
        "description": "Edge-guided low-light image enhancement with adaptive denoising",
        "scales": [[0.8, 1.6], [1.2, 2.4], [2.0, 4.0]],
        "edge_t": 2.0,
        "smooth_sigma": 0.9,
        "fusion_mode": "v2",
        "coverage_radius": 3,
        "denoise": True,
        "contrast": True,
        "pseudocolor": False,
        "detect_shapes": False,
        "detect_objects": False,
        "depth": False,
        "tracking": False,
    },
    "object_perception": {
        "name": "Object Perception",
        "icon": "box",
        "description": "Edge-guided object and geometric shape detection with tracking",
        "scales": [[1.0, 2.0], [1.5, 3.0], [2.5, 5.0]],
        "edge_t": 2.2,
        "smooth_sigma": 0.9,
        "fusion_mode": "v2",
        "coverage_radius": 3,
        "denoise": False,
        "contrast": False,
        "pseudocolor": False,
        "detect_shapes": True,
        "detect_objects": True,
        "depth": False,
        "tracking": True,
    },
    "industrial_inspection": {
        "name": "Industrial Inspection",
        "icon": "settings",
        "description": "High-precision sub-pixel edge detection for defect analysis",
        "scales": [[0.5, 1.0], [0.8, 1.6], [1.2, 2.4]],
        "edge_t": 1.8,
        "smooth_sigma": 0.5,
        "fusion_mode": "v2",
        "coverage_radius": 2,
        "denoise": True,
        "contrast": True,
        "pseudocolor": True,
        "detect_shapes": True,
        "detect_objects": False,
        "depth": False,
        "tracking": False,
    },
    "full_perception": {
        "name": "Full Perception",
        "icon": "cpu",
        "description": "Complete perception stack: all features enabled",
        "scales": [[0.8, 1.6], [1.2, 2.4], [2.0, 4.0]],
        "edge_t": 2.2,
        "smooth_sigma": 0.9,
        "fusion_mode": "v2",
        "coverage_radius": 3,
        "denoise": True,
        "contrast": True,
        "pseudocolor": False,
        "detect_shapes": True,
        "detect_objects": True,
        "depth": True,
        "tracking": True,
    },
}


class PhotonEdgePipeline:
    """Complete PhotonEdge processing pipeline for web application."""

    def __init__(self, mode: str = "edge_detection"):
        self.mode = mode
        self.params: dict = {}
        self.set_mode(mode)
        self._yolo = None
        self._midas = None
        self._midas_transform = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_mode(self, mode: str):
        if mode not in MODE_PRESETS:
            mode = "edge_detection"
        self.mode = mode
        self.params = {k: v for k, v in MODE_PRESETS[mode].items()}

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.params:
                self.params[k] = v

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_yolo(self):
        if self._yolo is None and YOLO_AVAILABLE:
            try:
                self._yolo = YOLO("yolov8n.pt")
            except Exception:
                pass
        return self._yolo

    def _load_midas(self):
        if self._midas is None and TORCH_AVAILABLE:
            try:
                self._midas = torch.hub.load(
                    "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
                )
                self._midas.eval()
                if torch.cuda.is_available():
                    self._midas = self._midas.cuda()
                self._midas_transform = torch.hub.load(
                    "intel-isl/MiDaS", "transforms", trust_repo=True
                ).small_transform
            except Exception:
                self._midas = None
        return self._midas

    # ------------------------------------------------------------------
    # Processing stages
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_snr(gray: np.ndarray) -> Tuple[float, float]:
        """Return (snr_db, noise_sigma) from a grayscale float32 image."""
        sigma = robust_sigma_mad(gray)
        signal_power = float(np.mean(gray ** 2))
        snr_db = 10.0 * np.log10(signal_power / (sigma ** 2 + 1e-12))
        return float(snr_db), float(sigma)

    @staticmethod
    def dog_filter(gray: np.ndarray, s1: float, s2: float) -> np.ndarray:
        """Difference-of-Gaussians band-pass filter."""
        return (gaussian_filter(gray, s1) - gaussian_filter(gray, s2)).astype(np.float32)

    @staticmethod
    def detect_geometric_shapes(edge_map: np.ndarray) -> List[Dict]:
        """Detect geometric shapes from edge-map contours."""
        edges_u8 = edge_map.astype(np.uint8) * 255
        contours, _ = cv2.findContours(edges_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri < 1:
                continue
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            n = len(approx)
            circularity = 4 * np.pi * area / (peri * peri + 1e-6)
            if n == 3:
                label = "Triangle"
            elif n == 4:
                ar = w / (h + 1e-6)
                label = "Square" if 0.85 < ar < 1.15 else "Rectangle"
            elif n == 5:
                label = "Pentagon"
            elif n == 6:
                label = "Hexagon"
            elif n > 6:
                label = "Circle" if circularity > 0.65 else f"Polygon({n})"
            else:
                label = "Shape"
            shapes.append({
                "label": label,
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": float(area),
                "vertices": n,
                "confidence": round(min(1.0, area / 2000 + 0.3), 2),
            })
        return shapes

    def detect_objects_yolo(self, frame_bgr: np.ndarray) -> List[Dict]:
        """Run YOLO object detection."""
        model = self._load_yolo()
        if model is None:
            return []
        results = model(frame_bgr, verbose=False)
        objects = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                objects.append({
                    "label": model.names[int(box.cls[0])],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "confidence": round(float(box.conf[0]), 2),
                    "class_id": int(box.cls[0]),
                })
        return objects

    def estimate_depth(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Monocular depth estimation via MiDaS."""
        model = self._load_midas()
        if model is None or self._midas_transform is None:
            return None
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inp = self._midas_transform(rgb)
            if TORCH_AVAILABLE and torch.cuda.is_available():
                inp = inp.cuda()
            with torch.no_grad():
                pred = model(inp)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=rgb.shape[:2],
                    mode="bicubic", align_corners=False
                ).squeeze()
            d = pred.cpu().numpy()
            d = (d - d.min()) / (d.max() - d.min() + 1e-6)
            return d.astype(np.float32)
        except Exception:
            return None

    @staticmethod
    def edge_guided_denoise(gray_u8: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Bilateral denoise preserving edge regions."""
        blurred = cv2.bilateralFilter(gray_u8, 9, 75, 75)
        out = gray_u8.copy()
        mask = edges.astype(bool)
        out[~mask] = blurred[~mask]
        return out

    @staticmethod
    def local_contrast(gray_u8: np.ndarray) -> np.ndarray:
        """CLAHE contrast enhancement."""
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray_u8)

    @staticmethod
    def pseudocolor_map(gray_u8: np.ndarray) -> np.ndarray:
        """Apply TURBO pseudo-color mapping."""
        return cv2.applyColorMap(gray_u8, cv2.COLORMAP_TURBO)

    @staticmethod
    def prioritize_targets(detections: List[Dict],
                           depth: Optional[np.ndarray] = None,
                           frame_shape: Tuple[int, ...] = (480, 640)) -> List[Dict]:
        """Score and rank detected objects by proximity, size, and confidence."""
        h, w = frame_shape[:2]
        frame_area = h * w
        for det in detections:
            x, y, bw, bh = det["bbox"]
            size_ratio = (bw * bh) / (frame_area + 1e-6)
            depth_score = 0.5
            if depth is not None:
                y1c = max(0, y)
                y2c = min(h, y + bh)
                x1c = max(0, x)
                x2c = min(w, x + bw)
                if y2c > y1c and x2c > x1c:
                    depth_score = 1.0 - float(np.mean(depth[y1c:y2c, x1c:x2c]))
            det["priority"] = round(
                0.4 * depth_score + 0.3 * min(size_ratio * 10, 1.0) + 0.3 * det.get("confidence", 0.5),
                3,
            )
        detections.sort(key=lambda d: d.get("priority", 0), reverse=True)
        return detections

    # ------------------------------------------------------------------
    # Main processing entry point
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame through the full PhotonEdge pipeline.

        Returns a dict with all output images, metrics, detections, and alerts.
        """
        t0 = time.perf_counter()
        p = self.params
        res: Dict[str, Any] = {"mode": self.mode}

        # -- Grayscale --
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            gray = frame.astype(np.float32)
            if gray.max() > 1.0:
                gray /= 255.0
        h, w = gray.shape
        res["frame_size"] = [w, h]

        gray_u8 = np.clip(gray * 255, 0, 255).astype(np.uint8)

        # -- SNR estimation --
        snr_db, noise_sigma = self.estimate_snr(gray)
        res["snr_db"] = round(snr_db, 1)
        res["noise_sigma"] = round(noise_sigma, 6)

        # -- Adaptive threshold --
        edge_t = adaptive_threshold(
            gray,
            slope=0.08,
            intercept=float(p.get("edge_t", 2.2)),
            t_min=1.5,
            t_max=6.0,
        )
        res["edge_threshold"] = round(edge_t, 2)

        # -- Multi-scale DoG edge detection --
        scales = p.get("scales", [[0.8, 1.6], [1.2, 2.4], [2.0, 4.0]])
        edge_maps: List[np.ndarray] = []
        dog_maps: List[np.ndarray] = []
        smooth = float(p.get("smooth_sigma", 0.9))

        for s1, s2 in scales:
            dog = self.dog_filter(gray, float(s1), float(s2))
            dog_maps.append(dog)
            em = edges_strict_zero_cross(dog, edge_t=edge_t, smooth_sigma=smooth, closing=True)
            edge_maps.append(em)

        res["edge_counts"] = [int(em.sum()) for em in edge_maps]

        # -- Fusion --
        fusion_mode = p.get("fusion_mode", "v2")
        cov_r = int(p.get("coverage_radius", 3))
        if len(edge_maps) >= 3 and fusion_mode == "v2":
            fused = fuse_backbone_with_fine(
                edge_maps[0], edge_maps[1], edge_maps[2],
                coverage_dilation_px=cov_r, closing=True,
            )
        elif fusion_mode == "or":
            fused = fuse_or(edge_maps)
        else:
            fused = edge_maps[0] if edge_maps else np.zeros((h, w), dtype=bool)

        res["fused_edge_count"] = int(fused.sum())

        # -- Enhancement pipeline --
        if p.get("denoise"):
            enhanced_u8 = self.edge_guided_denoise(gray_u8, fused)
        else:
            enhanced_u8 = gray_u8.copy()

        if p.get("contrast"):
            enhanced_u8 = self.local_contrast(enhanced_u8)

        # -- Shape detection --
        shapes: List[Dict] = []
        if p.get("detect_shapes"):
            shapes = self.detect_geometric_shapes(fused)
        res["shapes"] = shapes

        # -- Object detection --
        objects: List[Dict] = []
        if p.get("detect_objects") and len(frame.shape) == 3:
            objects = self.detect_objects_yolo(frame)
        res["objects"] = objects

        # -- Depth estimation --
        depth: Optional[np.ndarray] = None
        if p.get("depth") and len(frame.shape) == 3:
            depth = self.estimate_depth(frame)

        # -- Target prioritization --
        all_dets = shapes + objects
        if all_dets:
            all_dets = self.prioritize_targets(all_dets, depth, gray.shape)
        res["detections"] = all_dets

        # -- Alert --
        res["alert"] = None
        if all_dets and all_dets[0].get("priority", 0) > 0.6:
            res["alert"] = {
                "target": all_dets[0]["label"],
                "priority": round(all_dets[0]["priority"], 2),
            }

        # ---------------------------------------------------------------
        # Generate output images
        # ---------------------------------------------------------------

        # Per-scale edge images
        res["edge_images"] = [em.astype(np.uint8) * 255 for em in edge_maps]

        # Fused edges
        res["fused_edges"] = fused.astype(np.uint8) * 255

        # DoG response (first scale, normalized)
        if dog_maps:
            d0 = dog_maps[0]
            d0n = ((d0 - d0.min()) / (d0.max() - d0.min() + 1e-6) * 255).astype(np.uint8)
            res["dog_response"] = d0n

        # Enhanced
        res["enhanced"] = enhanced_u8

        # Overlay: edges + detections on original
        if len(frame.shape) == 3:
            overlay = frame.copy()
        else:
            overlay = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

        # Draw fused edges as cyan
        edge_layer = np.zeros_like(overlay)
        edge_layer[fused] = [0, 255, 255]
        overlay = cv2.addWeighted(overlay, 0.7, edge_layer, 0.3, 0)
        overlay[fused] = [0, 255, 255]

        # Draw shape detections (green)
        for sh in shapes:
            x, y, bw, bh = sh["bbox"]
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(overlay, sh["label"], (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw YOLO detections (blue)
        for obj in objects:
            x, y, bw, bh = obj["bbox"]
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (255, 100, 0), 2)
            cv2.putText(overlay, f"{obj['label']} {obj['confidence']:.0%}",
                        (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

        # Alert banner
        if res["alert"]:
            cv2.putText(overlay, f"ALERT: {res['alert']['target']}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        res["overlay"] = overlay

        # Pseudocolor
        if p.get("pseudocolor"):
            res["pseudocolor"] = self.pseudocolor_map(enhanced_u8)

        # Depth map visualization
        if depth is not None:
            d_u8 = (depth * 255).astype(np.uint8)
            res["depth_map"] = cv2.applyColorMap(d_u8, cv2.COLORMAP_MAGMA)

        # Timing
        res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return res

    # ------------------------------------------------------------------
    # Demo shape processing (optical simulation)
    # ------------------------------------------------------------------

    def process_demo_shape(self, shape_name: str, snr_db: float = 25.0) -> Dict[str, Any]:
        """Process a synthetic test shape through the full optical simulation."""
        from core.optics import dog_kernel_embedded, optical_sim_linear

        if shape_name not in SHAPES:
            shape_name = "circle_square"

        size = 256
        img = SHAPES[shape_name](size)
        gt = gt_edges_from_binary(img)

        kernel = dog_kernel_embedded(size, 21, 1.0, 2.0)
        rng = np.random.default_rng(42)
        Y = optical_sim_linear(img, kernel, snr_db, 0.05, rng)

        edges = edges_strict_zero_cross(Y, edge_t=2.2, smooth_sigma=0.9)
        metrics = edge_metrics_symmetric(edges, gt, tol_px=2)

        # Convert to BGR frame so `process` can handle it
        img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        frame = cv2.resize(img_bgr, (640, 480), interpolation=cv2.INTER_NEAREST)

        results = self.process(frame)

        # Add simulation-specific results
        results["gt_edges"] = gt.astype(np.uint8) * 255
        results["sim_edges"] = edges.astype(np.uint8) * 255
        results["sim_input"] = (img * 255).astype(np.uint8)
        results["sim_detector"] = ((Y - Y.min()) / (Y.max() - Y.min() + 1e-6) * 255).astype(np.uint8)
        results["sim_metrics"] = {
            "precision": round(metrics["p"], 3),
            "recall": round(metrics["r"], 3),
            "f1": round(metrics["f1"], 3),
        }
        results["shape_name"] = shape_name
        results["snr_db_sim"] = snr_db
        return results
