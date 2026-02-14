# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.

"""PhotonEdge Web Application Server.

Flask + Flask-SocketIO backend that exposes the full PhotonEdge pipeline
via REST endpoints (image upload, demo shapes) and WebSocket (real-time
webcam processing).
"""

import sys
import base64
import time
from pathlib import Path

import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

# Bootstrap
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from webapp.pipeline import PhotonEdgePipeline, MODE_PRESETS, YOLO_AVAILABLE, TORCH_AVAILABLE
from webapp.tracker import MultiObjectTracker
from core.shapes import SHAPES

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = "photonedge-prototype-2026"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

pipeline = PhotonEdgePipeline("edge_detection")
tracker = MultiObjectTracker()

_fps = {"t": time.time(), "n": 0, "val": 0.0}


def _encode(img, quality: int = 85) -> str:
    """Encode a numpy image to base64 JPEG."""
    if img is None:
        return ""
    if len(img.shape) == 2:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("ascii")


# ---------------------------------------------------------------------------
# HTML route
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.route("/api/capabilities")
def capabilities():
    cuda = False
    if TORCH_AVAILABLE:
        import torch
        cuda = torch.cuda.is_available()
    return jsonify({
        "yolo": YOLO_AVAILABLE,
        "torch": TORCH_AVAILABLE,
        "cuda": cuda,
        "modes": {k: {"name": v["name"], "description": v["description"], "icon": v["icon"]}
                  for k, v in MODE_PRESETS.items()},
        "shapes": list(SHAPES.keys()),
    })


@app.route("/api/process", methods=["POST"])
def process_image():
    """Process an uploaded image through the pipeline."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    raw = request.files["image"].read()
    arr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    # Limit size
    if max(frame.shape[:2]) > 1024:
        scale = 1024 / max(frame.shape[:2])
        frame = cv2.resize(frame, None, fx=scale, fy=scale)

    results = pipeline.process(frame)

    if pipeline.params.get("tracking"):
        tracked = tracker.update(results.get("detections", []))
        results["tracked"] = tracked

    return jsonify(_build_response(results))


@app.route("/api/demo/<shape_name>")
def process_demo(shape_name):
    snr = request.args.get("snr", 25.0, type=float)
    results = pipeline.process_demo_shape(shape_name, snr)
    resp = _build_response(results)
    # extra sim-specific fields
    if "sim_metrics" in results:
        resp["sim_metrics"] = results["sim_metrics"]
    if "gt_edges" in results:
        resp["images"]["gt_edges"] = _encode(results["gt_edges"])
    if "sim_edges" in results:
        resp["images"]["sim_edges"] = _encode(results["sim_edges"])
    if "sim_input" in results:
        resp["images"]["sim_input"] = _encode(results["sim_input"])
    if "sim_detector" in results:
        resp["images"]["sim_detector"] = _encode(results["sim_detector"])
    resp["shape_name"] = results.get("shape_name", shape_name)
    resp["snr_db_sim"] = results.get("snr_db_sim", snr)
    return jsonify(resp)


# ---------------------------------------------------------------------------
# WebSocket events
# ---------------------------------------------------------------------------

@socketio.on("connect")
def on_connect():
    emit("status", {"msg": "Connected to PhotonEdge server", "mode": pipeline.mode})


@socketio.on("set_mode")
def on_set_mode(data):
    mode = data.get("mode", "edge_detection")
    pipeline.set_mode(mode)
    tracker.reset()
    emit("mode_changed", {"mode": mode, "params": _safe_params()})


@socketio.on("update_params")
def on_update_params(data):
    pipeline.update_params(**data)
    emit("params_updated", {"params": _safe_params()})


@socketio.on("frame")
def on_frame(data):
    """Process a webcam frame received via WebSocket."""
    global _fps
    img_b64 = data.get("image", "")
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]
    raw = base64.b64decode(img_b64)
    arr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return

    frame = cv2.resize(frame, (640, 480))
    results = pipeline.process(frame)

    tracked = []
    if pipeline.params.get("tracking"):
        tracked = tracker.update(results.get("detections", []))

    # FPS
    _fps["n"] += 1
    now = time.time()
    if now - _fps["t"] >= 1.0:
        _fps["val"] = _fps["n"] / (now - _fps["t"])
        _fps["n"] = 0
        _fps["t"] = now

    resp = {
        "images": {
            "overlay": _encode(results["overlay"], 75),
            "fused_edges": _encode(results["fused_edges"], 75),
        },
        "metrics": {
            "snr_db": results["snr_db"],
            "edge_threshold": results["edge_threshold"],
            "fused_edge_count": results["fused_edge_count"],
            "latency_ms": results["latency_ms"],
            "fps": round(_fps["val"], 1),
        },
        "shapes": results["shapes"][:8],
        "objects": results.get("objects", [])[:8],
        "tracked": tracked[:10],
        "alert": results["alert"],
    }
    emit("result", resp)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_response(results: dict) -> dict:
    resp: dict = {
        "metrics": {
            "snr_db": results.get("snr_db"),
            "noise_sigma": results.get("noise_sigma"),
            "edge_threshold": results.get("edge_threshold"),
            "edge_counts": results.get("edge_counts"),
            "fused_edge_count": results.get("fused_edge_count"),
            "latency_ms": results.get("latency_ms"),
            "frame_size": results.get("frame_size"),
        },
        "shapes": results.get("shapes", []),
        "objects": results.get("objects", []),
        "detections": results.get("detections", []),
        "tracked": results.get("tracked", []),
        "alert": results.get("alert"),
        "images": {
            "overlay": _encode(results.get("overlay")),
            "fused_edges": _encode(results.get("fused_edges")),
            "enhanced": _encode(results.get("enhanced")),
        },
    }
    for i, ei in enumerate(results.get("edge_images", [])):
        resp["images"][f"edges_scale_{i}"] = _encode(ei)
    if "dog_response" in results:
        resp["images"]["dog_response"] = _encode(results["dog_response"])
    if "pseudocolor" in results:
        resp["images"]["pseudocolor"] = _encode(results["pseudocolor"])
    if "depth_map" in results:
        resp["images"]["depth_map"] = _encode(results["depth_map"])
    return resp


def _safe_params() -> dict:
    """Return params dict with only JSON-serializable values."""
    p = dict(pipeline.params)
    # scales are lists of lists, already safe
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PhotonEdge Web Application")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print()
    print("=" * 56)
    print("  PhotonEdge Web Application")
    print(f"  http://{args.host}:{args.port}")
    print("=" * 56)
    print()
    print(f"  YOLO available : {YOLO_AVAILABLE}")
    print(f"  Torch available: {TORCH_AVAILABLE}")
    print()

    socketio.run(app, host=args.host, port=args.port, debug=args.debug,
                 allow_unsafe_werkzeug=True)
