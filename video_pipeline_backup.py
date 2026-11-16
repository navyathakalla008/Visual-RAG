"""Video understanding pipeline: sample frames, scenes, tracking, ASR, index."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections import defaultdict

import cv2
import numpy as np
import torch


VIDEO_UPLOAD_DIR = os.path.join("uploads", "videos")
VIDEO_FRAME_DIR = os.path.join("uploads", "video_frames")
VIDEO_INDEX_DIR = os.path.join("uploads", "video_index")
PIPELINE_VERSION = 9

ACTION_LABELS = [
    "talking on the phone",
    "using a phone",
    "holding a phone",
    "walking",
    "running",
    "sitting",
    "standing",
    "waving",
    "eating",
    "drinking",
    "cooking",
    "driving",
    "playing sports",
    "dancing",
    "reading",
    "writing",
    "using a laptop",
    "riding a bicycle",
    "talking",
    "looking at a screen",
]

PHONE_OBJECTS = {"cell phone"}
VEHICLE_OBJECTS = {"car", "bus", "truck", "motorcycle"}
RIDE_OBJECTS = {"bicycle", "motorcycle", "skateboard"}
FOOD_OBJECTS = {
    "banana", "apple", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "sandwich",
}
DRINK_OBJECTS = {"bottle", "cup", "wine glass"}
SCREEN_OBJECTS = {"tv", "laptop", "cell phone"}
SPORT_OBJECTS = {
    "sports ball", "tennis racket", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "skis", "snowboard",
}
COOK_OBJECTS = {"oven", "microwave", "sink", "refrigerator", "bowl", "spoon", "fork", "knife"}
READ_OBJECTS = {"book"}
WRITE_OBJECTS = {"book", "laptop"}
LAPTOP_OBJECTS = {"laptop"}

# YOLO classes that often fire once on the wrong texture.
FLASH_FALSE_POSITIVES = {
    "frisbee", "kite", "surfboard", "skateboard", "snowboard",
    "baseball bat", "baseball glove", "tennis racket", "sports ball",
    "parking meter", "fire hydrant", "stop sign",
    "toothbrush", "hair drier", "toaster", "mouse", "remote",
    "train", "airplane", "boat", "bus", "truck", "car",
    "traffic light", "bench", "parking meter",
}
SMALL_OK_OBJECTS = {
    "cell phone", "book", "bottle", "cup", "wine glass", "fork",
    "knife", "spoon", "sports ball", "remote", "mouse", "clock",
    "toothbrush", "tie",
}
ACTION_NEAR_OBJECTS = {
    "using a phone": PHONE_OBJECTS,
    "talking on the phone": PHONE_OBJECTS,
    "holding a phone": PHONE_OBJECTS,
    "eating": FOOD_OBJECTS,
    "drinking": DRINK_OBJECTS,
    "reading": READ_OBJECTS,
    "writing": WRITE_OBJECTS,
    "using a laptop": LAPTOP_OBJECTS,
    "looking at a screen": SCREEN_OBJECTS,
    "riding a bicycle": {"bicycle", "motorcycle"},
    "playing sports": SPORT_OBJECTS,
    "cooking": {"oven", "microwave"},
    "driving": VEHICLE_OBJECTS,
}
CLASS_MIN_CONF = {
    "person": 0.22,
    "cell phone": 0.30,
    "book": 0.32,
    "laptop": 0.32,
    "tv": 0.32,
    "chair": 0.32,
    "bottle": 0.32,
    "cup": 0.32,
}

SCENE_TYPE_LABELS = [
    "indoor room",
    "living room",
    "bedroom",
    "kitchen",
    "office",
    "classroom",
    "restaurant",
    "street",
    "park",
    "market",
    "shop",
    "beach",
    "playground",
    "stadium",
    "forest",
    "city outdoor",
    "close-up of a person",
    "selfie",
]

PEOPLE_TYPE_LABELS = [
    "man",
    "woman",
    "boy",
    "girl",
    "child",
    "baby",
]

CLOTHING_LABELS = [
    "shirt",
    "t-shirt",
    "jeans",
    "pants",
    "dress",
    "skirt",
    "shorts",
    "jacket",
    "coat",
    "hat",
    "cap",
    "shoes",
    "saree",
    "hoodie",
    "sweater",
    "uniform",
]

_whisper_model = None
_whisper_error = None


def ensure_video_dirs():
    for path in (VIDEO_UPLOAD_DIR, VIDEO_FRAME_DIR, VIDEO_INDEX_DIR):
        os.makedirs(path, exist_ok=True)


def format_timestamp(seconds):
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_interval(start, end):
    return f"{format_timestamp(start)}–{format_timestamp(end)}"


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def video_index_path(video_id):
    return os.path.join(VIDEO_INDEX_DIR, f"{video_id}.json")


def video_embedding_path(video_id):
    return os.path.join(VIDEO_INDEX_DIR, f"{video_id}_embeddings.pt")


def original_video_name(filename):
    if not filename:
        return "video"
    base = os.path.basename(filename)
    prefix, sep, rest = base.partition("_")
    if sep and len(prefix) == 12 and all(c in "0123456789abcdef" for c in prefix):
        return rest or base
    return base


def load_video_index(video_id):
    path = video_index_path(video_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def list_video_indexes():
    ensure_video_dirs()
    items = []
    for name in os.listdir(VIDEO_INDEX_DIR):
        if not name.endswith(".json") or name.endswith("_embeddings.json"):
            continue
        path = os.path.join(VIDEO_INDEX_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                index = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        video_id = index.get("video_id")
        if not video_id:
            continue
        items.append(
            {
                "video_id": video_id,
                "filename": index.get("filename") or "",
                "original_name": index.get("original_name")
                or original_video_name(index.get("filename")),
                "duration": float(index.get("duration") or 0.0),
                "scenes": len(index.get("scenes") or []),
                "sampled_frames": index.get("sampled_frames") or 0,
                "has_transcript": bool(index.get("full_transcript")),
                "asr_error": index.get("asr_error"),
                "mtime": os.path.getmtime(path),
            }
        )
    items.sort(key=lambda item: item["mtime"], reverse=True)
    return items


def find_matching_index(original_name, file_size):
    """Reuse a finished analysis of the same uploaded file."""
    if not original_name or not file_size:
        return None
    wanted = os.path.basename(original_name).lower()
    for item in list_video_indexes():
        stored = os.path.join(VIDEO_UPLOAD_DIR, item["filename"])
        if not os.path.isfile(stored):
            continue
        if os.path.getsize(stored) != file_size:
            continue
        existing_name = (item.get("original_name") or original_video_name(item["filename"])).lower()
        if existing_name == wanted:
            index = load_video_index(item["video_id"])
            if (index or {}).get("pipeline_version") != PIPELINE_VERSION:
                continue
            return index
    return None


def save_video_index(index):
    ensure_video_dirs()
    path = video_index_path(index["video_id"])
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(index), handle, indent=2)
    return path


def load_scene_embeddings(video_id, device="cpu"):
    path = video_embedding_path(video_id)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=device)


def _histogram(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _open_video(video_path):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 25.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if frame_count > 0 else 0.0
    return capture, fps, frame_count, width, height, duration


def _adaptive_sample_rate(duration):
    duration = float(duration or 0.0)
    if duration <= 20:
        return 5.0, 250
    if duration <= 90:
        return 3.0, 400
    if duration <= 300:
        return 2.0, 700
    return 1.25, 1000


def sample_frames(video_path, sample_fps=1.5, max_frames=720):
    """Read the video at a low sample rate instead of every native frame."""
    capture, fps, frame_count, width, height, duration = _open_video(video_path)
    sample_fps = max(0.25, float(sample_fps or 1.5))
    max_frames = max(1, int(max_frames))
    if duration > 0:
        sample_fps = min(sample_fps, max_frames / duration)
    step = max(1, int(round(fps / max(0.25, sample_fps))))
    samples = []
    index = 0
    kept = 0

    while True:
        if index % step != 0:
            if not capture.grab():
                break
            index += 1
            continue
        ok, frame = capture.read()
        if not ok:
            break
        time_sec = index / fps
        samples.append(
            {
                "index": index,
                "time": time_sec,
                "frame": frame,
            }
        )
        kept += 1
        if kept >= max_frames:
            break
        index += 1

    capture.release()
    meta = {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration or (samples[-1]["time"] if samples else 0.0),
        "sample_fps": sample_fps,
        "sampled_frames": len(samples),
    }
    return samples, meta


def _scenes_from_cuts(cuts, duration):
    scenes = []
    for i, start in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else duration
        if end <= start + 0.05:
            continue
        scenes.append(
            {
                "id": i + 1,
                "start": float(start),
                "end": float(end),
                "duration": float(end - start),
            }
        )
    if not scenes:
        scenes.append(
            {
                "id": 1,
                "start": 0.0,
                "end": float(max(duration, 0.1)),
                "duration": float(max(duration, 0.1)),
            }
        )
    for i, scene in enumerate(scenes):
        scene["id"] = i + 1
    return scenes


def _merge_short_scenes(scenes, min_len):
    if len(scenes) <= 1:
        return scenes
    merged = [dict(scenes[0])]
    for scene in scenes[1:]:
        last = merged[-1]
        if last["duration"] < min_len or scene["duration"] < min_len:
            last["end"] = scene["end"]
            last["duration"] = last["end"] - last["start"]
        else:
            merged.append(dict(scene))
    for i, scene in enumerate(merged):
        scene["id"] = i + 1
    return merged


def detect_scenes_from_video(video_path, duration=None, progress=None):
    """Detect visual shot boundaries with adaptive multi-cue scoring.

    This function only creates scene start/end timestamps. It does not alter
    frame sampling, tracking, action recognition, transcription, or indexing.
    """
    capture = None
    try:
        capture, fps, frame_count, width, height, file_duration = _open_video(video_path)
        duration = float(duration or file_duration or 0.0)
        if duration <= 0:
            raise ValueError("Could not determine the video duration.")

        # Dense enough to catch quick edits while remaining practical on CPU.
        # Long clips need a coarser scan: decoding every 0.25 seconds on CPU
        # can take a long time. 0.5s still detects ordinary shot changes while
        # limiting work on videos longer than ten minutes.
        interval = 0.12 if duration <= 120 else (0.18 if duration <= 600 else 0.5)
        step = max(1, int(round(fps * interval)))
        times, hists, grays = [], [], []
        index = 0
        last_report_index = -report_every if frame_count else 0
        while index < frame_count:
            if progress and index - last_report_index >= report_every:
                pct = min(100, int(index * 100 / max(frame_count, 1)))
                _progress(progress, f"Step 2/5: Scanning video for scene changes ({pct}%)...")
                last_report_index = index
            if index % step:
                if not capture.grab():
                    break
                index += 1
                continue
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = index / fps
            if timestamp >= duration:
                break
            small = cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            # Blur slightly to reduce false cuts caused by noise and compression.
            gray = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32) / 255.0
            times.append(float(timestamp))
            hists.append(hist)
            grays.append(gray)
            index += 1

        capture.release()
        capture = None
        if len(times) < 2:
            return _scenes_from_cuts([0.0], duration)

        # Combine color-distribution change, pixel change, and edge-layout change.
        hist_change, pixel_change, edge_change = [], [], []
        total_pairs = max(1, len(hists) - 1)
        last_score_pct = -10
        for i in range(1, len(hists)):
            if progress:
                pct = int(i * 100 / total_pairs)
                if pct >= last_score_pct + 10:
                    _progress(progress, f"Step 2/5: Scoring scene changes ({pct}%)...")
                    last_score_pct = pct
            hist_change.append(float(cv2.compareHist(hists[i-1], hists[i], cv2.HISTCMP_BHATTACHARYYA)))
            pixel_change.append(float(np.mean(np.abs(grays[i] - grays[i-1]))))
            e0 = cv2.Canny((grays[i-1] * 255).astype(np.uint8), 60, 140)
            e1 = cv2.Canny((grays[i] * 255).astype(np.uint8), 60, 140)
            edge_change.append(float(np.mean(e0 != e1)))

        hist_change = np.asarray(hist_change, dtype=np.float32)
        pixel_change = np.asarray(pixel_change, dtype=np.float32)
        edge_change = np.asarray(edge_change, dtype=np.float32)
        # Normalize cues robustly so a single noisy frame does not set the scale.
        def robust_norm(values):
            med = float(np.median(values))
            mad = float(np.median(np.abs(values - med))) + 1e-6
            return np.clip((values - med) / (4.0 * mad), 0.0, 1.0)
        score = (0.50 * robust_norm(hist_change) +
                 0.35 * robust_norm(pixel_change) +
                 0.15 * robust_norm(edge_change))

        # A cut must be a local peak and show a meaningful change in at least
        # one strong cue. Local adaptive thresholds handle both quiet and busy clips.
        cuts = [0.0]
        min_cut_spacing = max(0.50, interval * 2.0)
        for i in range(len(score)):
            left = max(0, i - 15)
            right = min(len(score), i + 16)
            local = score[left:right]
            med = float(np.median(local)) if len(local) else 0.0
            mad = float(np.median(np.abs(local - med))) + 1e-6
            threshold = max(0.28, med + 3.0 * mad)
            local_peak = (i == 0 or score[i] >= score[i-1]) and (i == len(score)-1 or score[i] >= score[i+1])
            strong_color = hist_change[i] >= 0.30
            strong_pixels = pixel_change[i] >= 0.16
            strong_combined = score[i] >= threshold and (hist_change[i] >= 0.12 or pixel_change[i] >= 0.08)
            if not local_peak or not (strong_color or strong_pixels or strong_combined):
                continue
            timestamp = times[i + 1]
            if timestamp - cuts[-1] < min_cut_spacing:
                # Keep the stronger of two nearby candidates rather than always
                # accepting the first one.
                if len(cuts) > 1 and score[i] > getattr(detect_scenes_from_video, "_last_score", -1):
                    cuts[-1] = float(timestamp)
                continue
            cuts.append(float(timestamp))
            detect_scenes_from_video._last_score = float(score[i])

        cuts = sorted(set([0.0] + [min(max(c, 0.0), duration) for c in cuts]))
        scenes = _scenes_from_cuts(cuts, duration)
        if not scenes:
            scenes = [{"id": 1, "start": 0.0, "end": duration, "duration": duration}]
        # Guarantee contiguous, ordered, non-overlapping scene intervals.
        cursor = 0.0
        valid = []
        for scene in scenes:
            start = cursor
            end = min(max(float(scene["end"]), start), duration)
            if end - start > 0.01:
                valid.append({"id": len(valid) + 1, "start": start, "end": end, "duration": end - start})
                cursor = end
        if not valid or cursor < duration:
            if valid:
                valid[-1]["end"] = duration
                valid[-1]["duration"] = duration - valid[-1]["start"]
            else:
                valid = [{"id": 1, "start": 0.0, "end": duration, "duration": duration}]
        return valid
    finally:
        if capture is not None:
            capture.release()

def detect_scenes(samples, duration, min_scene_len=0.6, corr_threshold=0.55):
    """Fallback shot detector operating on frames already sampled by the pipeline."""
    if not samples:
        return _scenes_from_cuts([0.0], max(float(duration or 0.0), 0.1))

    cuts = [0.0]
    previous_hist = _histogram(samples[0]["frame"])
    previous_gray = cv2.resize(
        cv2.cvtColor(samples[0]["frame"], cv2.COLOR_BGR2GRAY),
        (160, 90),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)

    for sample in samples[1:]:
        frame = sample["frame"]
        current_hist = _histogram(frame)
        current_gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (160, 90),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)

        correlation = float(cv2.compareHist(
            previous_hist, current_hist, cv2.HISTCMP_CORREL
        ))
        difference = float(np.mean(np.abs(current_gray - previous_gray)) / 255.0)
        timestamp = float(sample["time"])

        if (
            timestamp - cuts[-1] >= min_scene_len
            and correlation < corr_threshold
            and difference >= 0.08
        ):
            cuts.append(timestamp)

        previous_hist = current_hist
        previous_gray = current_gray

    duration = float(duration or samples[-1]["time"])
    if duration <= 0:
        duration = max(float(samples[-1]["time"]), 0.1)
    cuts = [c for c in cuts if 0 <= c < duration]
    if not cuts or cuts[0] != 0.0:
        cuts.insert(0, 0.0)
    return _scenes_from_cuts(sorted(set(cuts)), duration)


def _scene_for_time(scenes, time_sec):
    for scene in scenes:
        if scene["start"] <= time_sec < scene["end"]:
            return scene
        if abs(time_sec - scene["end"]) < 1e-3:
            return scene
    return scenes[-1] if scenes else None


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _box_center(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _keep_detection(label, conf, hits, scene_frames):
    if label == "person" and conf >= 0.28:
        return True
    if conf >= 0.55:
        return True
    if label in FLASH_FALSE_POSITIVES:
        return hits >= 2 and conf >= 0.40
    min_hits = 2 if scene_frames >= 6 else 1
    return hits >= min_hits or conf >= 0.35


def track_objects(samples, yolo_model, scenes, confidence=0.28):
    """Detect people/objects and associate IDs with IoU across sampled frames."""
    detections_by_time = []
    active = []
    finished = []
    next_id = 1
    max_gap = 3.2

    for sample in samples:
        frame = sample["frame"]
        time_sec = sample["time"]
        try:
            results = yolo_model(frame, conf=confidence, iou=0.50, verbose=False)
        except Exception:
            results = []

        frame_dets = []
        boxes = results[0].boxes if results else None
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                label = yolo_model.names[int(box.cls.item())].lower()
                conf = float(box.conf.item())
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                frame_dets.append(
                    {
                        "label": label,
                        "confidence": conf,
                        "track_id": None,
                        "box": xyxy,
                    }
                )

        unmatched = set(range(len(frame_dets)))
        # Greedy one-to-one association: combine overlap with center/size distance.
        # IoU alone fragments tracks when a person turns, moves quickly, or changes scale.
        candidates = []
        for track_idx, track in enumerate(active):
            gap = time_sec - track["last_time"]
            if gap > max_gap:
                continue
            tx1, ty1, tx2, ty2 = track["box"]
            tcx, tcy = _box_center(track["box"])
            tw, th = max(1.0, tx2 - tx1), max(1.0, ty2 - ty1)
            for det_idx, det in enumerate(frame_dets):
                if det["label"] != track["label"]:
                    continue
                dx1, dy1, dx2, dy2 = det["box"]
                dcx, dcy = _box_center(det["box"])
                dw, dh = max(1.0, dx2 - dx1), max(1.0, dy2 - dy1)
                scale = max((tw * th) ** 0.5, (dw * dh) ** 0.5, 1.0)
                center_dist = (((tcx - dcx) ** 2 + (tcy - dcy) ** 2) ** 0.5) / scale
                iou = _box_iou(track["box"], det["box"])
                # Allow moderate movement; stricter for longer gaps.
                max_center_dist = 1.15 if gap <= 1.0 else 0.70
                if iou >= 0.08 or center_dist <= max_center_dist:
                    score = (1.0 - iou) + 0.45 * center_dist + 0.12 * gap
                    candidates.append((score, track_idx, det_idx))
        used_tracks = set()
        for _score, track_idx, best_i in sorted(candidates):
            if track_idx in used_tracks or best_i not in unmatched:
                continue
            track = active[track_idx]
            if time_sec - track["last_time"] > max_gap:
                continue
            det = frame_dets[best_i]
            det["track_id"] = track["id"]
            track["box"] = det["box"]
            track["last_time"] = time_sec
            track["times"].append(time_sec)
            track["confidences"].append(det["confidence"])
            track["boxes"].append(det["box"])
            unmatched.remove(best_i)
            used_tracks.add(track_idx)
        # Track update is complete; unmatched detections start new tracks.
        for track in active:
            pass
        
        still_active = []
        for track in active:
            if time_sec - track["last_time"] <= max_gap:
                still_active.append(track)
            else:
                finished.append(track)
        active = still_active

        for i in unmatched:
            det = frame_dets[i]
            track = {
                "id": next_id,
                "label": det["label"],
                "box": det["box"],
                "last_time": time_sec,
                "times": [time_sec],
                "confidences": [det["confidence"]],
                "boxes": [det["box"]],
            }
            det["track_id"] = next_id
            next_id += 1
            active.append(track)

        detections_by_time.append({"time": time_sec, "detections": frame_dets})

    finished.extend(active)

    video_duration = 0.0
    if samples:
        video_duration = float(samples[-1]["time"])
    short_clip = video_duration <= 20.0
    track_list = []
    for info in finished:
        times = info["times"]
        confs = info["confidences"]
        if not times:
            continue
        mean_conf = float(np.mean(confs))
        hits = len(times)
        label = info["label"]
        start, end = min(times), max(times)
        duration = end - start
        if label in FLASH_FALSE_POSITIVES and (hits < 2 or mean_conf < 0.45):
            continue
        if hits == 1 and not short_clip and mean_conf < 0.50:
            continue
        if hits == 1 and mean_conf < 0.38:
            continue
        pad = 0.35 if hits == 1 else 0.0
        track_list.append(
            {
                "track_id": int(info["id"]),
                "label": label,
                "start": float(start),
                "end": float(end + pad),
                "duration": float(max(duration, pad)),
                "confidence": mean_conf,
                "hits": hits,
                "scene_ids": sorted(
                    {
                        _scene_for_time(scenes, t)["id"]
                        for t in times
                        if _scene_for_time(scenes, t)
                    }
                ),
            }
        )
    track_list.sort(key=lambda t: (t["start"], -t["confidence"]))
    return detections_by_time, track_list


def _merge_intervals(items, gap=1.6):
    if not items:
        return []
    ordered = sorted(items, key=lambda x: x["start"])
    merged = [dict(ordered[0])]
    for item in ordered[1:]:
        last = merged[-1]
        same = last.get("label") == item.get("label")
        if same and item["start"] <= last["end"] + gap:
            last["end"] = max(last["end"], item["end"])
            last["duration"] = last["end"] - last["start"]
        else:
            merged.append(dict(item))
    return merged


def _visible(pt):
    return np.isfinite(pt[0]) and np.isfinite(pt[1]) and (pt[0] > 1 or pt[1] > 1)


def _pose_actions(pil, pose_model):
    """Pose labels only when enough joints are visible to tell sit vs stand."""
    labels = []
    results = pose_model(pil, conf=0.40, verbose=False)
    if not results or results[0].keypoints is None:
        return labels
    keypoints = results[0].keypoints.xy
    if keypoints is None:
        return labels
    height = float(pil.height)
    for person_points in keypoints:
        pts = person_points.cpu().numpy()
        if pts.shape[0] < 17:
            continue
        ls, rs = pts[5], pts[6]
        lh, rh = pts[11], pts[12]
        lk, rk = pts[13], pts[14]
        la, ra = pts[15], pts[16]
        lw, rw = pts[9], pts[10]
        hip_y = np.nanmean([lh[1], rh[1]])
        knee_y = np.nanmean([lk[1], rk[1]])
        ankle_y = np.nanmean([la[1], ra[1]])
        shoulder_y = np.nanmean([ls[1], rs[1]])
        if not np.isfinite(hip_y) or not np.isfinite(shoulder_y):
            continue
        torso = abs(hip_y - shoulder_y)
        if torso < 8:
            continue
        hips_ok = _visible(lh) and _visible(rh)
        knees_ok = _visible(lk) or _visible(rk)
        ankles_ok = _visible(la) or _visible(ra)
        if hips_ok and knees_ok and ankles_ok:
            thigh = abs(knee_y - hip_y)
            shin = abs(ankle_y - knee_y)
            # Sitting: thighs not extended and ankles not far below knees.
            if thigh < 0.55 * torso and shin < 0.85 * torso:
                labels.append("sitting")
            elif thigh >= 0.55 * torso and ankle_y > knee_y > hip_y:
                labels.append("standing")
        elif hips_ok and knees_ok and abs(knee_y - hip_y) >= 0.7 * torso:
            labels.append("standing")

        wrists = [p for p in (lw, rw) if _visible(p)]
        if wrists and min(p[1] for p in wrists) < shoulder_y - 0.15 * torso:
            labels.append("waving")
    return labels


def _labels_in_frame(timed):
    return {d["label"] for d in timed.get("detections") or []}


def _object_near_person(detections, object_labels, expansion=0.20):
    """Return True only when a target object is spatially close to a detected person."""
    people = [d["box"] for d in detections if d.get("label") == "person"]
    targets = [d["box"] for d in detections if d.get("label") in object_labels]
    if not people or not targets:
        return False

    for px1, py1, px2, py2 in people:
        pad_x = max(8.0, (px2 - px1) * expansion)
        pad_y = max(8.0, (py2 - py1) * expansion)
        expanded = (px1 - pad_x, py1 - pad_y, px2 + pad_x, py2 + pad_y)
        for ox1, oy1, ox2, oy2 in targets:
            cx, cy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0
            if expanded[0] <= cx <= expanded[2] and expanded[1] <= cy <= expanded[3]:
                return True
            if _box_iou(expanded, (ox1, oy1, ox2, oy2)) > 0.01:
                return True
    return False


def _person_speed(prev_centers, detections, time_sec, frame_h):
    centers = []
    for det in detections:
        if det["label"] != "person":
            continue
        centers.append(_box_center(det["box"]))
    if not centers or not prev_centers:
        return 0.0, centers
    speeds = []
    used = set()
    for cx, cy in centers:
        best = None
        for i, (px, py) in enumerate(prev_centers):
            if i in used:
                continue
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if best is None or dist < best[0]:
                best = (dist, i)
        if best is not None:
            used.add(best[1])
            speeds.append(best[0] / max(frame_h, 1.0))
    return (max(speeds) if speeds else 0.0), centers


def _ground_action(label, objects):
    objects = set(objects)
    if label in {"talking on the phone", "using a phone", "holding a phone"}:
        return bool(objects & PHONE_OBJECTS)
    if label == "driving":
        return bool(objects & VEHICLE_OBJECTS)
    if label == "riding a bicycle":
        return "bicycle" in objects or "motorcycle" in objects
    if label == "eating":
        return bool(objects & FOOD_OBJECTS)
    if label == "drinking":
        return bool(objects & DRINK_OBJECTS)
    if label == "cooking":
        return bool(objects & COOK_OBJECTS)
    if label == "reading":
        return bool(objects & READ_OBJECTS)
    if label == "writing":
        return bool(objects & WRITE_OBJECTS)
    if label == "using a laptop":
        return bool(objects & LAPTOP_OBJECTS)
    if label == "looking at a screen":
        return bool(objects & SCREEN_OBJECTS)
    if label == "playing sports":
        return bool(objects & SPORT_OBJECTS)
    if label in {"walking", "running", "sitting", "standing", "waving", "dancing", "talking"}:
        return "person" in objects
    return False


def detect_actions(samples, detections_by_time, vision, scenes, tracks=None):
    """Conservative action estimates from person crops and pose evidence.

    CLIP scores candidate actions in visible person crops. A label is retained
    only when it repeats across adjacent sampled frames; pose labels are an
    additional cue, not a substitute for temporal evidence. This is not a
    dedicated video transformer, so uncertain actions are omitted.
    """
    from PIL import Image

    pose_model = vision.get("pose")
    clip_rank_image = vision.get("clip_rank_image")
    get_pil_embedding = vision.get("get_pil_embedding")
    candidates = [
        "dancing", "walking", "running", "sitting", "standing", "waving",
        "playing sports", "cooking", "riding a bicycle", "eating", "drinking",
        "reading", "using a phone", "using a laptop", "talking", "working",
    ]
    per_frame = []
    for idx, (sample, timed) in enumerate(zip(samples, detections_by_time)):
        frame = sample.get("frame")
        if frame is None:
            per_frame.append((float(sample.get("time", 0)), set()))
            continue
        found = set()
        people = [d for d in (timed.get("detections") or []) if d.get("label") == "person"]
        # Only score the most visible people to keep CPU processing practical.
        people.sort(key=lambda d: (d.get("box", [0,0,0,0])[2]-d.get("box", [0,0,0,0])[0]) * (d.get("box", [0,0,0,0])[3]-d.get("box", [0,0,0,0])[1]), reverse=True)
        for det in people[:3]:
            crop = _crop_bgr(frame, det.get("box", [0,0,0,0]), pad=0.12)
            if crop is None or crop.size == 0:
                continue
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if clip_rank_image is not None and get_pil_embedding is not None:
                try:
                    emb = get_pil_embedding(crop_pil)
                    if emb is not None:
                        ranked = clip_rank_image(
                            emb, candidates, "a video frame showing a person {}",
                            threshold=0.34, margin=0.025, top_k=2,
                        )
                        for label in ranked or []:
                            label = str(label).lower().strip()
                            if label in candidates:
                                found.add(label)
                except Exception as exc:
                    print("CLIP action scoring warning:", exc)
            # Pose can support posture/raised-arm labels, but not infer dance alone.
            if pose_model is not None and (idx % 2 == 0):
                try:
                    found.update(_pose_actions(crop_pil, pose_model))
                except Exception as exc:
                    print("Pose action warning:", exc)
        per_frame.append((float(sample.get("time", 0)), found))

    # Temporal consensus: an action must occur in at least two adjacent samples
    # (or three samples for longer clips) to avoid single-frame CLIP guesses.
    if len(per_frame) < 2:
        return []
    min_support = 2 if len(per_frame) < 12 else 3
    hits = []
    for label in candidates + ["waving"]:
        matching = [i for i, (_, labels) in enumerate(per_frame) if label in labels]
        runs = []
        run = []
        for i in matching:
            if run and (i != run[-1] + 1 or per_frame[i][0] - per_frame[run[-1]][0] > 1.6):
                if len(run) >= min_support:
                    runs.append(run)
                run = []
            run.append(i)
        if len(run) >= min_support:
            runs.append(run)
        for run in runs:
            start_t = per_frame[run[0]][0]
            end_t = min(float(samples[-1].get("time", start_t)) + 0.5, per_frame[run[-1]][0] + 0.5)
            scene = _scene_for_time(scenes, start_t)
            hits.append({"label": label, "start": start_t, "end": max(start_t + 0.1, end_t),
                         "duration": max(0.1, end_t - start_t), "scene_id": scene["id"] if scene else None,
                         "evidence": "CLIP person-crop consensus"})
    # Merge adjacent spans of the same label without extending across long gaps.
    return _merge_intervals(hits, gap=0.6)


def _pick_keyframe(samples, scene):
    mid = (scene["start"] + scene["end"]) / 2.0
    return min(samples, key=lambda s: abs(s["time"] - mid))



def _classify_scene_setting(samples, scene, vision):
    """Return a conservative CLIP setting label for a scene, or None if uncertain."""
    try:
        from PIL import Image
        rank = vision.get("clip_rank_image") if vision else None
        embed = vision.get("get_pil_embedding") if vision else None
        if rank is None or embed is None:
            return None
        # Use several frames across the interval so a single keyframe does not decide it.
        start, end = float(scene["start"]), float(scene["end"])
        candidates = [s for s in samples if start <= float(s.get("time", -1)) < end and s.get("frame") is not None]
        if not candidates:
            return None
        targets = [start + (end - start) * p for p in (0.25, 0.5, 0.75)]
        chosen = []
        for target in targets:
            sample = min(candidates, key=lambda item: abs(float(item["time"]) - target))
            if all(sample is not prior for prior in chosen):
                chosen.append(sample)
        labels = []
        for sample in chosen:
            rgb = cv2.cvtColor(sample["frame"], cv2.COLOR_BGR2RGB)
            features = embed(Image.fromarray(rgb))
            if features is None:
                continue
            ranked = rank(features, SCENE_TYPE_LABELS, "a photo of {}", threshold=0.30, margin=0.018, top_k=1)
            if ranked:
                labels.append(str(ranked[0]).strip().lower())
        if not labels:
            return None
        # Require agreement from at least two frames when multiple are available.
        from collections import Counter
        label, count = Counter(labels).most_common(1)[0]
        required = 2 if len(chosen) >= 2 else 1
        return label if count >= required else None
    except Exception as exc:
        print(f"Scene setting classification warning: {exc}")
        return None


def _normalize_scene_timeline(scenes, duration, samples=None, vision=None, progress=None):
    """Normalize intervals, optionally merging only adjacent, confidently same-setting scenes."""
    duration = max(0.0, float(duration or 0.0))
    if duration <= 0:
        raise ValueError("Video duration must be positive to normalize scene timestamps.")
    ordered = sorted((dict(s) for s in (scenes or [])), key=lambda s: float(s.get("start", 0.0)))
    if not ordered:
        ordered = [{"start": 0.0, "end": duration, "duration": duration}]

    # Clamp boundaries to the real duration and discard malformed/zero-length candidates.
    cleaned = []
    for item in ordered:
        start = max(0.0, min(duration, float(item.get("start", 0.0))))
        end = max(0.0, min(duration, float(item.get("end", start))))
        if end <= start:
            continue
        item["start"], item["end"] = start, end
        item["duration"] = end - start
        cleaned.append(item)
    if not cleaned:
        cleaned = [{"start": 0.0, "end": duration, "duration": duration}]

    # Adjacent segments merge only when the semantic classifier confidently agrees.
    if samples and vision and len(cleaned) > 1:
        labels = []
        total_scenes = len(cleaned)
        for idx, item in enumerate(cleaned, start=1):
            if progress and (idx == 1 or idx % 10 == 0 or idx == total_scenes):
                _progress(progress, f"Step 2/5: Comparing scene settings ({idx}/{total_scenes})...")
            labels.append(_classify_scene_setting(samples, item, vision))
        merged = []
        for i, item in enumerate(cleaned):
            if merged and labels[i] is not None and labels[i] == merged[-1].get("_setting_label"):
                merged[-1]["end"] = item["end"]
                merged[-1]["duration"] = merged[-1]["end"] - merged[-1]["start"]
                # Preserve an existing useful metadata field if present; update only interval fields.
            else:
                clone = dict(item)
                clone["_setting_label"] = labels[i]
                merged.append(clone)
        cleaned = merged

    # Rebuild a contiguous timeline from detected boundaries; never invent equal-length scenes.
    boundaries = [0.0]
    for item in cleaned[1:]:
        boundary = float(item["start"])
        if 1e-3 < boundary < duration - 1e-3 and boundary - boundaries[-1] > 1e-3:
            boundaries.append(boundary)
    boundaries.append(duration)
    normalized = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end - start <= 1e-3:
            continue
        # Reuse the metadata belonging to the segment active at this interval's start.
        source = next((s for s in cleaned if s["start"] <= start + 1e-3 < s["end"]), {})
        scene = {k: v for k, v in source.items() if k not in {"_setting_label", "start", "end", "duration", "id"}}
        scene.update({"id": len(normalized) + 1, "start": float(start), "end": float(end), "duration": float(end - start)})
        normalized.append(scene)

    validate_scene_timeline(normalized, duration)
    for scene in normalized:
        print("Scene timeline: id={id} start={start:.3f}s end={end:.3f}s duration={duration:.3f}s".format(**scene))
    return normalized


def validate_scene_timeline(scenes, duration, tolerance=0.002):
    """Raise ValueError if scene timestamps have gaps, overlaps, or invalid durations."""
    duration = float(duration)
    if not scenes:
        raise ValueError("Scene timeline is empty")
    if abs(float(scenes[0]["start"])) > tolerance:
        raise ValueError(f"First scene must start at zero, got {scenes[0]['start']}")
    for i, scene in enumerate(scenes):
        start, end, span = float(scene["start"]), float(scene["end"]), float(scene["duration"])
        if start < -tolerance or end > duration + tolerance or end <= start:
            raise ValueError(f"Invalid interval for scene {scene.get('id')}: {start}–{end}")
        if abs((end - start) - span) > tolerance:
            raise ValueError(f"Duration mismatch for scene {scene.get('id')}")
        if i and abs(float(scenes[i-1]["end"]) - start) > tolerance:
            raise ValueError(f"Timeline gap/overlap between scenes {scenes[i-1].get('id')} and {scene.get('id')}")
        if int(scene.get("id", -1)) != i + 1:
            raise ValueError("Scene IDs must be sequential and chronological")
    if abs(float(scenes[-1]["end"]) - duration) > tolerance:
        raise ValueError(f"Last scene must end at video duration {duration}, got {scenes[-1]['end']}")
    return True

def _crop_bgr(frame, box, pad=0.08):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * bw))
    y1 = max(0, int(y1 - pad * bh))
    x2 = min(w, int(x2 + pad * bw))
    y2 = min(h, int(y2 + pad * bh))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _video_scene_contents(frame, detections, vision):
    """People/scene labels from the keyframe, objects only from YOLO boxes."""
    from PIL import Image

    contents = {}
    clip_rank_image = vision.get("clip_rank_image")
    get_pil_embedding = vision.get("get_pil_embedding")
    if clip_rank_image is None or get_pil_embedding is None:
        return contents

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    try:
        image_features = get_pil_embedding(pil)
        if image_features is not None:
            scene = clip_rank_image(
                image_features,
                SCENE_TYPE_LABELS,
                "a photo of {}",
                threshold=0.30,
                margin=0.018,
                top_k=1,
            )
            if scene:
                contents["Scene"] = [str(scene[0])]
    except Exception:
        pass

    people = []
    clothing = []
    person_dets = [d for d in detections if d["label"] == "person"]
    person_dets.sort(key=lambda d: d.get("confidence", 0), reverse=True)
    for det in person_dets[:4]:
        crop = _crop_bgr(frame, det["box"])
        if crop is None or crop.size == 0:
            continue
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        try:
            crop_features = get_pil_embedding(crop_pil)
            if crop_features is None:
                people.append("person")
                continue
            ranked_people = clip_rank_image(
                crop_features,
                PEOPLE_TYPE_LABELS,
                "a photograph of a {}",
                threshold=0.30,
                margin=0.02,
                top_k=1,
            )
            if ranked_people:
                people.append(str(ranked_people[0]))
            else:
                people.append("person")
            ranked_clothes = clip_rank_image(
                crop_features,
                CLOTHING_LABELS,
                "a photograph of a person wearing a {}",
                threshold=0.33,
                margin=0.02,
                top_k=1,
            )
            if ranked_clothes:
                clothing.append(str(ranked_clothes[0]))
        except Exception:
            people.append("person")

    if people:
        contents["People"] = list(dict.fromkeys(people))[:4]
    if clothing:
        contents["Clothing"] = list(dict.fromkeys(clothing))[:3]
    return contents


def understand_scenes(samples, scenes, detections_by_time, actions, vision, video_id, progress=None):
    """Describe each shot from tracked objects plus a conservative keyframe read."""
    get_image_embedding = vision["get_image_embedding"]
    frame_dir = os.path.join(VIDEO_FRAME_DIR, video_id)
    os.makedirs(frame_dir, exist_ok=True)

    embeddings = []
    understood = []

    for i, scene in enumerate(scenes):
        if progress:
            progress(f"Step 4/5: Analyzing scene {i + 1} of {len(scenes)}...")
        sample = _pick_keyframe(samples, scene)
        key_name = f"scene_{scene['id']:03d}.jpg"
        key_path = os.path.join(frame_dir, key_name)
        cv2.imwrite(key_path, sample["frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        scene_dets = []
        scene_frames = 0
        keyframe_dets = []
        best_dt = 1e9
        for item in detections_by_time:
            if scene["start"] <= item["time"] < scene["end"] + 1e-4:
                scene_frames += 1
                scene_dets.extend(item["detections"])
            dt = abs(item["time"] - sample["time"])
            if dt < best_dt:
                best_dt = dt
                keyframe_dets = item["detections"]

        object_counts = {}
        object_conf = {}
        for det in scene_dets:
            label = det["label"]
            object_counts[label] = object_counts.get(label, 0) + 1
            object_conf[label] = max(object_conf.get(label, 0.0), float(det.get("confidence") or 0.0))

        objects = []
        for label, count in sorted(object_counts.items(), key=lambda kv: (-kv[1], -object_conf.get(kv[0], 0))):
            if _keep_detection(label, object_conf.get(label, 0.0), count, scene_frames):
                objects.append(label)

        contents = {}
        try:
            contents = _video_scene_contents(sample["frame"], keyframe_dets, vision)
        except Exception as exc:
            print("Scene understanding warning:", exc)

        if objects:
            contents["Objects"] = objects[:12]
        animals = [
            obj for obj in objects
            if obj in {
                "bird", "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe",
            }
        ]
        food = [obj for obj in objects if obj in FOOD_OBJECTS]
        if animals:
            contents["Animals"] = animals
        if food:
            contents["Food / Vegetables"] = food

        scene_actions = [
            item["label"]
            for item in actions
            if item["start"] < scene["end"] and item["end"] > scene["start"]
        ]
        scene_actions = list(dict.fromkeys(scene_actions))
        if "sitting" in scene_actions and "standing" in scene_actions:
            sit_n = sum(1 for a in actions if a["label"] == "sitting" and a["start"] < scene["end"] and a["end"] > scene["start"])
            stand_n = sum(1 for a in actions if a["label"] == "standing" and a["start"] < scene["end"] and a["end"] > scene["start"])
            drop = "standing" if sit_n >= stand_n else "sitting"
            scene_actions = [a for a in scene_actions if a != drop]

        embedding = None
        try:
            embedding = get_image_embedding(key_path, normalize=True).detach().cpu()
        except Exception as exc:
            print("Scene embedding warning:", exc)

        if embedding is not None:
            embeddings.append(embedding.reshape(1, -1))
        else:
            embeddings.append(None)

        # Object co-occurrence in a scene is not enough to claim interaction.
        relationships = []

        understood.append(
            {
                **scene,
                "keyframe": f"{video_id}/{key_name}",
                "objects": objects[:12],
                "actions": scene_actions,
                "contents": contents,
                "relationships": list(dict.fromkeys(relationships))[:6],
                "transcript": "",
            }
        )

    stacked = None
    ready = [e for e in embeddings if e is not None]
    if ready and len(ready) == len(embeddings):
        stacked = torch.cat(ready, dim=0)
        torch.save(stacked, video_embedding_path(video_id))
    return understood, stacked


def _ffmpeg_executable():
    candidates = []
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(found)
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.isfile(bundled):
            bundled_dir = os.path.dirname(bundled)
            # Ensure standard ffmpeg.exe name exists for tools like Whisper
            standard_exe = os.path.join(bundled_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            if not os.path.exists(standard_exe):
                try:
                    shutil.copy2(bundled, standard_exe)
                except Exception:
                    pass
            if os.path.exists(standard_exe):
                candidates.append(standard_exe)
            candidates.append(bundled)
            if bundled_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bundled_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
    for path in candidates:
        if path and os.path.isfile(path):
            exe_dir = os.path.dirname(os.path.abspath(path))
            if exe_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = exe_dir + os.pathsep + os.environ.get("PATH", "")
            return os.path.abspath(path)
    return None


_ffmpeg_executable()


def _extract_audio_wav(video_path, wav_path):
    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        return False, "ffmpeg not found; install ffmpeg or imageio-ffmpeg for speech-to-text."
    video_path = os.path.abspath(video_path)
    wav_path = os.path.abspath(wav_path)
    if not os.path.isfile(video_path):
        return False, f"Video file is missing: {video_path}"
    command = [
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        wav_path,
    ]
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(ffmpeg) + os.pathsep + env.get("PATH", "")
    kwargs = {
        "check": True,
        "capture_output": True,
        "env": env,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        subprocess.run(command, **kwargs)
        return os.path.exists(wav_path), None
    except FileNotFoundError:
        return False, f"Could not launch ffmpeg at {ffmpeg}"
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or b"").decode("utf-8", "ignore").strip()
        return False, err[-500:] if err else str(exc)
    except Exception as exc:
        return False, str(exc)


def _load_whisper():
    global _whisper_model, _whisper_error
    if _whisper_model is not None:
        return _whisper_model
    try:
        import whisper

        model_name = "base" if torch.cuda.is_available() else "tiny"
        print(f"Loading Whisper speech-to-text model ({model_name})...")
        _whisper_model = whisper.load_model(model_name)
        print("Whisper model loaded.")
        _whisper_error = None
    except Exception as exc:
        _whisper_error = str(exc)
        print("Whisper unavailable:", exc)
        _whisper_model = None
    return _whisper_model


def transcribe_video(video_path, video_id):
    """Extract spoken words with timestamps when Whisper + ffmpeg are available."""
    wav_path = os.path.join(VIDEO_INDEX_DIR, f"{video_id}.wav")
    ok, error = _extract_audio_wav(video_path, wav_path)
    if not ok:
        return [], [], error or "Could not extract audio."

    model = _load_whisper()
    if model is None:
        return [], [], _whisper_error or "openai-whisper is not installed."

    try:
        result = model.transcribe(
            wav_path,
            word_timestamps=True,
            verbose=False,
            fp16=torch.cuda.is_available(),
        )
    except Exception as exc:
        return [], [], str(exc)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    segments = []
    words = []
    for seg in result.get("segments") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        segments.append({"start": start, "end": end, "text": text})
        for word in seg.get("words") or []:
            token = (word.get("word") or "").strip()
            if not token:
                continue
            words.append(
                {
                    "word": token,
                    "start": float(word.get("start") or start),
                    "end": float(word.get("end") or end),
                }
            )
        if not seg.get("words"):
            pieces = re.findall(r"[A-Za-z0-9']+", text)
            span = max(0.05, end - start)
            if pieces:
                step = span / len(pieces)
                for i, token in enumerate(pieces):
                    words.append(
                        {
                            "word": token,
                            "start": start + i * step,
                            "end": start + (i + 1) * step,
                        }
                    )
    return segments, words, None


def add_talking_from_transcript(actions, scenes, segments):
    """If speech is heard while a scene has a person, mark talking at that time."""
    if not segments:
        return actions
    extra = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if len(text) < 2:
            continue
        scene = _scene_for_time(scenes, float(seg.get("start") or 0.0))
        if scene is None:
            continue
        if "person" not in (scene.get("objects") or []):
            continue
        extra.append(
            {
                "label": "talking",
                "start": float(seg["start"]),
                "end": float(seg.get("end") or seg["start"] + 0.8),
                "duration": float((seg.get("end") or seg["start"] + 0.8) - seg["start"]),
                "scene_id": scene["id"],
            }
        )
        if "talking" not in scene.get("actions", []):
            scene.setdefault("actions", []).append("talking")
    if extra:
        actions = _merge_intervals(list(actions) + extra, gap=1.2)
    return actions


def attach_transcript_to_scenes(scenes, segments):
    for scene in scenes:
        parts = [
            seg["text"]
            for seg in segments
            if seg["start"] < scene["end"] and seg["end"] > scene["start"]
        ]
        scene["transcript"] = " ".join(parts).strip()
    return scenes


def flatten_concepts(scenes):
    concepts = defaultdict(list)
    for scene in scenes:
        for category, items in (scene.get("contents") or {}).items():
            for item in items:
                concepts[str(item).lower()].append(
                    {
                        "start": scene["start"],
                        "end": scene["end"],
                        "scene_id": scene["id"],
                        "category": category,
                    }
                )
        for obj in scene.get("objects") or []:
            concepts[str(obj).lower()].append(
                {
                    "start": scene["start"],
                    "end": scene["end"],
                    "scene_id": scene["id"],
                    "category": "Objects",
                }
            )
        for action in scene.get("actions") or []:
            concepts[str(action).lower()].append(
                {
                    "start": scene["start"],
                    "end": scene["end"],
                    "scene_id": scene["id"],
                    "category": "Actions",
                }
            )
    return {k: v for k, v in concepts.items()}


def _progress(callback, message):
    print(message)
    if callback:
        callback(message)


def apply_transcript(index, segments, words, asr_error):
    index["transcript"] = segments
    index["words"] = words
    index["asr_error"] = asr_error
    index["full_transcript"] = " ".join(seg["text"] for seg in segments).strip()
    attach_transcript_to_scenes(index.get("scenes") or [], segments)
    index["concepts"] = flatten_concepts(index.get("scenes") or [])
    return index


def update_index_transcript(index):
    video_path = os.path.join(VIDEO_UPLOAD_DIR, index.get("filename") or "")
    segments, words, asr_error = transcribe_video(video_path, index["video_id"])
    apply_transcript(index, segments, words, asr_error)
    save_video_index(index)
    return index


def analyze_video(
    video_path,
    video_id,
    filename,
    vision,
    sample_fps=1.5,
    original_name=None,
    progress=None,
):
    """Run the full understanding pipeline and write a searchable index."""
    ensure_video_dirs()
    _progress(progress, "Step 1/5: Sampling video frames...")
    probe = _open_video(video_path)
    probe[0].release()
    duration = float(probe[5] or 0.0)
    adaptive_fps, adaptive_max = _adaptive_sample_rate(duration)
    sample_fps = max(sample_fps, adaptive_fps) if duration <= 90 else adaptive_fps
    samples, meta = sample_frames(
        video_path,
        sample_fps=sample_fps,
        max_frames=adaptive_max,
    )
    _progress(
        progress,
        f"Sampled {len(samples)} frames from {meta['duration']:.1f}s video.",
    )

    _progress(progress, "Step 2/5: Detecting scene boundaries...")
    try:
        scenes = detect_scenes_from_video(video_path, meta["duration"], progress=progress)
    except Exception as exc:
        print("Full-file scene scan warning:", exc)
        scenes = detect_scenes(samples, meta["duration"])
    # Merge only adjacent segments with confident, multi-frame agreement on setting.
    # Non-adjacent returns to the same location remain separate chronological scenes.
    scenes = _normalize_scene_timeline(scenes, meta["duration"], samples=samples, vision=vision, progress=progress)
    _progress(progress, f"Found {len(scenes)} validated chronological scenes.")

    _progress(progress, "Step 3/5: Detecting and tracking objects...")
    detections_by_time, tracks = track_objects(
        samples,
        vision["yolo"],
        scenes,
    )

    _progress(progress, "Detecting actions and poses...")
    actions = detect_actions(
        samples,
        detections_by_time,
        vision,
        scenes,
        tracks=tracks,
    )

    _progress(progress, f"Step 4/5: Understanding {len(scenes)} scenes with VisualRAG...")
    scenes, _embeddings = understand_scenes(
        samples,
        scenes,
        detections_by_time,
        actions,
        vision,
        video_id,
        progress=progress,
    )

    _progress(progress, "Step 5/5: Transcribing speech with Whisper...")
    segments, words, asr_error = transcribe_video(video_path, video_id)
    scenes = attach_transcript_to_scenes(scenes, segments)
    actions = add_talking_from_transcript(actions, scenes, segments)
    if asr_error:
        print("ASR warning:", asr_error)

    # Free OpenCV frames from RAM.
    for sample in samples:
        sample.pop("frame", None)

    index = {
        "pipeline_version": PIPELINE_VERSION,
        "video_id": video_id,
        "filename": filename,
        "original_name": original_name or original_video_name(filename),
        "duration": meta["duration"],
        "fps": meta["fps"],
        "width": meta["width"],
        "height": meta["height"],
        "sample_fps": meta["sample_fps"],
        "sampled_frames": meta["sampled_frames"],
        "scenes": scenes,
        "tracks": tracks,
        "actions": actions,
        "transcript": segments,
        "words": words,
        "concepts": flatten_concepts(scenes),
        "asr_error": asr_error,
        "full_transcript": " ".join(seg["text"] for seg in segments).strip(),
    }
    save_video_index(index)
    _progress(progress, "Video index saved.")
    return index
