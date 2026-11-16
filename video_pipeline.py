"""Video understanding pipeline: sample frames, scenes, tracking, ASR, index."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict

import cv2
import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads", "videos")
VIDEO_FRAME_DIR = os.path.join(PROJECT_ROOT, "uploads", "video_frames")
VIDEO_INDEX_DIR = os.path.join(PROJECT_ROOT, "uploads", "video_index")
PIPELINE_VERSION = 19
NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "ru": "rus_Cyrl",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "fil": "tgl_Latn",
    "uk": "ukr_Cyrl",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "fa": "pes_Arab",
    "ur": "urd_Arab",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "pa": "pan_Guru",
}

_TRANSLATOR = None
_TRANSLATOR_TOKENIZER = None


def _get_translator():
    global _TRANSLATOR, _TRANSLATOR_TOKENIZER
    if _TRANSLATOR is not None:
        return _TRANSLATOR, _TRANSLATOR_TOKENIZER
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _TRANSLATOR_TOKENIZER = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        _TRANSLATOR = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        _TRANSLATOR.eval()
        return _TRANSLATOR, _TRANSLATOR_TOKENIZER
    except Exception as exc:
        print("Translation model unavailable:", exc)
        return None, None


def translate_text(text, source_lang="en", target_lang="te"):
    if not text or not text.strip():
        return ""
    if source_lang == target_lang:
        return text
    src_code = NLLB_LANGUAGE_CODES.get(source_lang, "eng_Latn")
    tgt_code = NLLB_LANGUAGE_CODES.get(target_lang)
    if tgt_code is None:
        raise ValueError(f"Unsupported target language: {target_lang}")
    model, tokenizer = _get_translator()
    if model is None or tokenizer is None:
        raise RuntimeError("Translation model is not available.")
    import torch
    tokenizer.src_lang = src_code
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)
    with torch.no_grad():
        output = model.generate(**inputs, forced_bos_token_id=forced_bos, max_length=512)
    result = tokenizer.batch_decode(output, skip_special_tokens=True)
    return result[0] if result else text


def translate_transcript_segments(segments, source_lang="en", target_lang="te"):
    translated = []
    for seg in segments:
        text = seg.get("text", "")
        try:
            new_text = translate_text(text, source_lang, target_lang)
        except Exception as exc:
            new_text = text
            print(f"Translation warning for segment: {exc}")
        new_seg = dict(seg)
        new_seg["text"] = new_text
        new_seg["original_text"] = text
        translated.append(new_seg)
    return translated

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


def format_time_minutes(seconds):
    """Convert seconds to clock display. Internal values remain unrounded seconds."""
    if seconds is None:
        return "00:00.00"
    total_centiseconds = max(0, int(round(float(seconds) * 100.0)))
    hours, remainder = divmod(total_centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centiseconds = divmod(remainder, 100)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"
    return f"{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def format_timestamp(seconds):
    """Display a precise clock timestamp. Internal values remain in seconds."""
    return format_time_minutes(seconds)


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
        index = json.load(handle)
    try:
        normalize_pipeline_timeline(index)
    except Exception as exc:
        print(f"Existing index timeline normalization warning for {video_id}: {exc}")
    return index


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
                "speech_status": index.get("speech_status") or speech_status(index.get("transcript") or [], index.get("asr_error")),
                "asr_error": index.get("asr_error"),
                "mtime": os.path.getmtime(path),
            }
        )
    items.sort(key=lambda item: item["mtime"], reverse=True)
    return items


def file_sha256(path, chunk_size=1024 * 1024):
    """Compute a stable hash for an uploaded video file."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def find_matching_index(original_name, file_size, file_hash=None):
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
        if existing_name != wanted:
            continue
        index = load_video_index(item["video_id"])
        if not index:
            continue
        if int(index.get("pipeline_version") or 0) != PIPELINE_VERSION:
            continue
        if file_hash and index.get("file_hash") and index.get("file_hash") != file_hash:
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


def _probe_video_duration(video_path):
    """Read the authoritative media duration with ffprobe, with OpenCV fallback."""
    try:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            import imageio_ffmpeg
            ffprobe = os.path.join(
                os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()),
                "ffprobe.exe" if os.name == "nt" else "ffprobe",
            )
        if ffprobe and os.path.isfile(ffprobe):
            result = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "json", os.path.abspath(video_path)],
                capture_output=True, timeout=30, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout.decode("utf-8", "ignore"))
                duration = float((data.get("format") or {}).get("duration") or 0.0)
                if duration > 0:
                    return duration
    except Exception as exc:
        print(f"ffprobe duration warning: {exc}")
    return 0.0


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
    duration = _probe_video_duration(video_path)
    if duration <= 0 and frame_count > 0:
        duration = frame_count / fps
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
        pos_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        time_sec = pos_msec / 1000.0 if pos_msec > 0 else index / fps
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


def _refine_scene_cut_to_frame(video_path, timestamp, fps, duration, window=0.5):
    """Snap a detected cut to the highest-change adjacent pair of native frames."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        capture.release()
        return float(timestamp)
    center_index = max(0, min(int(round(float(timestamp) * fps)), max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)))
    first_index = max(1, int(round((max(0.0, float(timestamp) - window) * fps))))
    last_index = min(
        max(first_index, int(round(min(duration, float(timestamp) + window) * fps))),
        max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1),
    )
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, first_index - 1))
    previous = None
    best_index = center_index
    best_score = -1.0
    index = max(0, first_index - 1)
    while index <= last_index:
        ok, frame = capture.read()
        if not ok:
            break
        if previous is not None:
            previous_gray = cv2.cvtColor(cv2.resize(previous, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            current_gray = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            previous_hist = _histogram(previous)
            current_hist = _histogram(frame)
            hist_score = float(cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_BHATTACHARYYA))
            pixel_score = float(np.mean(np.abs(previous_gray.astype(np.float32) - current_gray.astype(np.float32))) / 255.0)
            score = 0.80 * hist_score + 0.20 * pixel_score
            if score > best_score:
                best_score = score
                best_index = index
        previous = frame
        index += 1
    capture.release()
    refined = max(0.0, min(duration, best_index / fps))
    return refined


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

        # FAST MODE: sample less often for longer clips to reduce CPU work.
        # For 5–15 minute videos this samples about twice per second.
        # Scene timestamps remain based on the sampled frame timestamps.
        interval = 0.25 if duration <= 120 else (0.5 if duration <= 900 else 0.75)
        step = max(1, int(round(fps * interval)))
        times, hists, grays = [], [], []
        index = 0
        report_every = max(1, frame_count // 20)
        last_report_index = -report_every if frame_count else 0
        while index < frame_count:
            if progress and index - last_report_index >= report_every:
                pct = int(index * 45 / max(frame_count, 1))
                _progress(progress, f"Step 2/5: Detecting scenes: {pct}%")
                last_report_index = index
            if index % step:
                if not capture.grab():
                    break
                index += 1
                continue
            ok, frame = capture.read()
            if not ok:
                break
            pos_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            timestamp = pos_msec / 1000.0 if pos_msec > 0 else index / fps
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
        # Edge features are computed only for frames that already show a meaningful
        # color/pixel change, which avoids expensive Canny calls on static frames.
        hist_change, pixel_change, edge_change = [], [], []
        compare_count = max(1, len(hists) - 1)
        compare_report_every = max(1, compare_count // 10)
        for i in range(1, len(hists)):
            hist_value = float(cv2.compareHist(hists[i-1], hists[i], cv2.HISTCMP_BHATTACHARYYA))
            pixel_value = float(np.mean(np.abs(grays[i] - grays[i-1])))
            hist_change.append(hist_value)
            pixel_change.append(pixel_value)
            if hist_value >= 0.12 or pixel_value >= 0.08:
                e0 = cv2.Canny((grays[i-1] * 255).astype(np.uint8), 60, 140)
                e1 = cv2.Canny((grays[i] * 255).astype(np.uint8), 60, 140)
                edge_change.append(float(np.mean(e0 != e1)))
            else:
                edge_change.append(0.0)
            if progress and (i % compare_report_every == 0 or i == compare_count):
                pct = 45 + int(i * 40 / compare_count)
                _progress(progress, f"Step 2/5: Detecting scenes: {pct}%")

        hist_change = np.asarray(hist_change, dtype=np.float32)
        pixel_change = np.asarray(pixel_change, dtype=np.float32)
        edge_change = np.asarray(edge_change, dtype=np.float32)
        # Normalize cues robustly so a single noisy frame does not set the scale.
        def robust_norm(values):
            med = float(np.median(values))
            mad = float(np.median(np.abs(values - med))) + 1e-6
            return np.clip((values - med) / (4.0 * mad), 0.0, 1.0)
        score = (0.80 * robust_norm(hist_change) +
                 0.15 * robust_norm(pixel_change) +
                 0.05 * robust_norm(edge_change))

        # A cut must be a local peak and show a meaningful change in at least
        # one strong cue. Local adaptive thresholds handle both quiet and busy clips.
        cuts = [0.0]
        # Adaptive minimum cut spacing: shorter for short videos, longer for long videos.
        if duration <= 60:
            min_cut_spacing = max(1.0, interval * 2.0)
        elif duration <= 300:
            min_cut_spacing = max(5.0, interval * 8.0)
        else:
            min_cut_spacing = max(20.0, interval * 30.0)
        previous_score = -1.0
        for i in range(len(score)):
            left = max(0, i - 15)
            right = min(len(score), i + 16)
            local = score[left:right]
            med = float(np.median(local)) if len(local) else 0.0
            mad = float(np.median(np.abs(local - med))) + 1e-6
            threshold = max(0.60, med + 6.0 * mad)
            local_peak = (i == 0 or score[i] >= score[i-1]) and (i == len(score)-1 or score[i] >= score[i+1])
            strong_color = hist_change[i] >= 0.60
            strong_pixels = pixel_change[i] >= 0.35
            strong_combined = score[i] >= threshold and (hist_change[i] >= 0.50 or pixel_change[i] >= 0.25)
            if not local_peak or not (strong_color or strong_pixels or strong_combined):
                continue
            timestamp = times[i + 1]
            if timestamp - cuts[-1] < min_cut_spacing:
                # Keep the stronger of two nearby candidates rather than always
                # accepting the first one.
                if len(cuts) > 1 and score[i] > previous_score:
                    cuts[-1] = float(timestamp)
                continue
            cuts.append(float(timestamp))
            previous_score = float(score[i])

        cuts = sorted(set([0.0] + [min(max(c, 0.0), duration) for c in cuts]))
        refined_cuts = [0.0]
        for cut in cuts[1:]:
            if cut >= duration:
                continue
            refined = _refine_scene_cut_to_frame(video_path, cut, fps, duration)
            if refined - refined_cuts[-1] >= 0.05:
                refined_cuts.append(float(refined))
        cuts = sorted(set(refined_cuts))
        scenes = _scenes_from_cuts(cuts, duration)
        if not scenes:
            scenes = [{"id": 1, "start": 0.0, "end": duration, "duration": duration}]
        if progress:
            _progress(progress, "Step 2/5: Detecting scenes: 90%")
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
    if scenes and abs(time_sec - scenes[-1]["end"]) < 1e-3:
        return scenes[-1]
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
    return conf >= CLASS_MIN_CONF.get(label, 0.20)


def track_objects(
    samples,
    yolo_model,
    scenes,
    confidence=0.18,
    progress=None,
    supplemental_model=None,
):
    """Fuse two detectors and track their detections with persistent BoT-SORT IDs."""
    detections_by_time = []
    track_states = {}
    tracker_ids = {}
    aliases = {}
    next_id = 1
    previous_scene_id = None
    total_samples = len(samples)
    models = [yolo_model]
    if supplemental_model is not None and supplemental_model is not yolo_model:
        models.append(supplemental_model)

    def reset_trackers(model):
        predictor = getattr(model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()):
            tracker.reset()

    def resolve_track_id(track_id):
        if track_id is None:
            return None
        path = []
        while track_id in aliases:
            path.append(track_id)
            track_id = aliases[track_id]
        for alias in path:
            aliases[alias] = track_id
        return track_id

    def record_detection(detection, time_sec):
        track_id = detection["track_id"]
        if track_id is None:
            return
        state = track_states.setdefault(track_id, {"observations": []})
        state["observations"].append(
            {
                "time": time_sec,
                "confidence": detection["confidence"],
                "box": detection["box"],
                "label": detection["label"],
            }
        )

    def detect_and_track(model, frame, model_idx, scene_id):
        results = model.track(
            frame,
            conf=confidence,
            iou=0.50,
            imgsz=960,
            max_det=300,
            tracker="botsort.yaml",
            persist=True,
            verbose=False,
        )
        detections = []
        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            return detections
        class_ids = boxes.cls.tolist()
        confidences = boxes.conf.tolist()
        coordinates = boxes.xyxy.tolist()
        local_ids = boxes.id.tolist() if boxes.id is not None else [None] * len(boxes)
        names = model.names
        for class_id, conf, xyxy, local_id in zip(
            class_ids, confidences, coordinates, local_ids
        ):
            label = str(names[int(class_id)]).lower()
            track_id = None
            tracker_key = None
            if local_id is not None:
                tracker_key = (scene_id, model_idx, int(local_id))
                if tracker_key not in tracker_ids:
                    tracker_ids[tracker_key] = next_id_holder[0]
                    next_id_holder[0] += 1
                track_id = resolve_track_id(tracker_ids[tracker_key])
                tracker_ids[tracker_key] = track_id
            detections.append(
                {
                    "label": label,
                    "confidence": float(conf),
                    "track_id": track_id,
                    "tracker_key": tracker_key,
                    "box": [float(value) for value in xyxy],
                }
            )
        return detections

    next_id_holder = [next_id]
    for model in models:
        reset_trackers(model)
    try:
        for sample_idx, sample in enumerate(samples, start=1):
            if progress and (
                sample_idx == 1 or sample_idx % 25 == 0 or sample_idx == total_samples
            ):
                pct = int(sample_idx * 100 / max(1, total_samples))
                _progress(progress, f"Step 3/5: Detecting objects: {pct}%")
            time_sec = float(sample["time"])
            scene = _scene_for_time(scenes, time_sec)
            scene_id = scene["id"] if scene else None
            if sample_idx == 1 or scene_id != previous_scene_id:
                for model in models:
                    reset_trackers(model)
                previous_scene_id = scene_id

            next_id_holder[0] = next_id
            model_detections = [
                detect_and_track(model, sample["frame"], model_idx, scene_id)
                for model_idx, model in enumerate(models)
            ]
            next_id = next_id_holder[0]

            frame_dets = model_detections[0]
            for detection in frame_dets:
                record_detection(detection, time_sec)

            for supplemental_detections in model_detections[1:]:
                for detection in supplemental_detections:
                    overlaps = [
                        primary
                        for primary in frame_dets
                        if primary["label"] == detection["label"]
                        and _box_iou(primary["box"], detection["box"]) >= 0.45
                    ]
                    if overlaps:
                        matched = max(
                            overlaps,
                            key=lambda primary: _box_iou(
                                primary["box"], detection["box"]
                            ),
                        )
                        if detection["track_id"] is not None and matched["track_id"] is not None:
                            supplemental_id = resolve_track_id(detection["track_id"])
                            primary_id = resolve_track_id(matched["track_id"])
                            if supplemental_id != primary_id:
                                aliases[supplemental_id] = primary_id
                            if detection["tracker_key"] is not None:
                                tracker_ids[detection["tracker_key"]] = primary_id
                            matched["track_id"] = primary_id
                            matched["confidence"] = max(
                                matched["confidence"], detection["confidence"]
                            )
                        elif detection["track_id"] is not None:
                            matched["track_id"] = detection["track_id"]
                            record_detection(matched, time_sec)
                        continue

                    frame_dets.append(detection)
                    record_detection(detection, time_sec)

            for detection in frame_dets:
                detection.pop("tracker_key", None)
            detections_by_time.append({"time": time_sec, "detections": frame_dets})
    finally:
        for model in models:
            reset_trackers(model)

    track_list = []
    merged_states = {}
    for track_id, info in track_states.items():
        canonical_id = resolve_track_id(track_id)
        merged_states.setdefault(canonical_id, []).extend(info["observations"])

    for item in detections_by_time:
        for detection in item["detections"]:
            detection["track_id"] = resolve_track_id(detection["track_id"])

    for track_id, observations in merged_states.items():
        observations.sort(key=lambda observation: observation["time"])
        unique_observations = []
        for observation in observations:
            duplicate_idx = next(
                (
                    idx
                    for idx, existing in enumerate(unique_observations)
                    if abs(existing["time"] - observation["time"]) < 1e-3
                    and _box_iou(existing["box"], observation["box"]) >= 0.45
                ),
                None,
            )
            if duplicate_idx is None:
                unique_observations.append(observation)
            elif observation["confidence"] > unique_observations[duplicate_idx]["confidence"]:
                unique_observations[duplicate_idx] = observation

        times = [observation["time"] for observation in unique_observations]
        if not times:
            continue
        mean_conf = float(
            np.mean([observation["confidence"] for observation in unique_observations])
        )
        hits = len(times)
        label_scores = defaultdict(float)
        for observation in unique_observations:
            label_scores[observation["label"]] += observation["confidence"]
        label = max(label_scores, key=label_scores.get)
        start, end = min(times), max(times)
        track_list.append(
            {
                "track_id": int(track_id),
                "label": label,
                "start": float(start),
                "end": float(end),
                "duration": float(end - start),
                "confidence": mean_conf,
                "hits": hits,
                "timestamps": times,
                "times": times,
                "appearances": times,
                "boxes": [observation["box"] for observation in unique_observations],
                "bounding_boxes": [observation["box"] for observation in unique_observations],
                "scene_ids": sorted(
                    {
                        scene["id"]
                        for time_sec in times
                        if (scene := _scene_for_time(scenes, time_sec))
                    }
                ),
            }
        )
    next_id = next_id_holder[0]
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


def detect_actions(samples, detections_by_time, vision, scenes, tracks=None, progress=None):
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
    total_samples = min(len(samples), len(detections_by_time))
    for idx, (sample, timed) in enumerate(zip(samples, detections_by_time)):
        if progress and (idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == total_samples):
            pct = int((idx + 1) * 100 / max(1, total_samples))
            _progress(progress, f"Step 3/5: Detecting actions: {pct}%")
        frame = sample.get("frame")
        if frame is None:
            per_frame.append((float(sample.get("time", 0)), set()))
            continue
        found = set()
        people = [d for d in (timed.get("detections") or []) if d.get("label") == "person"]
        # Only score the most visible people to keep CPU processing practical.
        people.sort(key=lambda d: (d.get("box", [0,0,0,0])[2]-d.get("box", [0,0,0,0])[0]) * (d.get("box", [0,0,0,0])[3]-d.get("box", [0,0,0,0])[1]), reverse=True)
        for det in people[:2]:
            crop = _crop_bgr(frame, det.get("box", [0,0,0,0]), pad=0.12)
            if crop is None or crop.size == 0:
                continue
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if idx % 2 == 0 and clip_rank_image is not None and get_pil_embedding is not None:
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
            if pose_model is not None and idx % 4 == 0:
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
            if run and (i != run[-1] + 1 or per_frame[i][0] - per_frame[run[-1]][0] > 3.2):
                if len(run) >= min_support:
                    runs.append(run)
                run = []
            run.append(i)
        if len(run) >= min_support:
            runs.append(run)
        for run in runs:
            start_t = per_frame[run[0]][0]
            end_t = min(float(samples[-1].get("time", start_t)), per_frame[run[-1]][0])
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
        # Use the middle frame of the scene for a single, fast classification.
        start, end = float(scene["start"]), float(scene["end"])
        candidates = [s for s in samples if start <= float(s.get("time", -1)) < end and s.get("frame") is not None]
        if not candidates:
            return None
        mid = (start + end) / 2.0
        sample = min(candidates, key=lambda s: abs(float(s.get("time", 0.0)) - mid))
        rgb = cv2.cvtColor(sample["frame"], cv2.COLOR_BGR2RGB)
        features = embed(Image.fromarray(rgb))
        if features is None:
            return None
        ranked = rank(features, SCENE_TYPE_LABELS, "a photo of {}", threshold=0.25, margin=0.01, top_k=1)
        if ranked:
            return str(ranked[0]).strip().lower()
        return None
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

    # Keep all real detected shot boundaries, even in short videos. The detector
    # already enforces minimum spacing and the normalized timeline remains exact.

    # Preserve detected shot boundaries. Only redundant, very short scenes have
    # already been merged above; do not collapse distinct shots just because their
    # CLIP setting label or average color happens to match.
    if progress:
        _progress(progress, "Step 2/5: Detecting scenes: 95%")
    cleaned = [dict(s) for s in cleaned]
    for item in cleaned:
        item.pop("_setting_label", None)
        item.pop("_hist", None)

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


def _normalize_output_interval(item, duration, point_is_valid=False):
    """Clamp one seconds-based interval to the authoritative container timeline."""
    start = max(0.0, min(duration, float(item.get("start", 0.0) or 0.0)))
    end = max(start, min(duration, float(item.get("end", start) or start)))
    if point_is_valid and end <= start:
        end = min(duration, start)
    item["start"] = float(start)
    item["end"] = float(end)
    item["duration"] = float(max(0.0, end - start))
    return item


def normalize_pipeline_timeline(index, duration=None):
    """Apply one timeline and duration clamp to every indexed analysis output."""
    duration = float(duration if duration is not None else index.get("duration") or 0.0)
    if duration <= 0:
        return index

    index["duration"] = duration
    scenes = index.get("scenes") or []
    if scenes:
        try:
            validate_scene_timeline(scenes, duration)
        except ValueError as exc:
            print(f"Scene timeline validation warning: {exc}")
        for scene in scenes:
            _normalize_output_interval(scene, duration)

    for collection_name in ("tracks", "actions", "transcript", "words", "audio_events"):
        for item in index.get(collection_name) or []:
            _normalize_output_interval(item, duration)

    transcript = index.get("transcript") or []
    ordered_speakers = []
    for segment in sorted(transcript, key=lambda item: float(item.get("start", 0.0))):
        label = str(segment.get("speaker") or "").strip()
        if label and label.startswith("SPEAKER_") and label not in ordered_speakers:
            ordered_speakers.append(label)
    speaker_names = {
        label: f"Person {position}"
        for position, label in enumerate(ordered_speakers, start=1)
    }
    for segment in transcript:
        label = str(segment.get("speaker") or "").strip()
        if label in speaker_names:
            segment["speaker"] = speaker_names[label]

    for item in index.get("object_detections") or []:
        item["time"] = max(0.0, min(duration, float(item.get("time", 0.0) or 0.0)))

    for item in index.get("detections") or []:
        item["time"] = max(0.0, min(duration, float(item.get("time", 0.0) or 0.0)))

    return index

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
    for det in person_dets[:2]:
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
            pct = int((i + 1) * 100 / max(1, len(scenes)))
            _progress(progress, f"Step 4/5: Understanding scenes: {pct}%")
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

        # OCR text extraction from keyframe
        ocr_texts = []
        try:
            from ocr_service import extract_text_from_frame
            ocr_texts, _ = extract_text_from_frame(sample["frame"])
        except Exception:
            pass
        if ocr_texts:
            contents["On-screen Text"] = [t["text"] for t in ocr_texts]
        if objects:
            contents["Objects"] = objects
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
                "objects": objects,
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

        if torch.cuda.is_available():
            model_name = os.environ.get("WHISPER_MODEL", "base")
        else:
            model_name = os.environ.get("WHISPER_MODEL", "tiny")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        print(f"Loading Whisper speech-to-text model ({model_name})...")
        _whisper_model = whisper.load_model(model_name, device=device)
        print("Whisper model loaded.")
        _whisper_error = None
    except Exception as exc:
        _whisper_error = str(exc)
        print("Whisper unavailable:", exc)
        _whisper_model = None
    return _whisper_model


def _friendly_speaker_names(speaker_segments):
    """Convert diarization labels into stable Person 1, Person 2 labels."""
    if not speaker_segments:
        return {}
    ordered_labels = []
    for segment in sorted(speaker_segments, key=lambda item: float(item.get("start", 0.0))):
        label = str(segment.get("speaker") or "").strip()
        if label and label not in ordered_labels:
            ordered_labels.append(label)
    return {label: f"Person {index}" for index, label in enumerate(ordered_labels, start=1)}


def transcribe_video(video_path, video_id, progress=None, language=None):
    """Extract spoken words with timestamps when Whisper + ffmpeg are available."""
    video_duration = _probe_video_duration(video_path)
    _progress(progress, "Step 5/5: Transcribing speech: 0%")
    _progress(progress, "Step 5/5: Extracting audio for speech transcription...")
    wav_path = os.path.join(VIDEO_INDEX_DIR, f"{video_id}.wav")
    ok, error = _extract_audio_wav(video_path, wav_path)
    if not ok:
        return [], [], error or "Could not extract audio."

    model = _load_whisper()
    if model is None:
        return [], [], _whisper_error or "openai-whisper is not installed."

    _progress(progress, "Step 5/5: Running Whisper speech recognition...")
    try:
        transcribe_kwargs = {
            "word_timestamps": True,
            "verbose": False,
            "fp16": torch.cuda.is_available(),
            "beam_size": 1,
            "best_of": 1,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "no_speech_threshold": 0.6,
        }
        if language:
            transcribe_kwargs["language"] = language
        result = model.transcribe(wav_path, **transcribe_kwargs)
    except Exception as exc:
        return [], [], str(exc)
    _progress(progress, "Step 5/5: Transcribing speech: 100%")
    detected_language = result.get("language") or language or "unknown"
    # Speaker diarization (optional, CPU-compatible)
    speaker_segments = None
    try:
        from diarization_service import diarize_audio
        speaker_segments, _ = diarize_audio(wav_path)
    except Exception:
        pass
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    _progress(progress, "Step 5/5: Timestamping and indexing transcript words...")
    speaker_names = _friendly_speaker_names(speaker_segments)
    segments = []
    words = []
    for seg in result.get("segments") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        if video_duration > 0:
            start = max(0.0, min(video_duration, start))
            end = max(start, min(video_duration, end))
        # Find speaker for this segment
        speaker = None
        if speaker_segments:
            for sp in speaker_segments:
                if sp["start"] <= start < sp["end"]:
                    speaker = sp["speaker"]
                    break
        seg_data = {"start": start, "end": end, "text": text, "language": detected_language}
        if speaker:
            seg_data["speaker"] = speaker_names.get(speaker, speaker)
        segments.append(seg_data)
        for word in seg.get("words") or []:
            token = (word.get("word") or "").strip()
            if not token:
                continue
            word_start = float(word.get("start") or start)
            word_end = float(word.get("end") or end)
            if video_duration > 0:
                word_start = max(0.0, min(video_duration, word_start))
                word_end = max(word_start, min(video_duration, word_end))
            words.append(
                {
                    "word": token,
                    "start": word_start,
                    "end": word_end,
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
    """Flatten scene contents into a concept index for search."""
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


def cross_validate_events(scene_cuts, transcript_segments, ocr_data, diarization_data, object_detections):
    """Cross-modal validation: 'AI proposes, code disposes'."""
    validated = []
    for idx, cut_time in enumerate(scene_cuts):
        confidence = 0.50
        evidence = []

        # 1. Speech alignment
        for seg in transcript_segments or []:
            if abs(seg.get("start", 0) - cut_time) <= 1.5:
                confidence += 0.20
                evidence.append(f"Speech aligned at {seg.get('start')}s")
                break

        # 2. Speaker change
        for sp in diarization_data or []:
            if abs(sp.get("start", 0) - cut_time) <= 2.0:
                confidence += 0.15
                evidence.append(f"Speaker change: {sp.get('speaker')}")
                break

        # 3. OCR text change
        ocr = ocr_data.get(str(int(cut_time)) + "s", [])
        if any(item.get("confidence", 0) > 0.70 for item in ocr):
            confidence += 0.15
            evidence.append("High-confidence OCR text change")

        # 4. Object tracking boost
        for det in object_detections or []:
            if abs(det.get("time", 0) - cut_time) <= 1.0:
                confidence += 0.10
                evidence.append(f"Object {det.get('label')} at boundary")
                break

        if confidence >= 0.65:
            validated.append({
                "chapter_index": idx + 1,
                "timestamp": cut_time,
                "confidence": min(confidence, 1.0),
                "evidence": evidence,
            })
    return validated


def _progress(callback, message):
    print(message)
    if callback:
        callback(message)


def speech_status(segments, asr_error=None):
    """Normalize transcription outcome into one of the required UI statuses."""
    segments = segments or []
    error = (asr_error or "").strip()
    if segments and error:
        return "Completed with Warnings"
    if error:
        lowered = error.lower()
        if any(token in lowered for token in ("no audio", "no speech", "no recognizable", "empty")):
            return "No Speech Detected"
        return "Failed"
    if not segments:
        return "No Speech Detected"
    return "Completed"


def apply_transcript(index, segments, words, asr_error):
    index["transcript"] = segments
    index["words"] = words
    index["asr_error"] = asr_error
    detected_languages = {
        str(seg.get("language")).strip().lower()
        for seg in segments
        if seg.get("language") and str(seg.get("language")).lower() != "unknown"
    }
    if len(detected_languages) == 1:
        index["source_language"] = detected_languages.pop()
    elif language_hint := index.get("transcription_language"):
        index["source_language"] = str(language_hint).strip().lower()
    elif segments:
        index["source_language"] = "unknown"
    index["full_transcript"] = " ".join(seg["text"] for seg in segments).strip()
    index["speech_status"] = speech_status(segments, asr_error)
    attach_transcript_to_scenes(index.get("scenes") or [], segments)
    index["concepts"] = flatten_concepts(index.get("scenes") or [])
    normalize_pipeline_timeline(index)
    return index


def update_index_transcript(index, progress=None, language=None):
    video_path = os.path.join(VIDEO_UPLOAD_DIR, index.get("filename") or "")
    if language:
        index["transcription_language"] = str(language).strip().lower()
    segments, words, asr_error = transcribe_video(video_path, index["video_id"], progress=progress, language=language)
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
    file_hash = file_sha256(video_path)
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
        progress=progress,
        supplemental_model=vision.get("yolo_aux"),
    )

    _progress(progress, "Detecting actions and poses...")
    actions = detect_actions(
        samples,
        detections_by_time,
        vision,
        scenes,
        tracks=tracks,
        progress=progress,
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
    segments, words, asr_error = transcribe_video(video_path, video_id, progress=progress)
    scenes = attach_transcript_to_scenes(scenes, segments)
    actions = add_talking_from_transcript(actions, scenes, segments)
    if asr_error:
        print("ASR warning:", asr_error)

    # Build a persistent per-frame detection list with bounding boxes.
    object_detections = []
    for item in detections_by_time or []:
        for det in item.get("detections") or []:
            object_detections.append(
                {
                    "label": det.get("label"),
                    "confidence": det.get("confidence"),
                    "time": float(item.get("time", 0.0)),
                    "box": det.get("box"),
                    "track_id": det.get("track_id"),
                }
            )

    # Audio event detection
    _progress(progress, "Step 5b/5: Detecting audio events...")
    try:
        from audio_events import detect_audio_events
        audio_events, audio_error = detect_audio_events(video_path, video_id, progress=progress)
    except Exception as exc:
        audio_events = []
        audio_error = str(exc)
    if audio_error:
        print("Audio event detection warning:", audio_error)

    detected_languages = {
        str(seg.get("language")).strip().lower()
        for seg in segments
        if seg.get("language") and str(seg.get("language")).lower() != "unknown"
    }
    source_language = detected_languages.pop() if len(detected_languages) == 1 else "unknown"

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
        "object_detections": object_detections,
        "detections": object_detections,
        "actions": actions,
        "transcript": segments,
        "words": words,
        "concepts": flatten_concepts(scenes),
        "audio_events": audio_events,
        "audio_error": audio_error,
        "asr_error": asr_error,
        "source_language": source_language,
        "file_hash": file_hash,
        "speech_status": speech_status(segments, asr_error),
        "full_transcript": " ".join(seg["text"] for seg in segments).strip(),
    }
    normalize_pipeline_timeline(index, meta["duration"])
    save_video_index(index)
    _progress(progress, "Video index saved.")
    return index
