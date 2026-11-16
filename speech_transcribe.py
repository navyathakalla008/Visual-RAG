"""Robust multilingual speech transcription helpers for VisualRAG VPKE."""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

SAMPLE_RATE = 16000
_MODEL = None
_MODEL_NAME = None


class SpeechError(RuntimeError):
    """Readable speech/FFmpeg error for the UI."""


def find_ffmpeg() -> str:
    """Find and validate FFmpeg, returning its full executable path."""
    candidates: List[str] = []
    override = os.environ.get("FFMPEG_PATH")
    if override:
        candidates.append(override)
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(found)
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    if os.name == "nt":
        roots = [os.environ.get("LOCALAPPDATA", ""), r"C:\Program Files", r"C:\Program Files (x86)", r"C:\ProgramData\chocolatey"]
        patterns = [
            os.path.join(root, "**", "ffmpeg.exe") for root in roots if root
        ]
        for pattern in patterns:
            candidates.extend(glob.glob(pattern, recursive=True))
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(candidate)
        if candidate in seen or not os.path.isfile(candidate):
            continue
        seen.add(candidate)
        try:
            result = subprocess.run(
                [candidate, "-version"], capture_output=True, timeout=10,
                check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
            if result.returncode == 0:
                return candidate
        except (OSError, subprocess.SubprocessError):
            continue
    raise SpeechError(
        "FFmpeg was not found. In your project virtual environment run: "
        "python -m pip install --upgrade imageio-ffmpeg. Alternatively install FFmpeg "
        "or set FFMPEG_PATH to the full path of ffmpeg.exe."
    )


def extract_audio(video_path: str) -> np.ndarray:
    """Extract mono 16 kHz float audio by calling the resolved FFmpeg path directly."""
    if not os.path.isfile(video_path):
        raise SpeechError(f"Video file not found: {video_path}")
    ffmpeg = find_ffmpeg()
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", os.path.abspath(video_path),
               "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    try:
        proc = subprocess.run(command, capture_output=True, check=False,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
    except OSError as exc:
        raise SpeechError(f"Could not launch FFmpeg ({ffmpeg}): {exc}") from exc
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if proc.returncode != 0 or audio.size == 0:
        error = proc.stderr.decode("utf-8", "ignore").strip()[-400:]
        raise SpeechError("Could not extract audio. The video may have no audio track. " + error)
    return audio


def fmt_clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _get_model():
    global _MODEL, _MODEL_NAME
    try:
        import torch
        import whisper
    except ImportError as exc:
        raise SpeechError("Whisper dependencies are missing. Run: python -m pip install openai-whisper imageio-ffmpeg numpy") from exc
    name = os.environ.get("WHISPER_MODEL", "tiny")
    if _MODEL is None or _MODEL_NAME != name:
        try:
            _MODEL = whisper.load_model(name, device="cuda" if torch.cuda.is_available() else "cpu")
        except Exception as exc:
            raise SpeechError(f"Could not load Whisper model '{name}': {exc}") from exc
        _MODEL_NAME = name
    return _MODEL


def detect_language(model, audio: np.ndarray, windows: int = 3) -> str:
    import whisper
    win = 30 * SAMPLE_RATE
    n = len(audio)
    starts = [0] if n <= win else sorted(set(int(i * (n - win) / max(1, windows - 1)) for i in range(windows)))
    votes: Counter = Counter()
    for start in starts:
        chunk = audio[start:start + win]
        if chunk.size == 0 or float(np.abs(chunk).mean()) < 1e-4:
            continue
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(chunk), n_mels=model.dims.n_mels).to(model.device)
        _, probabilities = model.detect_language(mel)
        lang = max(probabilities, key=probabilities.get)
        votes[lang] += float(probabilities[lang])
    if not votes:
        raise SpeechError("No speech detected in the audio track.")
    return votes.most_common(1)[0][0]


def transcribe_audio(audio: np.ndarray, language: Optional[str] = None) -> Dict:
    import torch
    model = _get_model()
    if not torch.cuda.is_available():
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    # Let Whisper detect the spoken language during transcription itself. This avoids
    # a separate multi-window language-detection pass on long audio files. Do not use
    # task="translate": the transcript should remain in the language being spoken.
    transcribe_options = {
        "task": "transcribe",
        "word_timestamps": True,
        "fp16": torch.cuda.is_available(),
        "condition_on_previous_text": False,
        "verbose": False,
        "beam_size": 1,
        "best_of": 1,
        "temperature": 0.0,
        "no_speech_threshold": 0.6,
    }
    if language:
        transcribe_options["language"] = language
    result = model.transcribe(audio, **transcribe_options)
    lang = result.get("language") or language or "unknown"
    segments = []
    words = []
    for segment in result.get("segments") or []:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        segments.append({"start": start, "end": end, "start_label": fmt_clock(start),
                         "end_label": fmt_clock(end), "text": text, "language": lang})
        for word in segment.get("words") or []:
            token = (word.get("word") or "").strip()
            if token:
                words.append({"word": token, "start": float(word.get("start") or start),
                              "end": float(word.get("end") or end)})
    return {"language": lang, "segments": segments, "words": words,
            "text": " ".join(item["text"] for item in segments)}


def safe_transcribe(video_path: str, language: Optional[str] = None) -> Dict:
    try:
        return {"ok": True, **transcribe_audio(extract_audio(video_path), language)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "segments": [], "words": [], "language": None, "text": ""}
