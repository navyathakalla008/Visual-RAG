"""Sound-event classification with a pretrained AudioSet AST model."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading

import numpy as np
import torch
from scipy import signal
from scipy.io import wavfile

SAMPLE_RATE = 16000
WINDOW_SECONDS = 5
HOP_SECONDS = 2.5
DEFAULT_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

_audio_model = None
_audio_processor = None
_audio_model_id = None
_audio_device = None
_audio_model_lock = threading.Lock()


def _extract_audio(video_path, wav_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            return False, f"Could not locate ffmpeg for audio extraction: {exc}"

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", video_path,
        "-map", "0:a:0",
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        wav_path,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, "Audio extraction timed out after 300 seconds."
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", "replace").strip()
        if "matches no streams" in stderr or "Stream map" in stderr:
            return False, "No audio track was found in the video."
        return False, f"ffmpeg could not extract audio: {stderr[-800:] or exc}"
    except OSError as exc:
        return False, f"Could not run ffmpeg: {exc}"

    if not os.path.isfile(wav_path) or os.path.getsize(wav_path) == 0:
        return False, "ffmpeg completed without producing an audio file."
    return True, None


def _load_audio_classifier():
    global _audio_model, _audio_processor, _audio_model_id, _audio_device
    model_id = os.environ.get("AUDIO_EVENT_MODEL", DEFAULT_MODEL).strip()
    if not model_id:
        raise ValueError("AUDIO_EVENT_MODEL cannot be empty.")

    with _audio_model_lock:
        if _audio_model is not None and _audio_model_id == model_id:
            return _audio_processor, _audio_model, _audio_device, model_id

        try:
            from transformers import AutoModelForAudioClassification, AutoProcessor

            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForAudioClassification.from_pretrained(model_id)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Could not load audio event model '{model_id}': {exc}"
            ) from exc

        _audio_processor = processor
        _audio_model = model
        _audio_device = device
        _audio_model_id = model_id
        return processor, model, device, model_id


def _audio_as_float(audio, sample_rate):
    if audio.ndim == 2:
        channels = audio.astype(np.float32)
        mono = channels.mean(axis=1)
        channel_rms = np.sqrt(np.mean(np.square(channels), axis=0))
        mono_rms = float(np.sqrt(np.mean(np.square(mono))))
        if channel_rms.size and mono_rms < float(channel_rms.max()) * 0.25:
            mono = channels[:, int(np.argmax(channel_rms))]
        audio = mono
    elif audio.ndim != 1:
        raise ValueError(f"Expected mono or multi-channel audio, received shape {audio.shape}.")
    elif np.issubdtype(audio.dtype, np.integer):
        scale = float(max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max))
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples.")
    if sample_rate != SAMPLE_RATE:
        divisor = int(np.gcd(int(sample_rate), SAMPLE_RATE))
        audio = signal.resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def _classify_audio(audio, processor, model, device, model_id, progress=None):
    if audio.size == 0:
        return []

    sample_window = WINDOW_SECONDS * SAMPLE_RATE
    hop_samples = int(HOP_SECONDS * SAMPLE_RATE)
    duration = audio.size / SAMPLE_RATE
    threshold = float(os.environ.get("AUDIO_EVENT_MIN_CONFIDENCE", "0.25"))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("AUDIO_EVENT_MIN_CONFIDENCE must be between 0 and 1.")

    events = []
    starts = list(range(0, audio.size, hop_samples))
    for window_index, start_sample in enumerate(starts, start=1):
        end_sample = min(start_sample + sample_window, audio.size)
        chunk = audio[start_sample:end_sample]
        if chunk.size < SAMPLE_RATE // 4:
            continue
        inputs = processor(
            chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.sigmoid(logits[0].float())

        selected = torch.where(probabilities >= threshold)[0].tolist()
        start = start_sample / SAMPLE_RATE
        end = min(duration, (start_sample + chunk.size) / SAMPLE_RATE)
        for class_index in selected:
            label = str(model.config.id2label[int(class_index)])
            if label.strip().casefold() in {"silence", "no sound"}:
                continue
            events.append({
                "label": label,
                "start": float(start),
                "end": float(end),
                "duration": float(end - start),
                "confidence": float(probabilities[class_index].item()),
                "evidence": "AudioSet AST multi-label model",
                "model": model_id,
            })

        if progress:
            progress(
                f"Step 5b/5: Classifying sound events: "
                f"{int(window_index * 100 / len(starts))}%"
            )

    merged = []
    events.sort(key=lambda event: (event["label"].casefold(), event["start"]))
    for event in events:
        if (
            merged
            and merged[-1]["label"] == event["label"]
            and event["start"] <= merged[-1]["end"] + 0.25
        ):
            merged[-1]["end"] = max(merged[-1]["end"], event["end"])
            merged[-1]["duration"] = merged[-1]["end"] - merged[-1]["start"]
            merged[-1]["confidence"] = max(
                merged[-1]["confidence"], event["confidence"]
            )
        else:
            merged.append(event)
    merged.sort(key=lambda event: (event["start"], event["end"], event["label"].casefold()))
    return merged


def detect_audio_events(video_path, video_id=None, progress=None):
    """Extract audio and classify overlapping 5-second windows with AudioSet AST."""
    descriptor, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(descriptor)
    try:
        extracted, error = _extract_audio(video_path, wav_path)
        if not extracted:
            return [], error

        try:
            sample_rate, audio = wavfile.read(wav_path)
        except (OSError, ValueError) as exc:
            return [], f"Could not read extracted audio: {exc}"
        if sample_rate <= 0:
            return [], f"Invalid audio sample rate: {sample_rate}."
        try:
            audio = _audio_as_float(audio, sample_rate)
            if audio.size < SAMPLE_RATE // 4:
                return [], "Audio track is too short for sound-event classification."
            processor, model, device, model_id = _load_audio_classifier()
            events = _classify_audio(audio, processor, model, device, model_id, progress)
        except Exception as exc:
            return [], f"Audio event classification failed: {type(exc).__name__}: {exc}"
        return events, None
    finally:
        try:
            os.remove(wav_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"Could not remove temporary audio file {wav_path}: {exc}")
