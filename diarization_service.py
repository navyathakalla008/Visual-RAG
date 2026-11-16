"""Speaker diarization service using PyAnnote (CPU-compatible)."""
from __future__ import annotations

import os

_pipeline = None
_pipeline_error = None
_hf_token = None


def set_hf_token(token):
    global _hf_token
    _hf_token = token


def get_diarization_pipeline():
    global _pipeline, _pipeline_error
    if _pipeline is not None:
        return _pipeline
    if _pipeline_error is not None:
        return None
    token = _hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        _pipeline_error = "No HuggingFace token provided. Set HF_TOKEN env var."
        return None
    try:
        import torch
        from pyannote.audio import Pipeline
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
        _pipeline.to(torch.device("cpu"))
        return _pipeline
    except Exception as exc:
        _pipeline_error = str(exc)
        return None


def diarize_audio(audio_path):
    """Run speaker diarization on an audio file. Returns list of {start, end, speaker}."""
    pipe = get_diarization_pipeline()
    if pipe is None:
        return [], _pipeline_error or "Diarization unavailable"

    try:
        diarization = pipe(audio_path)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
                "speaker": speaker,
            })
        return segments, None
    except Exception as exc:
        return [], str(exc)


def is_available():
    return get_diarization_pipeline() is not None