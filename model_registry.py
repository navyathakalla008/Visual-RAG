"""Centralized model registry and configuration for VisualRAG."""
from __future__ import annotations

import os

import torch


def _env(key, default=None):
    return os.environ.get(key, default)


def _bool_env(key, default=False):
    return _env(key, str(default)).lower() in {"1", "true", "yes", "on"}


class ModelConfig:
    def __init__(self, name, feature, default_model, cpu_fallback=None,
                 requires_gpu=False, enabled=True):
        self.name = name
        self.feature = feature
        self.default_model = default_model
        self.cpu_fallback = cpu_fallback
        self.requires_gpu = requires_gpu
        self.enabled = enabled
        self.loaded = False
        self.error = None
        self.instance = None

    def is_available(self, has_gpu):
        if not self.enabled:
            return False
        if self.requires_gpu and not has_gpu:
            return False
        return True


class ModelRegistry:
    def __init__(self):
        self.has_gpu = torch.cuda.is_available()
        self.device = "cuda" if self.has_gpu else "cpu"
        self.models = {}
        self._register_defaults()

    def _register_defaults(self):
        whisper_model = _env("WHISPER_MODEL", "small" if not self.has_gpu else "large-v3")
        self.models["whisper"] = ModelConfig("whisper", "speech_transcription", whisper_model, "small", False, True)

        od_model = _env("OBJECT_DETECTION_MODEL", "yolov8s-world.pt")
        self.models["object_detection"] = ModelConfig(od_model, "object_detection", od_model, "yolo", False, True)

        od_aux_model = _env("OBJECT_DETECTION_AUX_MODEL", "yolo11n.pt")
        if od_aux_model:
            self.models["object_detection_aux"] = ModelConfig(
                od_aux_model,
                "object_detection",
                od_aux_model,
                None,
                False,
                True,
            )

        self.models["grounding_dino"] = ModelConfig("grounding_dino", "open_vocabulary_detection", "grounding_dino", None, True,
                                                    _bool_env("GROUNDING_DINO_ENABLED", False))

        self.models["sam2"] = ModelConfig("sam2", "segmentation_tracking", "sam2", None, True,
                                          _bool_env("SAM2_ENABLED", False))

        action_model = _env("ACTION_DETECTION_MODEL", "videomae")
        self.models["action_detection"] = ModelConfig(action_model, "action_detection", action_model, "clip", False, True)

        audio_model = _env(
            "AUDIO_EVENT_MODEL",
            "MIT/ast-finetuned-audioset-10-10-0.4593",
        )
        self.models["audio_events"] = ModelConfig(
            audio_model,
            "audio_event_detection",
            audio_model,
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            False,
            True,
        )

        self.models["clap"] = ModelConfig("clap", "audio_text_search", "clap", None, False,
                                          _bool_env("CLAP_ENABLED", False))

        sem_model = _env("SEMANTIC_EMBEDDING_MODEL", "clip")
        self.models["semantic_embedding"] = ModelConfig(sem_model, "semantic_search", sem_model, "clip", False, True)

        self.models["bge_m3"] = ModelConfig("bge_m3", "multilingual_text_embedding", "bge-m3", None, False,
                                            _bool_env("BGE_M3_ENABLED", False))

        self.models["video_reasoning"] = ModelConfig("qwen2.5-omni", "multimodal_reasoning", "qwen2.5-omni", None, True,
                                                     _bool_env("VIDEO_REASONING_MODEL_ENABLED", False))

        self.models["translation"] = ModelConfig("nllb-200", "speech_translation", "nllb-200-distilled-600M", None, False, True)

    def get(self, key):
        return self.models.get(key)

    def get_enabled(self, key):
        cfg = self.models.get(key)
        if cfg is None or not cfg.is_available(self.has_gpu):
            return None
        return cfg

    def status_summary(self):
        result = {}
        for key, cfg in self.models.items():
            result[key] = {
                "name": cfg.name,
                "feature": cfg.feature,
                "enabled": cfg.enabled,
                "available": cfg.is_available(self.has_gpu),
                "loaded": cfg.loaded,
                "error": cfg.error,
            }
        return result


_registry = None


def get_registry():
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
PY