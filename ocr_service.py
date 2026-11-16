"""OCR service using EasyOCR (CPU-compatible)."""
from __future__ import annotations

import os

_ocr_reader = None
_ocr_error = None


def get_ocr_reader(languages=None):
    global _ocr_reader, _ocr_error
    if _ocr_reader is not None:
        return _ocr_reader
    if _ocr_error is not None:
        return None
    try:
        import easyocr
        langs = languages or ["en"]
        _ocr_reader = easyocr.Reader(langs, gpu=False)
        return _ocr_reader
    except Exception as exc:
        _ocr_error = str(exc)
        return None


def extract_text_from_frame(frame, min_confidence=0.5):
    """Extract text from a single frame using EasyOCR."""
    reader = get_ocr_reader()
    if reader is None:
        return [], _ocr_error or "OCR unavailable"
    try:
        results = reader.readtext(frame)
        extracted = []
        for (bbox, text, prob) in results:
            if prob >= min_confidence and text.strip():
                extracted.append({
                    "text": text.strip(),
                    "confidence": round(float(prob), 3),
                    "bbox": [[int(p[0]), int(p[1])] for p in bbox],
                })
        return extracted, None
    except Exception as exc:
        return [], str(exc)


def extract_text_from_frames(frames, min_confidence=0.5, max_frames=5):
    """Extract text from multiple frames, deduplicate."""
    all_text = []
    seen = set()
    for frame in frames[:max_frames]:
        texts, err = extract_text_from_frame(frame, min_confidence)
        if err:
            continue
        for t in texts:
            key = t["text"].lower()
            if key not in seen:
                seen.add(key)
                all_text.append(t)
    return all_text, None


def is_available():
    return get_ocr_reader() is not None