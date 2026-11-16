"""Translation service for multilingual transcript translation."""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
import urllib.request

NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "uploads", "translation_cache")

# Map our dropdown codes to NLLB FLORES-200 language codes
LANG_CODE_MAP = {
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

WHISPER_TO_NLLB_MAP = dict(LANG_CODE_MAP)

GOOGLE_LANG_MAP = {
    "zh": "zh-CN",
    "fil": "tl",
    "he": "iw",
    "fa": "fa",
}

_tokenizer = None
_model = None
_model_error = None


def _ensure_model():
    global _tokenizer, _model, _model_error
    if _model is not None:
        return True
    if _model_error is not None:
        return False
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
        import torch
        _model.to(torch.device("cpu"))
        _model.eval()
        return True
    except Exception as exc:
        _model_error = str(exc)
        return False


def normalize_language_code(lang_code):
    """Normalize Whisper and UI language codes before translation."""
    raw = str(lang_code or "").strip().lower()
    aliases = {
        "zh-cn": "zh",
        "zh-tw": "zh",
        "iw": "he",
        "tl": "fil",
        "nb": "no",
        "jp": "ja",
        "kr": "ko",
        "auto": "unknown",
        "detect": "unknown",
        "none": "unknown",
    }
    if raw in aliases:
        return aliases[raw]
    code = raw.replace("_", "-").split("-")[0]
    return aliases.get(code, code)


def is_supported(lang_code):
    """Check if a UI language code is supported for translation."""
    return normalize_language_code(lang_code) in LANG_CODE_MAP


def _google_lang(lang_code):
    code = normalize_language_code(lang_code)
    if code in ("", "unknown"):
        return "auto"
    return GOOGLE_LANG_MAP.get(code, code)


def _web_translate(text, target_lang, source_lang=None):
    if not text or not str(text).strip():
        return text
    def request_translation(query):
        params = urllib.parse.urlencode({
            "client": "gtx",
            "sl": _google_lang(source_lang),
            "tl": _google_lang(target_lang),
            "dt": "t",
            "q": query,
        })
        url = "https://translate.googleapis.com/translate_a/single?" + params
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 VisualRAG/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not data or not data[0]:
            raise RuntimeError("Empty translation response.")
        return "".join(part[0] for part in data[0] if part and part[0])

    result = request_translation(str(text))
    if result.strip() == str(text).strip() and str(text).strip() != str(text).strip().lower():
        result = request_translation(str(text).lower())
    return result


def _nllb_translate(text, target_lang, source_lang):
    tgt_code = LANG_CODE_MAP[target_lang]
    src_code = WHISPER_TO_NLLB_MAP[source_lang]
    _tokenizer.src_lang = src_code
    try:
        inputs = _tokenizer(text, src_lang=src_code, return_tensors="pt", truncation=True, max_length=512)
    except TypeError:
        inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    tgt_token_id = _tokenizer.convert_tokens_to_ids(tgt_code)
    unk_id = getattr(_tokenizer, "unk_token_id", None)
    if tgt_token_id == unk_id:
        raise ValueError(f"NLLB does not support target language {target_lang}.")
    import torch
    with torch.inference_mode():
        outputs = _model.generate(**inputs, forced_bos_token_id=tgt_token_id, max_length=512)
    return _tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


def translate_text(text, target_lang, source_lang="en"):
    """Translate one transcript segment and preserve its exact text boundaries."""
    if not text or not str(text).strip():
        return text
    target_lang = normalize_language_code(target_lang)
    source_lang = normalize_language_code(source_lang)
    if not is_supported(target_lang):
        raise ValueError(f"Unsupported target language: {target_lang}")
    if source_lang == target_lang:
        return text
    errors = []
    try:
        web_result = _web_translate(text, target_lang, source_lang)
        if web_result.strip() != text.strip():
            return web_result
    except Exception as exc:
        errors.append(f"web: {exc}")
    if source_lang in WHISPER_TO_NLLB_MAP and _ensure_model():
        try:
            return _nllb_translate(text, target_lang, source_lang)
        except Exception as exc:
            errors.append(f"nllb: {exc}")
    raise RuntimeError("Translation failed (" + "; ".join(errors) + ")")


def detect_source_language(segments):
    """Infer source language from Whisper metadata, then unambiguous scripts."""
    if segments:
        codes = {
            normalize_language_code(seg.get("language"))
            for seg in segments
            if normalize_language_code(seg.get("language")) not in ("", "unknown", "none", "auto", "detect")
        }
        if len(codes) == 1:
            return codes.pop()
    sample = " ".join(str(seg.get("text", "")) for seg in (segments or [])[:20])
    script_ranges = (
        (("\u0c00", "\u0c7f"), "te"), (("\u0900", "\u097f"), "hi"),
        (("\u0b80", "\u0bff"), "ta"), (("\u0c80", "\u0cff"), "kn"),
        (("\u0d00", "\u0d7f"), "ml"), (("\u0600", "\u06ff"), "ar"),
        (("\u0980", "\u09ff"), "bn"), (("\u0a80", "\u0aff"), "gu"),
        (("\u0a00", "\u0a7f"), "pa"), (("\u0b00", "\u0b7f"), "or"),
        (("\u3040", "\u30ff"), "ja"), (("\u4e00", "\u9fff"), "zh"),
        (("\uac00", "\ud7af"), "ko"), (("\u0400", "\u04ff"), "ru"),
        (("\u0370", "\u03ff"), "el"), (("\u0590", "\u05ff"), "he"),
        (("\u0e00", "\u0e7f"), "th"),
    )
    for (start, end), code in script_ranges:
        if any(start <= ch <= end for ch in sample):
            return code
    if any(ch.isalpha() and ch.isascii() for ch in sample):
        return "en"
    return "unknown"


def _split_translated_batch(translated, count):
    parts = re.split(r"\s*⟦S⟧\s*", translated)
    parts = [part.strip() for part in parts if part.strip() or count == 1]
    if len(parts) == count:
        return parts
    parts = [part.strip() for part in translated.split("\n\n") if part.strip()]
    if len(parts) == count:
        return parts
    raise ValueError("Could not split batched translation.")


def _web_translate_many(texts, target_lang, source_lang):
    nonempty = [(idx, text) for idx, text in enumerate(texts) if str(text or "").strip()]
    if not nonempty:
        return list(texts)
    delimiter = "\n⟦S⟧\n"
    combined = delimiter.join(text for _, text in nonempty)
    if len(combined) <= 3500:
        translated = _web_translate(combined, target_lang, source_lang)
        parts = _split_translated_batch(translated, len(nonempty))
        result = list(texts)
        for (idx, _), part in zip(nonempty, parts):
            result[idx] = part
        return result
    result = list(texts)
    chunk, chunk_items = [], []
    chunk_len = 0

    def flush():
        nonlocal chunk, chunk_items, chunk_len
        if not chunk:
            return
        translated = _web_translate(delimiter.join(chunk), target_lang, source_lang)
        parts = _split_translated_batch(translated, len(chunk))
        for (idx, _), part in zip(chunk_items, parts):
            result[idx] = part
        chunk, chunk_items, chunk_len = [], [], 0

    for idx, text in nonempty:
        extra = len(text) + len(delimiter)
        if chunk and chunk_len + extra > 3500:
            flush()
        chunk.append(text)
        chunk_items.append((idx, text))
        chunk_len += extra
    flush()
    return result


def translate_segments(segments, target_lang, source_lang=None):
    """Translate every transcript segment, preserving timestamps."""
    target_lang = normalize_language_code(target_lang)
    source_lang = normalize_language_code(source_lang)
    if not is_supported(target_lang):
        raise ValueError(f"Unsupported target language: {target_lang}")
    texts = [str(seg.get("text", "") or "") for seg in segments]
    translated_texts = None
    if source_lang == target_lang:
        translated_texts = texts
    else:
        try:
            translated_texts = _web_translate_many(texts, target_lang, source_lang)
        except Exception:
            translated_texts = None
        if translated_texts is None:
            translated_texts = []
            for text in texts:
                translated_texts.append(translate_text(text, target_lang, source_lang))
        else:
            for idx, (text, translated) in enumerate(zip(texts, translated_texts)):
                if translated.strip() == text.strip() and source_lang != target_lang:
                    try:
                        individual = _web_translate(text, target_lang, source_lang)
                        if individual.strip() != text.strip():
                            translated_texts[idx] = individual
                            continue
                    except Exception:
                        pass
                    if source_lang in WHISPER_TO_NLLB_MAP and _ensure_model():
                        try:
                            translated_texts[idx] = _nllb_translate(text, target_lang, source_lang)
                        except Exception:
                            pass
    result = []
    for seg, translated in zip(segments, translated_texts):
        result.append({
            **seg,
            "translated_text": translated,
        })
    return result


def _cache_key(video_id, transcript_hash, target_lang, source_lang):
    return hashlib.sha256(
        f"v6:{video_id}:{transcript_hash}:{source_lang}:{target_lang}".encode()
    ).hexdigest()


def _transcript_hash(segments):
    payload = [{"start": seg.get("start"), "text": seg.get("text")} for seg in segments]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def get_cached_translation(video_id, segments, target_lang, source_lang=None):
    """Return cached translation if available."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(video_id, _transcript_hash(segments), target_lang, source_lang or "unknown")
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
    return None


def cache_translation(video_id, segments, target_lang, translated_segments, source_lang=None):
    """Cache a completed translation."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(video_id, _transcript_hash(segments), target_lang, source_lang or "unknown")
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(translated_segments, f, ensure_ascii=False)
    except OSError:
        pass


def translate_transcript(video_id, segments, target_lang, source_lang=None):
    """Translate transcript segments, using cache if available."""
    if not is_supported(target_lang):
        raise ValueError(f"Unsupported target language: {target_lang}")
    cached = get_cached_translation(video_id, segments, target_lang, source_lang)
    if cached is not None:
        return cached
    translated = translate_segments(segments, target_lang, source_lang)
    cache_translation(video_id, segments, target_lang, translated, source_lang)
    return translated
