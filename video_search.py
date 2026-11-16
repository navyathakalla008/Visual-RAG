"""Search over a video index: phrase, visual concept, semantic, multimodal."""

from __future__ import annotations

import re

import torch.nn.functional as F

from video_pipeline import format_interval, format_timestamp, load_scene_embeddings


def _normalize(text):
    """Unicode-safe normalization that preserves letters from non-Latin scripts."""
    import unicodedata

    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    chars = []
    for ch in text:
        category = unicodedata.category(ch)
        # Keep letters, marks (e.g. vowel signs), and numbers in every script.
        if category[0] in {"L", "M", "N"} or ch in {"'", "’"}:
            chars.append("'" if ch == "’" else ch)
        else:
            chars.append(" ")
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def _tokens(text):
    """Split on whitespace while retaining Unicode tokens; works for no-space scripts via fallback."""
    return [t for t in _normalize(text).split() if t]


def _scene_for_time(index, seconds):
    """Return scene metadata for a timestamp, using half-open intervals except final end."""
    scenes = index.get("scenes") or []
    t = float(seconds or 0.0)
    for scene in scenes:
        if float(scene.get("start", 0.0)) <= t < float(scene.get("end", 0.0)):
            return scene
    if scenes and abs(t - float(scenes[-1].get("end", -1))) <= 0.002:
        return scenes[-1]
    return None


def _result(kind, start, end, title, evidence, score, extra=None):
    item = {
        "kind": kind,
        "start": float(start),
        "end": float(end),
        "label": format_interval(start, end),
        "title": title,
        "evidence": evidence,
        "score": float(score),
    }
    if extra:
        item.update(extra)
    return item


def search_phrase(index, query):
    """Find each occurrence in speech, using word timings where possible and Unicode-safe text matching otherwise."""
    original_query = str(query or "").strip()
    normalized_query = _normalize(original_query)
    if not normalized_query:
        return {"count": 0, "query": original_query, "results": []}

    scenes = index.get("scenes") or []
    hits = []
    seen = set()

    def add_hit(start, end, evidence, score, match_type, occurrence_key=None):
        start = max(0.0, float(start or 0.0))
        end = max(start, float(end if end is not None else start))
        scene = _scene_for_time(index, start)
        # Deduplicate overlapping word-level and segment-level detections.
        key = (round(start, 3), round(end, 3), _normalize(evidence), occurrence_key if occurrence_key is not None else match_type)
        if key in seen:
            return
        seen.add(key)
        extra = {"match_type": match_type}
        if scene:
            extra.update({
                "scene_id": scene.get("id"),
                "scene_start": float(scene.get("start", 0.0)),
                "scene_end": float(scene.get("end", 0.0)),
            })
        hits.append(_result("speech", start, end, "Spoken match", evidence, score, extra))

    # Prefer word-level timestamps for token-sequence matches.
    raw_words = sorted(
        (w for w in (index.get("words") or []) if w.get("start") is not None and w.get("end") is not None),
        key=lambda w: (float(w.get("start", 0.0)), float(w.get("end", 0.0))),
    )
    query_tokens = _tokens(normalized_query)
    cleaned = []
    for word in raw_words:
        normalized_word = _normalize(word.get("word") or word.get("text") or "")
        # A transcript word can contain punctuation; normally it maps to one token.
        parts = normalized_word.split()
        for part in parts:
            cleaned.append({
                "word": part,
                "start": float(word["start"]),
                "end": float(word["end"]),
                "raw": str(word.get("word") or word.get("text") or ""),
            })

    if query_tokens and cleaned:
        n = len(query_tokens)
        for i in range(len(cleaned) - n + 1):
            window = [cleaned[i + j]["word"] for j in range(n)]
            if window == query_tokens:
                start, end = cleaned[i]["start"], cleaned[i + n - 1]["end"]
                scene = _scene_for_time(index, start)
                context = " ".join(w["raw"] for w in cleaned[i:i+n]).strip() or " ".join(window)
                add_hit(start, end, context, 1.0, "exact_word_timing")

    # Transcript fallback: find ALL repeated occurrences, not just one hit per segment.
    # This also supports scripts where whitespace tokenization is not suitable.
    precise_in_segments = [
        hit for hit in hits
        if hit.get("match_type") == "exact_word_timing"
    ]
    for seg in sorted((index.get("transcript") or []), key=lambda x: float(x.get("start", 0.0))):
        segment_start = float(seg.get("start", 0.0))
        segment_end = float(seg.get("end", segment_start))
        if any(
            segment_start - 0.002 <= hit["start"] <= segment_end + 0.002
            or segment_start - 0.002 <= hit["end"] <= segment_end + 0.002
            for hit in precise_in_segments
        ):
            continue
        raw_text = str(seg.get("text") or "")
        normalized_text = _normalize(raw_text)
        if not normalized_text:
            continue
        # Match normalized phrase boundaries. For a single token in space-delimited scripts,
        # enforce token boundaries; for no-space scripts, permit substring matching.
        if " " in normalized_query:
            pattern = re.compile(r"(?<!\w)" + re.escape(normalized_query) + r"(?!\w)", re.UNICODE)
        else:
            pattern = re.compile(r"(?<!\w)" + re.escape(normalized_query) + r"(?!\w)", re.UNICODE)
            if not pattern.search(normalized_text):
                pattern = re.compile(re.escape(normalized_query), re.UNICODE)
        for match in pattern.finditer(normalized_text):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            segment_duration = max(0.0, end - start)
            if segment_duration > 0 and len(normalized_text) > 0:
                match_start_ratio = match.start() / len(normalized_text)
                match_end_ratio = match.end() / len(normalized_text)
                segment_start = start
                start = segment_start + segment_duration * match_start_ratio
                end = segment_start + segment_duration * match_end_ratio
            add_hit(start, end, raw_text, 0.85, "transcript_segment_timing", occurrence_key=(id(seg), match.start(), match.end()))

    hits.sort(key=lambda item: (item["start"], item["end"]))
    # Remove segment-level fallback hits that duplicate a more precise word-level match
    # within the same segment, while preserving genuinely repeated matches with distinct times.
    precise = [h for h in hits if h.get("match_type") == "exact_word_timing"]
    filtered = []
    for hit in hits:
        if hit.get("match_type") == "transcript_segment_timing" and any(
            p["start"] <= hit["end"] + 0.002 and p["end"] >= hit["start"] - 0.002
            for p in precise
        ):
            continue
        filtered.append(hit)

    # If the word or phrase is visible but not spoken, return visual timeline
    # occurrences instead of an empty result. This keeps Word/Phrase useful for
    # words related to the video's content while speech matches remain preferred.
    if not filtered:
        visual_result = search_visual(index, original_query)
        for visual_hit in visual_result.get("results") or []:
            visual_hit["kind"] = "visual"
            visual_hit["match_type"] = "visual_concept_fallback"
            visual_hit["match_source"] = "visual"
            visual_hit["title"] = "Visual occurrence"
            filtered.append(visual_hit)

    for occurrence_number, hit in enumerate(filtered, start=1):
        hit["occurrence_number"] = occurrence_number

    return {
        "count": len(filtered),
        "query": original_query,
        "results": filtered,
    }

def _concept_text(scene):
    parts = list(scene.get("objects") or [])
    parts.extend(scene.get("actions") or [])
    parts.extend(scene.get("relationships") or [])
    for items in (scene.get("contents") or {}).values():
        parts.extend(items or [])
    parts.append(scene.get("transcript") or "")
    return _normalize(" ".join(str(p) for p in parts))


def search_visual(index, query):
    """Find things that appear on screen, even if they were never spoken."""
    query = _normalize(query)
    if not query:
        return {"count": 0, "query": query, "results": []}

    concepts = index.get("concepts") or {}
    hits = []
    seen = set()

    for name, occurrences in concepts.items():
        if query in name or name in query:
            for occ in occurrences:
                key = (occ["scene_id"], round(occ["start"], 2), name)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    _result(
                        "visual",
                        occ["start"],
                        occ["end"],
                        f"Visual: {name}",
                        occ.get("category") or "Visual concept",
                        0.95 if name == query else 0.8,
                        extra={"scene_id": occ.get("scene_id")},
                    )
                )

    if not hits:
        for scene in index.get("scenes") or []:
            blob = _concept_text(scene)
            if query in blob:
                hits.append(
                    _result(
                        "visual",
                        scene["start"],
                        scene["end"],
                        f"Scene {scene['id']}",
                        scene.get("transcript") or ", ".join(scene.get("objects") or []),
                        0.7,
                        extra={"scene_id": scene["id"]},
                    )
                )

    hits.sort(key=lambda x: (-x["score"], x["start"]))
    return {"count": len(hits), "query": query, "results": hits}


def _keyword_overlap(query_tokens, text):
    if not query_tokens:
        return 0.0
    hay = set(_tokens(text))
    if not hay:
        return 0.0
    hits = sum(1 for t in query_tokens if t in hay)
    return hits / max(1, len(query_tokens))


def search_semantic(index, query, vision):
    """Natural-language questions like 'When does the person start talking on the phone?'."""
    query_text = query.strip()
    tokens = _tokens(query_text)
    if not tokens:
        return {"count": 0, "query": query_text, "results": []}

    get_text_embeddings = vision.get("get_text_embeddings")
    embeddings = load_scene_embeddings(
        index["video_id"],
        device=str(vision.get("device") or "cpu"),
    )

    query_vec = None
    if get_text_embeddings is not None:
        try:
            query_vec = get_text_embeddings([query_text], normalize=True)
        except Exception:
            query_vec = None

    results = []
    scenes = index.get("scenes") or []
    for i, scene in enumerate(scenes):
        visual = 0.0
        if query_vec is not None and embeddings is not None and i < embeddings.shape[0]:
            scene_vec = F.normalize(
                embeddings[i].float().unsqueeze(0).to(query_vec.device),
                dim=-1,
            )
            visual = float((query_vec @ scene_vec.T)[0, 0].item())

        blob = _concept_text(scene)
        overlap = _keyword_overlap(tokens, blob)
        action_hit = max(
            (
                _keyword_overlap(tokens, action)
                for action in scene.get("actions") or [""]
            ),
            default=0.0,
        )
        speech_hit = _keyword_overlap(tokens, scene.get("transcript") or "")
        score = 0.55 * visual + 0.2 * overlap + 0.15 * action_hit + 0.1 * speech_hit
        evidence_parts = []
        if scene.get("actions"):
            evidence_parts.append("Actions: " + ", ".join(scene["actions"][:4]))
        if scene.get("objects"):
            evidence_parts.append("Objects: " + ", ".join(scene["objects"][:6]))
        if scene.get("transcript"):
            evidence_parts.append(scene["transcript"][:180])
        results.append(
            _result(
                "semantic",
                scene["start"],
                scene["end"],
                f"Scene {scene['id']}",
                " · ".join(evidence_parts) or "Scene match",
                score,
                extra={
                    "scene_id": scene["id"],
                    "visual_score": visual,
                    "speech_score": speech_hit,
                },
            )
        )

    for action in index.get("actions") or []:
        hit = _keyword_overlap(tokens, action.get("label") or "")
        if hit >= 0.34:
            results.append(
                _result(
                    "action",
                    action["start"],
                    action["end"],
                    action["label"],
                    "Detected activity",
                    0.45 + 0.5 * hit,
                )
            )

    results.sort(key=lambda x: (-x["score"], x["start"]))
    results = [r for r in results if r["score"] >= 0.22][:12]
    return {"count": len(results), "query": query_text, "results": results}


def search_multimodal(index, query, vision):
    """Combine visual, speech, and temporal scores."""
    phrase = search_phrase(index, query)
    visual = search_visual(index, query)
    semantic = search_semantic(index, query, vision)

    combined = []
    buckets = [
        (phrase["results"], 1.05),
        (visual["results"], 1.0),
        (semantic["results"], 1.0),
    ]
    seen = []
    for rows, weight in buckets:
        for row in rows:
            score = row["score"] * weight
            merged = False
            for existing in seen:
                overlap = min(existing["end"], row["end"]) - max(
                    existing["start"], row["start"]
                )
                span = max(existing["end"], row["end"]) - min(
                    existing["start"], row["start"]
                )
                if span > 0 and overlap / span > 0.55:
                    existing["score"] = max(existing["score"], score)
                    if row["kind"] not in existing["kind"]:
                        existing["kind"] = existing["kind"] + "+" + row["kind"]
                    if row["evidence"] and row["evidence"] not in existing["evidence"]:
                        existing["evidence"] = (
                            existing["evidence"] + " · " + row["evidence"]
                        )[:320]
                    merged = True
                    break
            if not merged:
                item = dict(row)
                item["score"] = score
                seen.append(item)

    seen.sort(key=lambda x: (-x["score"], x["start"]))
    combined = seen[:16]
    return {
        "count": len(combined),
        "query": query,
        "results": combined,
        "phrase_count": phrase["count"],
        "visual_count": visual["count"],
    }


def run_search(index, query, mode, vision):
    mode = (mode or "multimodal").lower()
    if mode == "phrase":
        payload = search_phrase(index, query)
    elif mode == "visual":
        payload = search_visual(index, query)
    elif mode == "semantic":
        payload = search_semantic(index, query, vision)
    else:
        payload = search_multimodal(index, query, vision)
    payload["mode"] = mode
    payload["formatted_duration"] = format_timestamp(index.get("duration") or 0)
    return payload
