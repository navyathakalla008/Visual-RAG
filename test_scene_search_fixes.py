"""Lightweight regression tests for scene timelines and multilingual speech search."""
from video_pipeline import _normalize_scene_timeline, validate_scene_timeline
from video_search import search_phrase


def test_scene_timeline_contiguous_and_chronological():
    raw = [
        {"id": 9, "start": 0.0, "end": 3.25, "duration": 3.25, "label": "restaurant"},
        {"id": 3, "start": 3.25, "end": 7.1, "duration": 3.85, "label": "house"},
        {"id": 7, "start": 7.1, "end": 10.0, "duration": 2.9, "label": "restaurant"},
    ]
    scenes = _normalize_scene_timeline(raw, 10.0)
    assert validate_scene_timeline(scenes, 10.0)
    assert [s["id"] for s in scenes] == [1, 2, 3]
    assert scenes[0]["start"] == 0.0
    assert abs(scenes[-1]["end"] - 10.0) < 0.002
    assert scenes[0]["label"] == "restaurant" and scenes[-1]["label"] == "restaurant"
    assert len(scenes) == 3, "Non-adjacent repeated locations must remain separate scenes"


def test_multilingual_word_phrase_and_scene_mapping():
    index = {
        "duration": 12.0,
        "scenes": [
            {"id": 1, "start": 0.0, "end": 5.0, "duration": 5.0},
            {"id": 2, "start": 5.0, "end": 12.0, "duration": 7.0},
        ],
        "words": [
            {"word": "Phone", "start": 1.0, "end": 1.3},
            {"word": "phone", "start": 2.0, "end": 2.3},
            {"word": "అమ్మ", "start": 6.0, "end": 6.5},
        ],
        "transcript": [
            {"text": "Phone phone here", "start": 0.8, "end": 2.5},
            {"text": "అమ్మ వచ్చారు", "start": 6.0, "end": 7.0},
        ],
    }
    phone = search_phrase(index, "phone")
    assert phone["count"] == 2
    assert [r["scene_id"] for r in phone["results"]] == [1, 1]
    telugu = search_phrase(index, "అమ్మ")
    assert telugu["count"] == 1
    assert telugu["results"][0]["scene_id"] == 2
    phrase = search_phrase(index, "phone phone")
    assert phrase["count"] == 1
    assert search_phrase(index, "not present")["count"] == 0


if __name__ == "__main__":
    test_scene_timeline_contiguous_and_chronological()
    test_multilingual_word_phrase_and_scene_mapping()
    print("All lightweight scene/search regression tests passed.")
