import app

from translation_service import _split_translated_batch, translate_segments


def test_batched_translation_split_preserves_segment_count():
    translated = "नमस्ते\n⟦S⟧\nकैसे हैं"
    assert _split_translated_batch(translated, 2) == ["नमस्ते", "कैसे हैं"]


def test_backend_route_returns_translated_segments_with_timestamps():
    client = app.app.test_client()
    response = client.post(
        "/video/translate",
        json={
            "video_id": "testsmall1",
            "language": "hi",
            "source_language": "en",
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["language"] == "hi"
    assert data["source_language"] == "en"
    assert len(data["segments"]) == 2
    assert data["segments"][0]["start"] == 0.0
    assert data["segments"][0]["translated_text"]


def test_translate_segments_preserves_segment_timestamps():
    segments = [
        {"start": 0.0, "end": 1.2, "text": "Hello"},
        {"start": 1.2, "end": 2.4, "text": "Goodbye"},
    ]
    translated = translate_segments(segments, "hi", "en")
    assert [item["start"] for item in translated] == [0.0, 1.2]
    assert [item["end"] for item in translated] == [1.2, 2.4]
    assert all(item["translated_text"] for item in translated)


if __name__ == "__main__":
    test_batched_translation_split_preserves_segment_count()
    test_backend_route_returns_translated_segments_with_timestamps()
    test_translate_segments_preserves_segment_timestamps()
    print("Translation backend regression tests passed.")
