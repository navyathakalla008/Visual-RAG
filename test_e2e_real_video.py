"""Temporary end-to-end validation against a real 10-minute video segment."""
import json
import os
import shutil
import subprocess
import time

import app
import cv2
from video_pipeline import analyze_video, load_scene_embeddings, load_video_index

SOURCE = r"uploads\videos\b8511dd4cd31_videoplayback.mp4"
CLIP = r"uploads\videos\e2e_10min_source.mp4"
VIDEO_ID = "e2e10min001"


def make_clip():
    ffmpeg = os.path.abspath(__import__("imageio_ffmpeg").get_ffmpeg_exe())
    command = [ffmpeg, "-y", "-ss", "120", "-i", SOURCE, "-t", "600", "-vf", "scale=640:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-movflags", "+faststart", CLIP]
    subprocess.run(command, check=True, capture_output=True)


def valid_timeline(index):
    scenes = index["scenes"]
    assert scenes and scenes[0]["start"] == 0.0
    assert abs(scenes[-1]["end"] - index["duration"]) < 0.002
    for previous, scene in zip(scenes, scenes[1:]):
        assert abs(previous["end"] - scene["start"]) < 0.002
        assert abs((scene["end"] - scene["start"]) - scene["duration"]) < 0.002
    for scene in scenes:
        path = os.path.join("uploads", "video_frames", scene["keyframe"])
        assert os.path.isfile(path) and os.path.getsize(path) > 100
    return True


def main():
    started = time.monotonic()
    make_clip()
    shutil.copyfile(CLIP, os.path.join("uploads", "videos", f"{VIDEO_ID}_{os.path.basename(CLIP)}"))
    stored = f"{VIDEO_ID}_{os.path.basename(CLIP)}"
    progress_log = []

    def progress(message):
        progress_log.append((round(time.monotonic() - started, 2), message))
        print(progress_log[-1])

    index = analyze_video(
        os.path.join("uploads", "videos", stored),
        VIDEO_ID,
        stored,
        app.video_vision(),
        original_name="e2e_10min_source.mp4",
        progress=progress,
    )
    assert valid_timeline(index)
    assert load_scene_embeddings(VIDEO_ID).shape[0] == len(index["scenes"])
    assert index["tracks"], "No object tracks were produced"
    assert all("start" in track and "end" in track and track["start"] <= track["end"] for track in index["tracks"])
    assert all("start" in action and "end" in action and action["start"] <= action["end"] for action in index["actions"])
    assert all("start" in word and "end" in word for word in index["words"])

    client = app.app.test_client()
    searches = {}
    for query, mode in [("person", "visual"), ("phone", "phrase"), ("person talking on phone", "semantic")]:
        response = client.post("/video/search", json={"video_id": VIDEO_ID, "query": query, "mode": mode})
        assert response.status_code == 200, response.get_data(as_text=True)
        payload = response.get_json()
        searches[f"{mode}:{query}"] = payload
        assert payload["count"] >= 0

    summary = {
        "duration": index["duration"],
        "scene_count": len(index["scenes"]),
        "thumbnail_count": len(index["scenes"]),
        "track_count": len(index["tracks"]),
        "action_count": len(index["actions"]),
        "segment_count": len(index["transcript"]),
        "word_count": len(index["words"]),
        "asr_error": index["asr_error"],
        "saved_index": load_video_index(VIDEO_ID) is not None,
        "searches": {key: value["count"] for key, value in searches.items()},
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "progress_last": progress_log[-1][1],
    }
    print("E2E_SUMMARY=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
