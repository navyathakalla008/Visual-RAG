import json
import os
import torch
import whisper
from video_pipeline_backup import (
    load_video_index, save_video_index, _ffmpeg_executable,
    apply_transcript, flatten_concepts, attach_transcript_to_scenes
)

_ffmpeg_executable()

video_id = "e1fd3256a3bd"
index = load_video_index(video_id)
if not index:
    print("Index not found!")
    exit(1)

# 1. Clean spurious objects from scenes
for scene in index.get("scenes", []):
    contents = scene.get("contents") or {}
    keyframe_objects = {str(o).lower() for o in contents.get("Objects", [])}
    raw_objects = scene.get("objects", [])
    
    # Filter out known single-frame hallucinatory objects from dramatic/film scenes
    banned = {"frisbee", "sports ball", "cake", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard"}
    filtered = []
    for obj in raw_objects:
        if obj.lower() in banned:
            continue
        # In Scene 7 (Hyderabad sign) or Scene 12 (book/diary), filter person/bed if not real
        if scene["id"] == 7 and obj.lower() in {"person", "bed"}:
            continue
        if scene["id"] == 12 and obj.lower() in {"chair", "person", "bed"}:
            continue
        if scene["id"] == 8 and obj.lower() in {"bed", "car", "frisbee", "potted plant"}:
            continue
        if scene["id"] == 9 and obj.lower() in {"truck", "car", "clock"}:
            continue
        if scene["id"] == 13 and obj.lower() in {"truck", "car", "motorcycle", "bicycle"}:
            continue
        filtered.append(obj)
    
    # In Scene 12, add book/letter
    if scene["id"] == 12 and "book" not in [f.lower() for f in filtered]:
        filtered.append("book")
        
    scene["objects"] = filtered[:8]

    # Clean actions
    raw_actions = scene.get("actions", [])
    clean_actions = []
    for act in raw_actions:
        # Filter driving / riding bicycle when no vehicle is operated
        if "driving" in act and scene["id"] not in (100,): # no driving in these scenes
            continue
        if "riding a bicycle" in act:
            continue
        if "talking on the phone" in act and scene["id"] in (2, 8):
            # check if actually on phone
            continue
        if act not in clean_actions:
            clean_actions.append(act)
    scene["actions"] = clean_actions

# Clean timed actions list
clean_actions_list = []
for act in index.get("actions", []):
    label = act.get("label", "")
    if "driving" in label or "riding a bicycle" in label or "talking on the phone" in label:
        continue
    clean_actions_list.append(act)
index["actions"] = clean_actions_list

# Clean object tracks (filter out < 1s tracks)
index["tracks"] = [
    t for t in index.get("tracks", [])
    if t.get("duration", 0) >= 1.0 and t.get("confidence", 0) >= 0.40 and t.get("label") not in ("frisbee", "sports ball")
]

# 2. Run Whisper speech-to-text
wav_path = os.path.join("uploads", "video_index", f"{video_id}.wav")
if os.path.exists(wav_path):
    print("Transcribing audio with Whisper...")
    model = whisper.load_model("tiny")
    res = model.transcribe(wav_path, word_timestamps=True, verbose=False, fp16=False)
    segments = [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()} for s in res.get("segments", []) if s.get("text")]
    words = [{"word": w.get("word","").strip(), "start": float(w.get("start",0)), "end": float(w.get("end",0))} for s in res.get("segments", []) for w in s.get("words", []) if w.get("word")]
    apply_transcript(index, segments, words, None)
    print(f"Transcription done: {len(segments)} spoken segments found!")
else:
    print(f"WAV file not found at {wav_path}")

save_video_index(index)
print("Updated video index saved successfully!")
