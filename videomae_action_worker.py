"""Isolated VideoMAE Kinetics-400 worker; invoked by video_pipeline.py."""
import contextlib, json, os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
torch.set_num_threads(2)

def main(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV cannot open source video")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or n <= 0:
        cap.release(); raise RuntimeError("Invalid video metadata")
    duration = n / fps
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID, use_safetensors=True)
    model.eval()
    # Predict on overlapping clips. For short clips, use the whole clip.
    clip_len = min(4.0, duration)
    stride = 2.0 if duration > clip_len else clip_len
    starts = list(np.arange(0, max(0.001, duration - clip_len + 0.001), stride))
    if not starts or (duration > clip_len and starts[-1] + clip_len < duration - 0.25):
        starts.append(max(0.0, duration - clip_len))
    starts = sorted(set(round(float(s), 3) for s in starts))
    results = []
    for st in starts:
        en = min(duration, st + clip_len)
        positions = np.linspace(int(st * fps), max(int(st * fps), int((en - 1.0 / fps) * fps)), 16, dtype=int)
        frames = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, int(pos)))
            ok, frame = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if len(frames) < 8:
            continue
        # Repeat nearest available frames if decode glitches leave fewer than 16.
        while len(frames) < 16:
            frames.append(frames[-1].copy())
        frames = frames[:16]
        inputs = processor(frames, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs).logits[0].float()
        probs = torch.softmax(logits, dim=-1)
        top_score, top_idx = torch.max(probs, dim=-1)
        results.append({"start": st, "end": en,
                        "label": model.config.id2label.get(int(top_idx), str(int(top_idx)),),
                        "score": float(top_score)})
    cap.release()
    print(json.dumps({"duration": duration, "clips": results}, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: videomae_action_worker.py VIDEO_PATH")
    main(sys.argv[1])
