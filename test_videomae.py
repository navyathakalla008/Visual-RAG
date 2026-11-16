import cv2
import torch
import numpy as np
import time
import psutil

from PIL import Image
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
)

VIDEO = r"C:\Users\HP\Downloads\WhatsApp Video 2026-10-12 at 11.35.11 AM.mp4"
MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"

torch.set_num_threads(2)


def main():
    print("Loading VideoMAE...")

    processor = VideoMAEImageProcessor.from_pretrained(MODEL)
    model = VideoMAEForVideoClassification.from_pretrained(MODEL)
    model.eval()

    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + VIDEO)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # VideoMAE expects a sequence of frames.
    positions = np.linspace(0, total - 1, 16, dtype=int)
    frames = []

    for position in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame = cap.read()

        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()

    if len(frames) != 16:
        raise RuntimeError(f"Expected 16 frames; received {len(frames)}")

    print("Frames sampled:", len(frames))
    print("Running inference on CPU...")

    inputs = processor(frames, return_tensors="pt")

    start = time.time()

    with torch.inference_mode():
        logits = model(**inputs).logits[0]

    probabilities = torch.softmax(logits.float(), dim=-1)
    scores, indices = torch.topk(probabilities, 5)

    print("\nTOP 5 ACTION PREDICTIONS")
    print("-" * 40)

    for rank, (score, index) in enumerate(
        zip(scores.tolist(), indices.tolist()), 1
    ):
        label = model.config.id2label[index]
        print(f"{rank}. {label}: {score * 100:.2f}%")

    print("-" * 40)
    print(f"Inference time: {time.time() - start:.2f} seconds")
    print(
        "Available RAM:",
        round(psutil.virtual_memory().available / 1024**3, 2),
        "GB",
    )


if __name__ == "__main__":
    main()