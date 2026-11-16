import torch
import open_clip
from PIL import Image
from ultralytics import YOLO

yolo = YOLO("yolo11n.pt")

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

image_path = r"D:\Flickr30K\Images\1000268201.jpg"
img = Image.open(image_path).convert("RGB")

results = yolo(img, conf=0.25)

candidates = [
    "person",
    "child",
    "girl",
    "boy",
    "bicycle",
    "toy",
    "chair",
    "stairs",
    "house",
    "door",
    "window",
    "toilet",
    "wooden object",
    "playhouse",
    "background"
]

texts = tokenizer(
    [f"a photo of {x}" for x in candidates]
)

print("\nDetection verification")
print("=" * 60)

for i, box in enumerate(results[0].boxes):

    yolo_label = yolo.names[int(box.cls)]
    yolo_conf = float(box.conf)

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = img.crop((x1, y1, x2, y2))

    image_tensor = preprocess(crop).unsqueeze(0)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        text_features = clip_model.encode_text(texts)

        image_features /= image_features.norm(
            dim=-1, keepdim=True
        )
        text_features /= text_features.norm(
            dim=-1, keepdim=True
        )

        scores = (image_features @ text_features.T)[0]

    ranked = sorted(
        zip(candidates, scores.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\nDetection {i + 1}")
    print(f"YOLO label      : {yolo_label}")
    print(f"YOLO confidence : {yolo_conf:.3f}")

    print("\nTop CLIP matches:")
    for label, score in ranked[:5]:
        print(f"  {label:18s} {score:.3f}")

    best_label, best_score = ranked[0]
    second_score = ranked[1][1]
    margin = best_score - second_score

    print(f"\nBest match      : {best_label}")
    print(f"Best score      : {best_score:.3f}")
    print(f"Score margin    : {margin:.3f}")

