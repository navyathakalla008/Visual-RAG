import os
import csv
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration


DATA_ROOT = r"C:\Users\HP\Downloads\archive"
OUTPUT_FILE = os.path.join(DATA_ROOT, "auxiliary_captions.csv")

# BLIP-2 checkpoint.
# The paper specifies BLIP-2, but does not specify the exact checkpoint.
MODEL_NAME = "Salesforce/blip2-opt-2.7b"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)
print("Loading BLIP-2...")

processor = Blip2Processor.from_pretrained(MODEL_NAME)

model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

model.to(device)
model.eval()

image_dir = os.path.join(DATA_ROOT, "Images")

image_names = ["984950.jpg"]

image_names.sort()

print("Images:", len(image_names))

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(["image", "auxiliary_caption"])

    for name in tqdm(image_names):

        image_path = os.path.join(image_dir, name)

        try:
            image = Image.open(image_path).convert("RGB")

            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

            with torch.no_grad():

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=40
                )

            caption = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()

            writer.writerow([
                name,
                caption
            ])

        except Exception as e:

            print(
                f"ERROR {name}: {e}"
            )

print()
print("Finished.")
print("Saved:", OUTPUT_FILE)