# Step 81 — Automatic image description

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


# ============================================================
# Load BLIP image captioning model
# ============================================================

print("Loading image understanding model...")

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

model.eval()

print("Image understanding model loaded.")


# ============================================================
# Generate image description
# ============================================================

def get_image_contents(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")


    # Simple captioning prompt
    text = "A detailed description of"


    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt"
    ).to(device)


    # Generate caption
    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=5,
            repetition_penalty=1.2
        )


    description = processor.decode(
        output[0],
        skip_special_tokens=True
    )


    return description.strip()