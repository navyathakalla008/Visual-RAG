# Step 1 — BLIP-2 auxiliary caption generator

import torch
from PIL import Image


class BLIP2Captioner:
    """
    Generates an auxiliary caption from an image using BLIP-2.

    This auxiliary caption is intended to provide additional
    image-related semantic information for the VisualRAG pipeline.
    """

    def __init__(
        self,
        model_name="Salesforce/blip2-opt-2.7b",
        device=None
    ):

        from transformers import (
            Blip2Processor,
            Blip2ForConditionalGeneration
        )

        self.device = (
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        print(
            "Loading BLIP-2 auxiliary caption model..."
        )

        self.processor = (
            Blip2Processor.from_pretrained(
                model_name
            )
        )

        self.model = (
            Blip2ForConditionalGeneration
            .from_pretrained(
                model_name,
                torch_dtype=(
                    torch.float16
                    if self.device == "cuda"
                    else torch.float32
                )
            )
            .to(self.device)
        )

        self.model.eval()

        print(
            "BLIP-2 auxiliary caption model loaded."
        )


    # ========================================================
    # Step 2 — Generate auxiliary caption
    # ========================================================

    @torch.no_grad()
    def generate(
        self,
        image,
        max_new_tokens=40
    ):

        # Accept either a PIL image or an image path.
        if isinstance(image, str):

            image = Image.open(
                image
            ).convert("RGB")

        elif not isinstance(
            image,
            Image.Image
        ):

            raise TypeError(
                "image must be a PIL Image or image path"
            )


        # BLIP-2 receives the image and generates
        # an auxiliary natural-language caption.
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )


        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }


        generated_ids = (
            self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=5,
                repetition_penalty=1.1
            )
        )


        captions = (
            self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
        )


        # Clean the generated caption.
        cleaned = []

        for caption in captions:

            caption = caption.strip()

            if caption:

                caption = (
                    caption[0].upper()
                    + caption[1:]
                )

            if (
                caption
                and not caption.endswith(".")
            ):

                caption += "."

            cleaned.append(
                caption
            )


        return cleaned


    # ========================================================
    # Step 3 — Generate one auxiliary caption
    # ========================================================

    @torch.no_grad()
    def generate_one(
        self,
        image,
        max_new_tokens=40
    ):

        captions = self.generate(
            image,
            max_new_tokens
        )

        if not captions:

            return ""

        return captions[0]