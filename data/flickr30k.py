# Step 10 — Use the official Flickr30K train/val/test splits

import os
import csv
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset


class Flickr30KDataset(Dataset):

    def __init__(self, root, transform=None, split="train", max_samples=None):

        self.root = root
        self.transform = transform
        self.items = []

        # Read captions
        caption_file = os.path.join(root, "captions.txt")

        groups = defaultdict(list)

        with open(caption_file, "r", encoding="utf-8") as f:

            reader = csv.DictReader(f)

            for row in reader:

                name = row["image"].strip()
                caption = row["caption"].strip()

                groups[name].append(caption)


        # Read official split file
        split_file = os.path.join(
            root,
            f"{split}.txt"
        )

        if not os.path.exists(split_file):

            raise FileNotFoundError(
                f"Missing split file: {split_file}"
            )


        with open(split_file, "r", encoding="utf-8") as f:

            names = [
                line.strip()
                for line in f
                if line.strip()
            ]


        # Add .jpg if split file does not contain extension
        names = [
            name if name.lower().endswith(".jpg")
            else name + ".jpg"
            for name in names
        ]


        # Build image-caption pairs
        image_dir = os.path.join(
            root,
            "Images"
        )

        for name in names:

            if name not in groups:
                continue

            image_path = os.path.join(
                image_dir,
                name
            )

            if not os.path.exists(image_path):
                continue

            for caption in groups[name]:

                self.items.append(
                    (name, caption)
                )


        # Optional small dataset for testing
        if max_samples is not None:

            self.items = self.items[:max_samples]


        print(
            f"Flickr30K {split}: "
            f"{len(set(name for name, _ in self.items))} images, "
            f"{len(self.items)} image-caption pairs"
        )


    def __len__(self):

        return len(self.items)


    def __getitem__(self, idx):

        name, caption = self.items[idx]

        image_path = os.path.join(
            self.root,
            "Images",
            name
        )

        image = Image.open(
            image_path
        ).convert("RGB")


        if self.transform:

            image = self.transform(image)


        return image, caption, name