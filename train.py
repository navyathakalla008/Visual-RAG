import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.flickr30k import Flickr30KDataset
from models.vpke import VPKE
from losses import dual_triplet_loss

def collate(batch):
    images, captions, names = zip(*batch)
    return torch.stack(images), list(captions), list(names)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    model = VPKE(args.clip_model, 512, 256, 4, 6, 10).to(device)
    dataset = Flickr30KDataset(args.data_root, model.clip.preprocess, "train", args.max_train_samples)
    loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=collate, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train()

    for epoch in range(args.epochs):
        total = 0.0
        for images, captions, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = images.to(device)
            out = model(images, captions)
            loss, _, _ = dual_triplet_loss(out["sim_f"], out["sim_c"], 8.0, 1.0, 0.5)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()

        avg = total / max(1, len(loader))
        print(f"Epoch {epoch+1}: loss={avg:.6f}")
        torch.save({"epoch": epoch + 1, "model": model.state_dict()},
                   os.path.join(args.output_dir, "last.pt"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--clip_model", default="ViT-B-16")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--output_dir", default="checkpoints")
    main(p.parse_args())
