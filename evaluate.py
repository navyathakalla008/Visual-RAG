import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.flickr30k import Flickr30KDataset
from models.vpke import VPKE
from metrics import retrieval_metrics, rsum

def collate(batch):
    images, captions, names = zip(*batch)
    return torch.stack(images), list(captions), list(names)

@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VPKE(args.clip_model, 512, 256, 4, 6, 10).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    dataset = Flickr30KDataset(args.data_root, model.clip.preprocess, "test", args.max_test_samples)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)

    imgs, txts = [], []
    for images, captions, _ in tqdm(loader):
        out = model(images.to(device), captions)
        imgs.append(out["image"].cpu())
        txts.append(out["fused_text"].cpu())

    imgs = torch.cat(imgs)
    txts = torch.cat(txts)
    sim = imgs @ txts.T

    i2t = retrieval_metrics(sim)
    t2i = retrieval_metrics(sim.T)

    print("Image-to-Text:", i2t)
    print("Text-to-Image:", t2i)
    print("rSum:", rsum(i2t, t2i))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--clip_model", default="ViT-B-16")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max_test_samples", type=int, default=None)
    main(p.parse_args())
