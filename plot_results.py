import argparse
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--r1", type=float, required=True)
p.add_argument("--r5", type=float, required=True)
p.add_argument("--r10", type=float, required=True)
p.add_argument("--output", default="results.png")
args = p.parse_args()

plt.figure(figsize=(7, 5))
plt.bar(["R@1", "R@5", "R@10"], [args.r1, args.r5, args.r10])
plt.ylim(0, 100)
plt.ylabel("Recall (%)")
plt.title("VisualRAG Retrieval Results")
plt.tight_layout()
plt.savefig(args.output, dpi=200)
print("Saved:", args.output)
