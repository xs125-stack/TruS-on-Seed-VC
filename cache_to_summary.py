import torch
import os
import argparse

def cache_to_summary(cache):
    summary = {}
    eps = 1e-8
    for key, h in cache.items():
        # h: [1, T, H]
        h = h.float()
        h_unit = h / (h.norm(p=2, dim=-1, keepdim=True) + eps)
        vec = h_unit.mean(dim=(0, 1))  # [H]
        summary[key] = vec.cpu()
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    cache = torch.load(args.cache, map_location="cpu")
    summary = cache_to_summary(cache)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(summary, args.out)
    print(f"saved summary to {args.out}")
    print(f"num entries: {len(summary)}")
    first_key = next(iter(summary))
    print(f"first key: {first_key}, shape: {summary[first_key].shape}")