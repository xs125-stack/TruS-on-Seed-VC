import torch
import glob
import os
import argparse

def summary_mean(summary_files):
    acc = {}
    counts = {}

    for path in summary_files:
        summary = torch.load(path, map_location="cpu")
        for key, vec in summary.items():
            if key not in acc:
                acc[key] = vec.clone()
                counts[key] = 1
            else:
                acc[key] += vec
                counts[key] += 1

    prototype = {k: acc[k] / counts[k] for k in acc}
    return prototype

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if len(files) == 0:
        raise ValueError("No summary files found.")

    prototype = summary_mean(files)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save({"prototype": prototype, "files": files}, args.out)
    print(f"saved prototype to {args.out}")
    print(f"used {len(files)} summary files")
    first_key = next(iter(prototype))
    print(f"first key: {first_key}, shape: {prototype[first_key].shape}")