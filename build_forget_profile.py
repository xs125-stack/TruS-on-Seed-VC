import torch
import torch.nn.functional as F
import os
import argparse
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype", type=str, required=True)
    parser.add_argument("--forget-summary", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--k", type=float, default=1.0)
    args = parser.parse_args()

    prototype_obj = torch.load(args.prototype, map_location="cpu")
    prototype = prototype_obj["prototype"]
    forget_summary = torch.load(args.forget_summary, map_location="cpu")

    steering_vectors = {}
    cos_sims = {}

    # 1) 三路分别构建 steering vector + cosine similarity
    for key, vec in forget_summary.items():
        if key not in prototype:
            continue

        p = prototype[key]
        s = vec - p
        s = s / (s.norm(p=2) + 1e-8)
        steering_vectors[key] = s

        c = F.cosine_similarity(vec.unsqueeze(0), p.unsqueeze(0), dim=-1).item()
        cos_sims[key] = c

    # 2) 按 branch 分别统计 layer_mean
    # key: (layer, branch) -> list of c over steps
    layer_branch_to_vals = defaultdict(list)
    for (step, layer, branch), c in cos_sims.items():
        layer_branch_to_vals[(layer, branch)].append(c)

    layer_mean = {
        (layer, branch): sum(vals) / len(vals)
        for (layer, branch), vals in layer_branch_to_vals.items()
    }

    # 3) 按 branch 分别算 mu / sigma / tau
    branch_stats = {}
    for branch in sorted(set(b for (_, b) in layer_mean.keys())):
        vals = [v for (layer, b), v in layer_mean.items() if b == branch]
        vals_t = torch.tensor(vals, dtype=torch.float32)
        mu = vals_t.mean().item()
        sigma = vals_t.std(unbiased=False).item()
        tau = mu + args.k * sigma
        branch_stats[branch] = {
            "mu": mu,
            "sigma": sigma,
            "tau": tau,
        }

    # 4) 按 branch 分别选 layer
    selected_layers = set()
    for (layer, branch), cm in layer_mean.items():
        tau = branch_stats[branch]["tau"]
        if cm < tau:
            selected_layers.add((layer, branch))

    # 5) 再按 branch 内 layer_mean 选 step
    selected_points = []
    for (step, layer, branch), c in cos_sims.items():
        if (layer, branch) in selected_layers and c < layer_mean[(layer, branch)]:
            selected_points.append((step, layer, branch))

    profile = {
        "steering_vectors": steering_vectors,
        "selected_points": selected_points,
        "cos_sims": cos_sims,
        "layer_mean": layer_mean,
        "branch_stats": branch_stats,
        "k": args.k,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(profile, args.out)

    print(f"saved profile to {args.out}")
    print(f"num selected points: {len(selected_points)}")

    # 打印每个 branch 的选中数量
    for branch in sorted(branch_stats.keys()):
        cnt = sum(1 for pt in selected_points if pt[2] == branch)
        print(f"branch {branch}: selected_points = {cnt}, stats = {branch_stats[branch]}")