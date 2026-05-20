"""
DaoTi V53 Benchmark Evaluation Script
======================================
Reproducible evaluation of the DaoTi V53 foundation model on structured tasks.

Usage:
    python eval_benchmark.py

Outputs:
    - 64-hexagram palace classification accuracy
    - 64-hexagram retrieval top-1 / top-5 accuracy
    - Coherence distribution statistics
    - Per-palace breakdown
"""

import torch
import torch.nn.functional as F
import json
import sys
import os
from inference import (
    load_daoti, predict, compute_coherence, verify_sha256,
    generate_response, GUA_64, BA_GONG, GUA_TRIGRAM, GUA_WUXING,
    find_palace, sparse_expand_input, STATE_DIM,
)

PALACE_NAMES  = ["乾宫","坤宫","震宫","巽宫","坎宫","离宫","艮宫","兑宫"]
LIUQIN_NAMES  = ["父母","兄弟","子孙","妻财","官鬼","空亡"]
LIUSHEN_NAMES = ["青龙","朱雀","勾陈","螣蛇","白虎","玄武"]

def eval_palace_classification(model, device='cpu'):
    correct = 0
    total = 64
    per_palace = {p: {"correct": 0, "total": 0} for p in PALACE_NAMES}
    text_ids = torch.randint(1, 100, (1, 256), dtype=torch.long, device=device)

    for gi in range(64):
        gua_name = GUA_64[gi]
        ground_truth = find_palace(gua_name)
        r = predict(model, text_ids, gua_idx=gi, method='traditional', device=device)
        pred_palace = PALACE_NAMES[r['palace'].argmax().item()]
        per_palace[ground_truth]["total"] += 1
        if pred_palace == ground_truth:
            correct += 1
            per_palace[ground_truth]["correct"] += 1

    accuracy = correct / total
    return {
        "task": "64-Hexagram Palace Classification",
        "accuracy": f"{100*accuracy:.1f}%",
        "correct": correct,
        "total": total,
        "per_palace": {p: f"{v['correct']}/{v['total']}" for p, v in per_palace.items()},
    }

def eval_retrieval(model, device='cpu'):
    with torch.no_grad():
        proto = model.gua_prototype.weight
        proto_n = F.normalize(proto, p=2, dim=-1)

    text_ids = torch.randint(1, 100, (1, 256), dtype=torch.long, device=device)
    top1_correct = 0
    top5_correct = 0
    similarities = []

    for gi in range(64):
        with torch.no_grad():
            text_feat = model.encode_text(text_ids.to(device))
            feat_n = F.normalize(text_feat, p=2, dim=-1)
            sim = torch.mm(feat_n, proto_n.t()).squeeze()
            similarities.append(sim.max().item())
            top1_idx = sim.argmax().item()
            top5_idx = sim.topk(5).indices.tolist()
        if top1_idx == gi:
            top1_correct += 1
        if gi in top5_idx:
            top5_correct += 1

    return {
        "task": "64-Hexagram Prototype Retrieval",
        "top1_accuracy": f"{100*top1_correct/64:.1f}%",
        "top5_accuracy": f"{100*top5_correct/64:.1f}%",
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "total": 64,
        "avg_similarity": f"{sum(similarities)/len(similarities):.4f}",
        "min_similarity": f"{min(similarities):.4f}",
        "max_similarity": f"{max(similarities):.4f}",
    }

def eval_coherence_distribution(model, device='cpu'):
    text_ids = torch.randint(1, 100, (1, 256), dtype=torch.long, device=device)
    coherences = []
    for gi in range(64):
        c = compute_coherence(model, text_ids, gi, device)
        coherences.append(c)

    coherences.sort()
    n = len(coherences)
    return {
        "task": "Coherence Distribution (Self-Calibrating Quality Signal)",
        "mean": f"{sum(coherences)/n:.4f}",
        "median": f"{coherences[n//2]:.4f}",
        "min": f"{min(coherences):.4f}",
        "max": f"{max(coherences):.4f}",
        "p25": f"{coherences[n//4]:.4f}",
        "p75": f"{coherences[3*n//4]:.4f}",
        "below_threshold_0.3": sum(1 for c in coherences if c < 0.3),
        "above_threshold_0.7": sum(1 for c in coherences if c > 0.7),
    }

def eval_random_baseline(device='cpu'):
    import random
    correct = 0
    for gi in range(64):
        ground_truth = find_palace(GUA_64[gi])
        random_palace = random.choice(PALACE_NAMES)
        if random_palace == ground_truth:
            correct += 1
    return {
        "task": "Random Baseline (8-class uniform)",
        "expected_accuracy": "12.5%",
        "sampled_accuracy": f"{100*correct/64:.1f}%",
    }

def main():
    print("=" * 60)
    print("  DaoTi V53 Benchmark Evaluation")
    print("=" * 60)

    if not verify_sha256("yijing_v53_daoti.pt"):
        print("[FAIL] Weight verification failed")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_daoti("yijing_v53_daoti.pt", device=device)

    results = {}

    print("\n[1/4] Random Baseline...")
    results["random_baseline"] = eval_random_baseline(device)
    for k, v in results["random_baseline"].items():
        if k != "task": print(f"  {k}: {v}")

    print("\n[2/4] Palace Classification...")
    results["palace_classification"] = eval_palace_classification(model, device)
    for k, v in results["palace_classification"].items():
        if k != "task": print(f"  {k}: {v}")

    print("\n[3/4] Prototype Retrieval...")
    results["retrieval"] = eval_retrieval(model, device)
    for k, v in results["retrieval"].items():
        if k != "task": print(f"  {k}: {v}")

    print("\n[4/4] Coherence Distribution...")
    results["coherence"] = eval_coherence_distribution(model, device)
    for k, v in results["coherence"].items():
        if k != "task": print(f"  {k}: {v}")

    output_path = "benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
