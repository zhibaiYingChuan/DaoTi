"""
DaoTi V53 Domain Adapter Training Tool
========================================
Train LoRA adapters on your own text data while keeping the DaoTi core frozen.

Usage:
    python train_adapter.py \
        --data_path ./my_domain_data.jsonl \
        --output_path ./my_adapter.pt \
        --domain_name "optics" \
        --method traditional \
        --epochs 10 \
        --batch_size 8 \
        --lr 1e-4

Data format (JSONL, one record per line):
    {"text": "your text here", "domain": "optics"}
    {"text": "another text", "domain": "optics"}

The "domain" field is optional. If omitted, a generic adapter is trained.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

from daoti.inference import (
    load_daoti, verify_sha256,
    GUA_64, BA_GONG, find_palace, GUA_64_DETAIL,
    sparse_expand_input, MAX_SEQ, STATE_DIM,
    METHOD_MAP, PALACE_MAP,
)


def load_tokenizer(tokenizer_path):
    if not os.path.exists(tokenizer_path):
        print(f"[WARN] Tokenizer not found: {tokenizer_path}")
        print("       Text will be tokenized with fallback character-level mapping.")
        return None
    data = torch.load(tokenizer_path, map_location='cpu', weights_only=False)
    return data


def tokenize_text(text, char_to_id, max_seq=256):
    tokens = []
    for ch in text:
        idx = char_to_id.get(ch, 1)
        if idx == 0:
            idx = 1
        tokens.append(idx)
    if len(tokens) > 0 and len(tokens) < max_seq:
        repeated = tokens * (max_seq // len(tokens) + 1)
        tokens = repeated[:max_seq]
    elif len(tokens) < max_seq:
        tokens = tokens + [0] * (max_seq - len(tokens))
    else:
        tokens = tokens[:max_seq]
    return torch.tensor([tokens], dtype=torch.long)


def load_data(data_path):
    records = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if 'text' not in record:
                    continue
                records.append(record)
            except json.JSONDecodeError:
                continue
    return records


def freeze_dao_core(model):
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Frozen: {total - trainable:,} params")
    print(f"  Trainable (LoRA): {trainable:,} params ({100*trainable/total:.2f}%)")


def build_domain_palace_map(domain_name):
    palace_idx = hash(domain_name) % 8
    palace_names = list(PALACE_MAP.keys())
    return palace_names[palace_idx], palace_idx


class AdapterDataset:
    def __init__(self, records, char_to_id, domain_name=None):
        self.items = []
        for rec in records:
            text = rec.get('text', '')
            domain = rec.get('domain', domain_name or 'default')
            if len(text.strip()) < 2:
                continue
            self.items.append({
                'text': text,
                'domain': domain,
            })
        self.char_to_id = char_to_id

    def __len__(self):
        return len(self.items)

    def get_batch(self, indices):
        batch = []
        for i in indices:
            item = self.items[i]
            text_ids = tokenize_text(item['text'], self.char_to_id)
            domain = item['domain']
            batch.append({
                'text_ids': text_ids,
                'domain': domain,
            })
        return batch


def train_adapter(args):
    print("=" * 60)
    print("  DaoTi V53 Domain Adapter Training")
    print("=" * 60)

    if not verify_sha256("yijing_v53_daoti.pt"):
        print("[FAIL] Weight verification failed")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    model = load_daoti("yijing_v53_daoti.pt", device=device)

    tokenizer_data = load_tokenizer(args.tokenizer_path)
    char_to_id = tokenizer_data.get('char_to_id', {}) if tokenizer_data else {}

    print(f"\n  Domain: {args.domain_name}")
    print(f"  Method: {args.method}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")

    records = load_data(args.data_path)
    print(f"  Data records loaded: {len(records)}")
    if len(records) == 0:
        print("[FAIL] No valid records found in data file")
        return

    dataset = AdapterDataset(records, char_to_id, domain_name=args.domain_name)
    print(f"  Valid items: {len(dataset)}")

    freeze_dao_core(model)

    method_idx = METHOD_MAP.get(args.method, 0)
    head_map = {
        'traditional': model.head_traditional,
        'meihua': model.head_meihua,
        'liuyao': model.head_liuyao,
    }
    target_head = head_map[args.method]

    lora_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_params.append(param)

    if len(lora_params) == 0:
        print("[FAIL] No trainable LoRA parameters found")
        return

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_batches = max(1, len(dataset) // args.batch_size)
    print(f"\n  Training: {args.epochs} epochs x {n_batches} batches")

    model.train()
    best_loss = float('inf')
    best_state = None

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_retrieval_loss = 0.0
        epoch_palace_loss = 0.0
        n_steps = 0

        perm = np.random.permutation(len(dataset))

        for batch_idx in range(n_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(dataset))
            indices = perm[start:end].tolist()
            batch = dataset.get_batch(indices)

            batch_loss = 0.0
            batch_ret = 0.0
            batch_pal = 0.0

            for item in batch:
                text_ids = item['text_ids'].to(device)
                domain = item['domain']

                gua_idx = hash(domain + str(batch_idx)) % 64
                gua_tensor = torch.tensor([gua_idx], dtype=torch.long, device=device)
                method_tensor = torch.tensor([method_idx], dtype=torch.long, device=device)
                symbol_x = torch.tensor(
                    [sparse_expand_input(gua_idx)],
                    dtype=torch.float32, device=device
                )

                with torch.no_grad():
                    text_feat = model.encode_text(text_ids)

                outputs = model(symbol_x, text_ids, method_tensor, gua_tensor)
                head_out = outputs[args.method]

                proto = model.gua_prototype.weight
                proto_n = F.normalize(proto, p=2, dim=-1)
                feat_n = F.normalize(text_feat, p=2, dim=-1)
                retrieval_sim = torch.mm(feat_n, proto_n.t()).squeeze()
                retrieval_target = torch.zeros(64, device=device)
                retrieval_target[gua_idx] = 1.0
                retrieval_loss = F.cross_entropy(
                    retrieval_sim.unsqueeze(0) / 0.07,
                    retrieval_target.unsqueeze(0)
                )

                palace_target = torch.tensor(
                    [hash(domain) % 8], dtype=torch.long, device=device
                )
                palace_loss = F.cross_entropy(
                    head_out['palace'], palace_target
                )

                loss = retrieval_loss + 0.3 * palace_loss

                batch_loss += loss.item()
                batch_ret += retrieval_loss.item()
                batch_pal += palace_loss.item()

            batch_loss /= len(batch)
            batch_ret /= len(batch)
            batch_pal /= len(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()

            epoch_loss += batch_loss
            epoch_retrieval_loss += batch_ret
            epoch_palace_loss += batch_pal
            n_steps += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_steps, 1)
        avg_ret = epoch_retrieval_loss / max(n_steps, 1)
        avg_pal = epoch_palace_loss / max(n_steps, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {
                name: param.data.clone()
                for name, param in model.named_parameters()
                if param.requires_grad
            }

        print(f"  Epoch {epoch+1}/{args.epochs}  "
              f"Loss={avg_loss:.4f}  "
              f"Ret={avg_ret:.4f}  "
              f"Pal={avg_pal:.4f}  "
              f"LR={scheduler.get_last_lr()[0]:.6f}")

    adapter_weights = {}
    if best_state is not None:
        adapter_weights = best_state
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                adapter_weights[name] = param.data.clone()

    adapter_data = {
        'domain': args.domain_name,
        'method': args.method,
        'lora_rank': 8,
        'lora_alpha': 0.1,
        'weights': {k: v.cpu() for k, v in adapter_weights.items()},
        'train_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'data_path': args.data_path,
            'n_records': len(records),
        },
    }

    torch.save(adapter_data, args.output_path)
    print(f"\n  Adapter saved to: {args.output_path}")
    print(f"  Domain: {args.domain_name}")
    print(f"  Method: {args.method}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Trainable params: {len(adapter_weights)}")


def main():
    parser = argparse.ArgumentParser(description="DaoTi V53 Domain Adapter Training")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to JSONL data file')
    parser.add_argument('--output_path', type=str, default='adapter.pt',
                        help='Output path for adapter weights')
    parser.add_argument('--domain_name', type=str, default='custom',
                        help='Domain name for the adapter')
    parser.add_argument('--method', type=str, default='traditional',
                        choices=['traditional', 'meihua', 'liuyao'],
                        help='Which method head to train adapter for')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--tokenizer_path', type=str,
                        default='daoti_v53_tokenizer.pt',
                        help='Path to tokenizer file')
    args = parser.parse_args()
    train_adapter(args)


if __name__ == "__main__":
    main()
