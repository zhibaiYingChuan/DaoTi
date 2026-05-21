"""
DaoTi V53 Physics Adapter Training Tool
=========================================
Train a physics-to-spectrum adapter on top of the frozen DaoTi core.

This adapter bridges two gaps:
  1. INPUT: Physics parameters → lightweight MLP → 176-dim state vector (DaoTi's language)
  2. OUTPUT: DaoTi internal state → regression head → spectrum values

The DaoTi core remains completely frozen. Only the input encoder and output
regression head are trained.

Usage:
    python train_physics_adapter.py \
        --data_path ./optics_data.jsonl \
        --output_path ./optics_adapter.pt \
        --input_dim 8 \
        --output_dim 100 \
        --epochs 50 \
        --batch_size 16 \
        --lr 1e-3

Data format (JSONL, one record per line):
    {"params": [0.5, 0.3, 1.45, 0.2, ...], "spectrum": [0.1, 0.5, 0.9, ...]}
    {"params": [0.6, 0.4, 1.50, 0.3, ...], "spectrum": [0.2, 0.6, 0.8, ...]}
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from inference import (
    load_daoti, verify_sha256, MAX_SEQ, STATE_DIM,
)


class PhysicsInputEncoder(nn.Module):
    def __init__(self, input_dim, state_dim=176, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
            nn.LayerNorm(state_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class SpectrumRegressionHead(nn.Module):
    def __init__(self, state_dim=176, output_dim=100, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.regressor(x)


class PhysicsAdapterModel(nn.Module):
    def __init__(self, daoti_model, input_dim, output_dim, state_dim=176, freeze_daoti=True):
        super().__init__()
        self.daoti = daoti_model
        self.input_encoder = PhysicsInputEncoder(input_dim, state_dim)
        self.regression_head = SpectrumRegressionHead(state_dim, output_dim)

        if freeze_daoti:
            for param in self.daoti.parameters():
                param.requires_grad = False

    def forward(self, physics_params):
        state = self.input_encoder(physics_params)

        with torch.no_grad():
            gua_idx = torch.zeros(physics_params.size(0), dtype=torch.long, device=physics_params.device)
            method_idx = torch.zeros(physics_params.size(0), dtype=torch.long, device=physics_params.device)
            dummy_text = torch.ones(physics_params.size(0), MAX_SEQ, dtype=torch.long, device=physics_params.device)

            text_feat = self.daoti.encode_text(dummy_text)

            gate = self.daoti.text_gate(torch.cat([state, text_feat], dim=-1))
            fused_x = state + gate * text_feat

            method_vec = self.daoti.method_embed(method_idx)
            c = torch.cat([fused_x, method_vec], dim=1)

            features = self.daoti.heluo_ladder(c, gua_idx)

            fused, _ = self.daoti.method_fusion(
                features.unsqueeze(0), features.unsqueeze(0), features.unsqueeze(0)
            )
            features = features + fused.squeeze(0)

        spectrum = self.regression_head(features)
        return spectrum


def load_physics_data(data_path, input_dim, output_dim):
    records = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                params = record.get('params', [])
                spectrum = record.get('spectrum', [])

                if len(params) != input_dim:
                    print(f"  [WARN] Line {line_num}: expected {input_dim} params, got {len(params)}, skipping")
                    continue
                if len(spectrum) != output_dim:
                    print(f"  [WARN] Line {line_num}: expected {output_dim} spectrum values, got {len(spectrum)}, skipping")
                    continue

                records.append({
                    'params': torch.tensor(params, dtype=torch.float32),
                    'spectrum': torch.tensor(spectrum, dtype=torch.float32),
                })
            except (json.JSONDecodeError, TypeError) as e:
                print(f"  [WARN] Line {line_num}: parse error: {e}")
                continue
    return records


def normalize_data(records):
    if not records:
        return records, None, None

    all_params = torch.stack([r['params'] for r in records])
    all_spectrum = torch.stack([r['spectrum'] for r in records])

    params_mean = all_params.mean(dim=0)
    params_std = all_params.std(dim=0).clamp(min=1e-8)
    spectrum_mean = all_spectrum.mean(dim=0)
    spectrum_std = all_spectrum.std(dim=0).clamp(min=1e-8)

    for r in records:
        r['params_norm'] = (r['params'] - params_mean) / params_std
        r['spectrum_norm'] = (r['spectrum'] - spectrum_mean) / spectrum_std

    norm_stats = {
        'params_mean': params_mean,
        'params_std': params_std,
        'spectrum_mean': spectrum_mean,
        'spectrum_std': spectrum_std,
    }
    return records, norm_stats


def train_physics_adapter(args):
    print("=" * 60)
    print("  DaoTi V53 Physics Adapter Training")
    print("=" * 60)

    if not verify_sha256("yijing_v53_daoti.pt"):
        print("[FAIL] Weight verification failed")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    print(f"  Input dim: {args.input_dim}")
    print(f"  Output dim: {args.output_dim}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")

    daoti_model = load_daoti("yijing_v53_daoti.pt", device=device)

    print(f"\n  Loading data from: {args.data_path}")
    records = load_physics_data(args.data_path, args.input_dim, args.output_dim)
    print(f"  Valid records: {len(records)}")
    if len(records) == 0:
        print("[FAIL] No valid records found")
        return

    records, norm_stats = normalize_data(records)
    print(f"  Data normalized (mean/std computed)")

    split_idx = int(len(records) * (1 - args.val_split))
    train_records = records[:split_idx]
    val_records = records[split_idx:]
    print(f"  Train: {len(train_records)}  Val: {len(val_records)}")

    model = PhysicsAdapterModel(
        daoti_model, args.input_dim, args.output_dim,
        state_dim=STATE_DIM, freeze_daoti=True
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Frozen (DaoTi core): {frozen:,} params")
    print(f"  Trainable (adapter): {trainable:,} params ({100*trainable/total:.2f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    print(f"\n  Training started...")
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(train_records)

        train_loss = 0.0
        n_train_batches = 0

        for batch_start in range(0, len(train_records), args.batch_size):
            batch = train_records[batch_start:batch_start + args.batch_size]

            params_batch = torch.stack([r['params_norm'] for r in batch]).to(device)
            spectrum_batch = torch.stack([r['spectrum_norm'] for r in batch]).to(device)

            pred_spectrum = model(params_batch)

            loss = F.mse_loss(pred_spectrum, spectrum_batch)

            loss_r2 = 1 - F.mse_loss(pred_spectrum, spectrum_batch) / spectrum_batch.var()
            if torch.isnan(loss_r2):
                loss_r2 = torch.tensor(0.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        scheduler.step()

        avg_train_loss = train_loss / max(n_train_batches, 1)

        if len(val_records) > 0:
            model.eval()
            val_loss = 0.0
            val_r2_sum = 0.0
            n_val = 0

            with torch.no_grad():
                for batch_start in range(0, len(val_records), args.batch_size):
                    batch = val_records[batch_start:batch_start + args.batch_size]
                    params_batch = torch.stack([r['params_norm'] for r in batch]).to(device)
                    spectrum_batch = torch.stack([r['spectrum_norm'] for r in batch]).to(device)

                    pred_spectrum = model(params_batch)
                    loss = F.mse_loss(pred_spectrum, spectrum_batch)
                    r2 = 1 - loss / spectrum_batch.var()
                    if torch.isnan(r2):
                        r2 = torch.tensor(0.0)

                    val_loss += loss.item()
                    val_r2_sum += r2.item()
                    n_val += 1

            avg_val_loss = val_loss / max(n_val, 1)
            avg_val_r2 = val_r2_sum / max(n_val, 1)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {
                    name: param.data.clone()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}  "
                      f"Train={avg_train_loss:.6f}  "
                      f"Val={avg_val_loss:.6f}  "
                      f"R²={avg_val_r2:.4f}  "
                      f"LR={scheduler.get_last_lr()[0]:.6f}")
        else:
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                best_state = {
                    name: param.data.clone()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}  "
                      f"Train={avg_train_loss:.6f}  "
                      f"LR={scheduler.get_last_lr()[0]:.6f}")

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch+1} (patience={args.patience})")
            break

    adapter_data = {
        'type': 'physics_adapter',
        'input_dim': args.input_dim,
        'output_dim': args.output_dim,
        'state_dim': STATE_DIM,
        'norm_stats': {
            k: v.tolist() if isinstance(v, torch.Tensor) else v
            for k, v in norm_stats.items()
        } if norm_stats else None,
        'weights': {k: v.cpu() for k, v in (best_state or {}).items()},
        'train_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'data_path': args.data_path,
            'n_records': len(records),
            'n_train': len(train_records),
            'n_val': len(val_records),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(adapter_data, args.output_path)

    print(f"\n  Adapter saved to: {args.output_path}")
    print(f"  Input dim: {args.input_dim}")
    print(f"  Output dim: {args.output_dim}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Trainable params: {len(adapter_data['weights'])}")


def main():
    parser = argparse.ArgumentParser(description="DaoTi V53 Physics Adapter Training")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to JSONL data file with params/spectrum fields')
    parser.add_argument('--output_path', type=str, default='physics_adapter.pt',
                        help='Output path for adapter weights')
    parser.add_argument('--input_dim', type=int, required=True,
                        help='Number of physics input parameters')
    parser.add_argument('--output_dim', type=int, required=True,
                        help='Number of spectrum output values')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (0.0-0.5)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (epochs)')
    args = parser.parse_args()
    train_physics_adapter(args)


if __name__ == "__main__":
    main()
