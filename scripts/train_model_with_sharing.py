#!/usr/bin/env python
# scripts/train_model_with_sharing.py

import os, math
import numpy as np
import torch
from numpy.linalg import norm
from prp_model.task_generator import generate_fixed_task_set
from prp_model.nn_wrapper import TaskNetworkWrapper

# ------------------------ utilities ------------------------

TASK_MAP = {"A": (0, 0), "B": (1, 1), "C": (2, 2), "D": (0, 1), "E": (1, 0)}

def make_task_vec(name, N_pathways=3):
    i, o = TASK_MAP[name]
    v = torch.zeros(N_pathways * N_pathways)
    v[i * N_pathways + o] = 1
    return v

def sample_stimulus(N_pathways=3, N_features=3):
    x = torch.zeros(N_pathways * N_features)
    feats = torch.randint(0, N_features, (N_pathways,))
    for i in range(N_pathways):
        x[i * N_features + feats[i]] = 1
    return x

@torch.no_grad()
def activation_similarity_matrix(model, device="cpu", n_trials=200,
                                 N_pathways=3, N_features=3,
                                 task_names=("A","B","C","D","E")):
    """
    Mean hidden activation per task (single-step, p=0), then cosine sims.
    """
    model.eval()
    H = []
    for name in task_names:
        t = make_task_vec(name, N_pathways).to(device).float().unsqueeze(0)
        hs = []
        for _ in range(n_trials):
            x = sample_stimulus(N_pathways, N_features).to(device).float().unsqueeze(0)
            # forward once; TaskNetwork returns (y_o, y_h)
            y_o, y_h = model(x, t)
            hs.append(y_h.squeeze(0).cpu())
        h_mean = torch.stack(hs, 0).mean(0)
        h_mean = h_mean / (h_mean.norm() + 1e-9)
        H.append(h_mean)
    H = torch.stack(H, 0)              # [5, H]
    cos = (H @ H.T).cpu().numpy()      # cosine similarity matrix
    # convenience: lookups
    names = list(task_names)
    def s(a,b): return float(cos[names.index(a), names.index(b)])
    return cos, {"AD": s("A","D"), "BE": s("B","E")}

def sharing_score(scores):
    # simple, paper-aligned: want AD and BE high
    return scores["AD"] + scores["BE"]

# ------------------------ main training ------------------------

def main():
    # 1) Hyper-parameters
    N_pathways      = 3
    N_features      = 3
    samples_per_task= 100   # per task A–E
    hidden_dim      = 100
    learning_rate   = 0.3
    stop_loss       = 1e-3
    max_epochs      = 8000
    device          = "cpu"  # or "cuda"

    # sharing/ckpt settings
    eval_every      = 100     # compute similarity every N epochs
    n_trials_sim    = 200     # trials per task for activation mean
    ckpt_every      = 200     # save rolling checkpoints
    outdir          = "output"
    os.makedirs(outdir, exist_ok=True)

    # 2) Generate the fixed training set A–E
    inp, task, target, _ = generate_fixed_task_set(
        N_pathways=N_pathways,
        N_features=N_features,
        samples_per_task=samples_per_task,
        sd_scale=0.25,
        seed=42
    )
    stim = torch.tensor(inp,    dtype=torch.float32)
    tmat = torch.tensor(task,   dtype=torch.float32)
    y_gt = torch.tensor(target, dtype=torch.float32)

    # 3) Build model/wrapper
    wrapper = TaskNetworkWrapper(
        stim_input_dim = N_pathways * N_features,
        task_input_dim = N_pathways**2,
        hidden_dim     = hidden_dim,
        output_dim     = N_pathways * N_features,
        learning_rate  = learning_rate,
        device         = device
    )
    model = wrapper.model
    opt   = wrapper.optimizer
    crit  = wrapper.loss_fn

    print("🛠️  Starting training with sharing-tracking…")
    best_share = -1e9
    best_path  = os.path.join(outdir, "best_sharing.pth")

    loss_log = []
    acc_log  = []
    share_log= []

    for epoch in range(max_epochs):
        # shuffle each epoch
        perm = torch.randperm(stim.size(0))
        total_loss = 0.0
        correct = 0

        for i in perm:
            x_i = stim[i:i+1].to(device)
            t_i = tmat[i:i+1].to(device)
            y_i = y_gt[i:i+1].to(device)

            opt.zero_grad()
            y_pred, _ = model(x_i, t_i)
            loss = crit(y_pred, y_i)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            correct += (y_pred.argmax(dim=1) == y_i.argmax(dim=1)).sum().item()

        avg_loss = total_loss / stim.size(0)
        acc      = correct / stim.size(0)
        loss_log.append(avg_loss); acc_log.append(acc)

        # periodic checkpoint
        if (epoch+1) % ckpt_every == 0:
            p = os.path.join(outdir, f"epoch_{epoch+1:04d}.pth")
            torch.save(model.state_dict(), p)

        # measure sharing every eval_every epochs
        if (epoch+1) % eval_every == 0:
            cos, pair = activation_similarity_matrix(
                model, device=device, n_trials=n_trials_sim,
                N_pathways=N_pathways, N_features=N_features
            )
            score = sharing_score(pair)
            share_log.append((epoch+1, pair["AD"], pair["BE"], score))

            # save the best-by-sharing checkpoint
            if score > best_share:
                best_share = score
                torch.save(model.state_dict(), best_path)

            print(f"Epoch {epoch+1:04d} | Loss {avg_loss:.4f} | Acc {acc:.3f} "
                  f"| AD {pair['AD']:.3f} | BE {pair['BE']:.3f} | Share {score:.3f}")

        # early stop on loss
        if avg_loss <= stop_loss:
            print(f"🛑 Early stop on loss at epoch {epoch+1}")
            break

    # 4) Save final model AND best-sharing model
    final_path = os.path.join(outdir, "trained_model_sharing.pth")
    torch.save(model.state_dict(), final_path)
    print(f"✅ Final model saved to {final_path}")
    print(f"⭐ Best-by-sharing model saved to {best_path} (score={best_share:.3f})")

    # optional: dump logs
    with open(os.path.join(outdir, "sharing_log.txt"), "w") as f:
        for e, ad, be, sc in share_log:
            f.write(f"{e}\t{ad:.4f}\t{be:.4f}\t{sc:.4f}\n")

if __name__ == "__main__":
    main()
