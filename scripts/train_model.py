#!/usr/bin/env python
# scripts/train_model.py

import os
import torch
from prp_model.task_generator import generate_fixed_task_set
from prp_model.nn_wrapper import TaskNetworkWrapper

def main():
    # 1) Hyper-parameters
    N_pathways      = 3
    N_features      = 3
    samples_per_task= 100   # per task A–E
    hidden_dim      = 100
    learning_rate   = 0.3
    # The stop_loss value has been relaxed from 1e-3 to 0.005 for early stopping
    stop_loss       = 0.005
    max_epochs      = 8000
    device          = "cpu"  # or "cuda"

    # 2) Generate the fixed training set A–E
    inp, task, target, _ = generate_fixed_task_set(
        N_pathways=N_pathways,
        N_features=N_features,
        samples_per_task=samples_per_task,
        sd_scale=0.25,
        seed=42
    )
    # convert to torch tensors
    inp_t   = torch.tensor(inp,    dtype=torch.float32)
    task_t  = torch.tensor(task,   dtype=torch.float32)
    tgt_t   = torch.tensor(target, dtype=torch.float32)

    # 3) Build the wrapper & train
    wrapper = TaskNetworkWrapper(
        stim_input_dim = N_pathways * N_features,
        task_input_dim = N_pathways**2,
        hidden_dim     = hidden_dim,
        output_dim     = N_pathways * N_features,
        learning_rate  = learning_rate,
        device         = device
    )
    print("🛠️  Starting training...")
    wrapper.train_online(
        stim_inputs = inp_t,
        task_inputs = task_t,
        targets     = tgt_t,
        max_epochs  = max_epochs,
        stop_loss   = stop_loss,
        print_every = 20
    )

    # 4) Save model weights
    os.makedirs("output", exist_ok=True)
    save_path = os.path.join("output", "trained_model_005.pth")
    torch.save(wrapper.model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")

    # 5) (Optional) print final training stats
    losses, accs = wrapper.logs()
    print(f"Final loss = {losses[-1]:.5f}, final accuracy = {accs[-1]*100:.2f}%")

if __name__ == "__main__":
    main()