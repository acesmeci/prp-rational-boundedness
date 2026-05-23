# nn_wrapper.py
"""Thin wrapper around `TaskNetwork` for training and temporal integration.

Responsibilities:
- Online training with SGD + MSE on (stim, task) → one-hot target pairs.
- Single-step prediction (`predict`) for evaluation/debugging.
- Time-course `integrate`: runs the network over T steps and applies
  persistence (carry-over) by exponentially smoothing the pre-activations
  at hidden and output layers, matching the paper’s p-parameter dynamics.

Conventions:
- Inputs: `stim_sequence` shape (T, I), `task_sequence` shape (T, Tdim).
- Outputs: list of length T with output activations (torch.Tensor on CPU).
- Task cues are ROW-MAJOR one-hots (index = in_dim * N_pathways + out_dim).

This wrapper is what the PRP simulator calls to obtain output time series
for subsequent LCA readout and reward-rate analyses.
"""

import torch
import torch.nn as nn
from prp_model.task_network import TaskNetwork


class TaskNetworkWrapper:
    """
    Wraps the TaskNetwork to handle training, prediction, and
    time-course integration (for PRP simulations).
    """

    def __init__(
        self,
        stim_input_dim: int,
        task_input_dim: int,
        hidden_dim: int,
        output_dim: int,
        # training hyperparams
        learning_rate: float = 0.3,
        weight_decay: float | None = None,  # if None → model.default_weight_decay
        device: str = "cpu",
        # TaskNetwork init knobs (Sim-3 parity defaults)
        init_scale: float = 0.1,
        init_task_scale: float | None = None,  # None → equals init_scale
        bias_offset: float = -2.0,
        default_weight_decay: float = 0.0,     # Sim-3 uses 0.0
    ):
        self.device = torch.device(device)
        self.model = TaskNetwork(
            stim_input_dim=stim_input_dim,
            task_input_dim=task_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            init_scale=init_scale,
            init_task_scale=init_task_scale,
            bias_offset=bias_offset,
            default_weight_decay=default_weight_decay,
        ).to(self.device)

        self.loss_fn = nn.MSELoss()
        # Use the model's helper so default_weight_decay is respected
        self.optimizer = self.model.build_optimizer(lr=learning_rate, weight_decay=weight_decay)

        self.loss_log: list[float] = []
        self.accuracy_log: list[float] = []

    def train_online(
        self,
        stim_inputs: torch.Tensor,
        task_inputs: torch.Tensor,
        targets: torch.Tensor,
        max_epochs: int = 8000,
        stop_loss: float = 1e-3,
        print_every: int = 10,
    ):
        """
        Trains on (stim, task) -> one-hot target pairs until MSE loss <= stop_loss
        or max_epochs is reached.

        stim_inputs: (N, stim_input_dim) float tensor
        task_inputs: (N, task_input_dim) float tensor
        targets:     (N, output_dim)     float tensor (one-hot)
        """
        self.model.train()
        stim = stim_inputs.to(self.device).float()
        task = task_inputs.to(self.device).float()
        y_true = targets.to(self.device).float()

        N = stim.size(0)
        for epoch in range(max_epochs):
            total_loss = 0.0
            correct = 0

            perm = torch.randperm(N, device=self.device)
            for i in perm:
                x_i = stim[i : i + 1]
                t_i = task[i : i + 1]
                y_i = y_true[i : i + 1]

                self.optimizer.zero_grad(set_to_none=True)
                y_pred, _ = self.model(x_i, t_i)
                loss = self.loss_fn(y_pred, y_i)
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss)
                correct += int((y_pred.argmax(dim=1) == y_i.argmax(dim=1)).sum())

            avg_loss = total_loss / N
            acc = correct / N
            self.loss_log.append(avg_loss)
            self.accuracy_log.append(acc)

            if print_every and (epoch % print_every == 0):
                print(f"Epoch {epoch:04d} | Loss: {avg_loss:.4f} | Acc: {acc:.3f}")

            if avg_loss <= stop_loss:
                print(f"✅ Converged at epoch {epoch:04d} | Loss: {avg_loss:.4f}")
                break
        else:
            print(f"⚠️ Max epochs ({max_epochs}) reached | Final loss: {avg_loss:.4f}")

    def predict(self, stim_input: torch.Tensor, task_input: torch.Tensor):
        """Single-timestep prediction: returns output vector."""
        self.model.eval()
        with torch.no_grad():
            x = stim_input.unsqueeze(0).to(self.device).float()
            t = task_input.unsqueeze(0).to(self.device).float()
            y_pred, _ = self.model(x, t)
        return y_pred.squeeze(0).cpu()

    def integrate(
        self,
        stim_sequence: torch.Tensor,
        task_sequence: torch.Tensor,
        persistence: float = 0.0,
    ):
        """
        Runs the network over a sequence of timesteps, applying
        temporal persistence to net inputs. Returns list of outputs.

        stim_sequence: (T, stim_input_dim)
        task_sequence: (T, task_input_dim)
        persistence:   float in [0,1] controlling leak (p in paper)
        """
        self.model.eval()
        outputs = []
        prev_net_h = None
        prev_net_o = None

        with torch.no_grad():
            for t_idx in range(stim_sequence.size(0)):
                x_t = stim_sequence[t_idx : t_idx + 1].to(self.device).float()
                t_t = task_sequence[t_idx : t_idx + 1].to(self.device).float()

                net_h = (
                    self.model.fc_input_hidden(x_t)
                    + self.model.fc_task_hidden(t_t)
                    + self.model.bias_offset
                )
                if prev_net_h is not None:
                    net_h = (1 - persistence) * net_h + persistence * prev_net_h
                y_h = torch.sigmoid(net_h)
                prev_net_h = net_h

                net_o = (
                    self.model.fc_hidden_output(y_h)
                    + self.model.fc_task_output(t_t)
                    + self.model.bias_offset
                )
                if prev_net_o is not None:
                    net_o = (1 - persistence) * net_o + persistence * prev_net_o
                y_o = torch.sigmoid(net_o)
                prev_net_o = net_o

                outputs.append(y_o.squeeze(0).cpu())

        return outputs

    def get_weights(self):
        return {
            "W_input_hidden": self.model.fc_input_hidden.weight.detach().cpu().numpy(),
            "W_task_hidden":  self.model.fc_task_hidden.weight.detach().cpu().numpy(),
            "W_hidden_output":self.model.fc_hidden_output.weight.detach().cpu().numpy(),
            "W_task_output":  self.model.fc_task_output.weight.detach().cpu().numpy(),
        }

    def logs(self):
        return self.loss_log, self.accuracy_log
