# task_network.py
"""Core task-modulated feedforward network (Sim Study 3).

Implements the 3-layer architecture from Musslick et al. (2023), Fig. 9:
    stim_input → hidden → output
                ↑          ↑
             task_input  task_input

Key details:
- Both hidden and output layers receive additive task-control input.
- Fixed bias offset added to pre-activations at hidden and output (default -2.0).
- Sigmoid activations.
- Weight init mirrors MATLAB style:
    * stim→hidden, hidden→output, task→output ~ U[−init_scale, +init_scale]
    * task→hidden ~ U[−init_task_scale, +init_task_scale]
  If init_task_scale is None, it defaults to init_scale (as in NNmodel.m).
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


class TaskNetwork(nn.Module):
    """
    3-layer feedforward network with task-control inputs.
    Architecture (Fig. 9):
      stim_input (one-hot) ──► hidden ──► output ──► responses
                         ▲                ▲
           task_input ──┘                └──► output
    No learnable biases; a fixed bias offset is added pre-activation.
    """

    def __init__(
        self,
        stim_input_dim: int,
        task_input_dim: int,
        hidden_dim: int,
        output_dim: int,
        init_scale: float = 0.1,
        init_task_scale: Optional[float] = None,  # if None → equals init_scale
        bias_offset: float = -2.0,
        default_weight_decay: float = 0.0,        # Sim-3 uses 0.0
    ):
        super().__init__()
        self.bias_offset = float(bias_offset)
        self.init_scale = float(init_scale)
        self.init_task_scale = float(init_scale if init_task_scale is None else init_task_scale)
        self.default_weight_decay = float(default_weight_decay)

        # Layers (no learnable bias terms)
        self.fc_input_hidden = nn.Linear(stim_input_dim, hidden_dim, bias=False)
        self.fc_task_hidden  = nn.Linear(task_input_dim, hidden_dim, bias=False)
        self.fc_hidden_output = nn.Linear(hidden_dim, output_dim, bias=False)
        self.fc_task_output   = nn.Linear(task_input_dim, output_dim, bias=False)

        self.act = nn.Sigmoid()
        self._reset_parameters()

    # ----- Initialization (MATLAB-style uniform) -----
    def _reset_parameters(self):
        nn.init.uniform_(self.fc_input_hidden.weight,  -self.init_scale,     self.init_scale)
        nn.init.uniform_(self.fc_hidden_output.weight, -self.init_scale,     self.init_scale)
        nn.init.uniform_(self.fc_task_output.weight,   -self.init_scale,     self.init_scale)
        nn.init.uniform_(self.fc_task_hidden.weight,   -self.init_task_scale, self.init_task_scale)

    # ----- Forward -----
    def forward(self, stim_input: torch.Tensor, task_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param stim_input: (batch, stim_input_dim)
        :param task_input: (batch, task_input_dim)
        :return: (y_o, y_h) = (output activations, hidden activations)
        """
        # Hidden
        net_h = self.fc_input_hidden(stim_input) + self.fc_task_hidden(task_input) + self.bias_offset
        y_h = self.act(net_h)
        # Output
        net_o = self.fc_hidden_output(y_h) + self.fc_task_output(task_input) + self.bias_offset
        y_o = self.act(net_o)
        return y_o, y_h

    # ----- Convenience: optimizer with default weight decay -----
    def build_optimizer(
        self,
        lr: float = 0.3,
        weight_decay: Optional[float] = None,
        momentum: float = 0.0,
        nesterov: bool = False,
    ) -> torch.optim.Optimizer:
        """
        Build an SGD optimizer. If weight_decay is None, uses default_weight_decay.
        """
        if weight_decay is None:
            weight_decay = self.default_weight_decay
        return torch.optim.SGD(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
        )

    # ----- Activation-based task similarity (CorrTaskAvg-style) -----
    # Figure 14, Rational boundedness paper
    @torch.no_grad()
    def task_similarity_hidden(
        self,
        X_stim: torch.Tensor,
        T_task: torch.Tensor,
        tasks_index: Optional[torch.Tensor] = None,
        reduce: str = "mean",            # "mean" or "median"
        metric: str = "pearson",         # "pearson" or "cosine"
        device: Optional[torch.device] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute task similarity from HIDDEN ACTIVATIONS.

        Steps:
          1) Pass all (stim, task) pairs through the net.
          2) Aggregate hidden activations per task across stimuli (mean/median).
          3) Compute a task×task similarity matrix over the aggregated hidden vectors.

        Args
        ----
        X_stim: (N, stim_input_dim) tensor
        T_task: (N, task_input_dim) one-hot tensor (or probabilities)
        tasks_index: (N,) int tensor of task ids [0..T-1]; if None, inferred via argmax over T_task.
        reduce: "mean" or "median" aggregation across stimuli.
        metric: "pearson" (np.corrcoef over hidden units) or "cosine".
        device: optional device override.
        batch_size: optional micro-batch size for memory-friendly forward.

        Returns
        -------
        task_hidden_means: (T, hidden_dim) numpy array
        S: (T, T) similarity matrix (numpy array)
        """
        if device is None:
            device = next(self.parameters()).device

        X = X_stim.to(device)
        T = T_task.to(device)

        # Infer task ids if not provided (assume one-hot / argmax over task dim)
        if tasks_index is None:
            tasks_index = torch.argmax(T, dim=1)
        else:
            tasks_index = tasks_index.to(device)

        Tdim = T.shape[1]
        H = self.fc_input_hidden.out_features

        # Forward pass in (optional) micro-batches
        def _forward_all():
            Ys = []
            if batch_size is None:
                _, Yh = self.forward(X, T)
                return Yh
            # micro-batch
            for s in range(0, X.shape[0], batch_size):
                e = s + batch_size
                _, y_h = self.forward(X[s:e], T[s:e])
                Ys.append(y_h)
            return torch.cat(Ys, dim=0)

        Yh = _forward_all()  # (N, H)

        # Aggregate per task
        task_hidden_means = torch.zeros((Tdim, H), device=device)
        for tid in range(Tdim):
            mask = (tasks_index == tid)
            if mask.any():
                if reduce == "median":
                    task_hidden_means[tid] = Yh[mask].median(dim=0).values
                else:  # "mean"
                    task_hidden_means[tid] = Yh[mask].mean(dim=0)
            else:
                # Leave zeros for tasks with no samples
                pass

        A = task_hidden_means.detach().cpu().numpy()  # (T, H)

        # Similarity matrix
        if metric.lower() == "cosine":
            # Normalize rows then dot
            denom = np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
            An = A / denom
            S = An @ An.T
        else:  # "pearson"
            # Corr between rows across hidden units
            if A.shape[0] == 1:
                S = np.ones((1, 1), dtype=np.float64)
            else:
                S = np.corrcoef(A)  # rowvar=True by default when 2D
        return A, S
