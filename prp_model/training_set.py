"""
training_set.py
----------------
MATLAB-faithful single-task training set generator used in Sim Study 3.

This mirrors the behavior of `createTrainingPatterns.m` for the single-task
regime:

- Exhaustive, **noise-free** stimuli: all `N_features ** N_pathways` combos
  (e.g., 3**3 = 27 for a 3×3 environment).
- One-hot **row-major** task cues: index = `in_dim * N_pathways + out_dim`.
- Targets copy the active feature from input pathway `i` to output `o`.
- Optionally reuse the **same 27 stimuli across tasks** (default=True), which
  encourages representational sharing (as in the original MATLAB training).

Typical usage
-------------
    from prp.training_set import generate_training_set_matlab_style
    X, T, Y, meta = generate_training_set_matlab_style(
        N_pathways=3, N_features=3, tasks=("A","B","C","D","E"),
        same_stimuli_across_tasks=True
    )
    # X/T/Y can be fed directly to the TaskNetworkWrapper.train_online(...)
"""

import numpy as np


def generate_training_set_matlab_style(
    N_pathways: int = 3,
    N_features: int = 3,
    tasks=("A", "B", "C", "D", "E"),
    same_stimuli_across_tasks: bool = True,
):
    """
    Build an exhaustive, noise-free single-task training set (MATLAB style).

    Parameters
    ----------
    N_pathways : int, default=3
        Number of input/output pathways (feature dimensions).
    N_features : int, default=3
        Number of features per pathway (units per dimension).
    tasks : iterable of str, default=("A","B","C","D","E")
        Which tasks to include. Tasks are named by (input→output) pairs:
            A: (0→0), B: (1→1), C: (2→2), D: (0→1), E: (1→0)
        (You can pass a subset/ordered tuple to select/reorder tasks.)
    same_stimuli_across_tasks : bool, default=True
        If True, each task sees **the same** exhaustive set of stimuli (27 rows
        for the 3×3 case). If False, each task gets its own (identical) copy
        of the exhaustive set; values are still noise-free/one-hot either way.

    Returns
    -------
    X : np.ndarray, shape (len(tasks)*S, N_pathways*N_features), dtype=float32
        Stimulus matrix. S = N_features ** N_pathways (e.g., 27 for 3×3).
        Each row has exactly one active feature per pathway (pure one-hot).
    T : np.ndarray, shape (len(tasks)*S, N_pathways**2), dtype=float32
        One-hot task cues in **row-major** encoding: index = in*N_pathways + out.
        For a given task, the same cue is repeated S times (once per stimulus).
    Y : np.ndarray, shape (len(tasks)*S, N_pathways*N_features), dtype=float32
        Target outputs: copy the active feature from input pathway `i` into
        output pathway `o` (all other outputs remain 0).
    meta : dict
        Metadata:
            meta["task_indices"]     -> np.ndarray[str] of task names per row
            meta["stimulus_indices"] -> np.ndarray[int] row index within the S stimuli

    Notes
    -----
    - This generator is **noise-free** by design. If you want to inject
      variability for ablations, do it outside this function.
    - The exact reuse of the same stimuli across tasks (`same_stimuli_across_tasks=True`)
      was critical in reproducing the representational sharing reported in the paper.
    """
    I = N_pathways * N_features
    Tdim = N_pathways ** 2

    # Mapping used throughout the project; matches paper’s task definitions.
    # Tasks are (input_dim, output_dim) with 0-based indices.
    task_map = {"A": (0, 0), "B": (1, 1), "C": (2, 2), "D": (0, 1), "E": (1, 0)}

    # ---- Build the exhaustive, clean stimuli (e.g., 27 for 3×3) ----
    # Each pathway picks exactly one active feature.
    combos = np.array(
        np.meshgrid(*[np.arange(N_features) for _ in range(N_pathways)])
    ).T.reshape(-1, N_pathways)

    def onehot_stim(fvec):
        """Convert a length-N_pathways vector of feature indices to a 1×(I) one-hot stimulus."""
        x = np.zeros(I, dtype=np.float32)
        for p in range(N_pathways):
            x[p * N_features + fvec[p]] = 1.0
        return x

    base_X = np.stack([onehot_stim(f) for f in combos], axis=0)  # (S, I)

    # ---- Tile stimuli and cues per task; build corresponding targets ----
    X_all, T_all, Y_all, names, sidx = [], [], [], [], []
    for name in tasks:
        i_in, i_out = task_map[name]

        # Task cue in **row-major** encoding: index = in * N + out
        cue = np.zeros(Tdim, dtype=np.float32)
        cue[i_in * N_pathways + i_out] = 1.0

        # Either share the exact same S stimuli across tasks,
        # or give each task its own copy (still deterministic).
        X_block = base_X if same_stimuli_across_tasks else base_X.copy()

        # Build targets for this (i_in → i_out) mapping:
        # For each stimulus row, find active feature in input pathway i_in,
        # copy it to the output pathway i_out.
        Y_block = np.zeros_like(X_block)
        for k in range(len(X_block)):
            # argmax within the relevant input slice (pure one-hot so this is exact)
            f = np.argmax(X_block[k, i_in * N_features : (i_in + 1) * N_features])
            Y_block[k, i_out * N_features + f] = 1.0

        # Stack blocks
        X_all.append(X_block)
        T_all.append(np.tile(cue, (len(X_block), 1)))
        Y_all.append(Y_block)
        names.extend([name] * len(X_block))
        sidx.extend(list(range(len(X_block))))

    # ---- Final tensors + metadata ----
    X = np.vstack(X_all).astype(np.float32)
    T = np.vstack(T_all).astype(np.float32)
    Y = np.vstack(Y_all).astype(np.float32)
    meta = {
        "task_indices": np.array(names),
        "stimulus_indices": np.array(sidx),
    }
    return X, T, Y, meta
