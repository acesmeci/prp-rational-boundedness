"""
Shared utilities for the PRP simulation pipeline.

Consolidates functions that were previously duplicated across
train_ensemble.py, run_prp_ensemble.py, and plot_prp_ensemble.py:

- Trial generation (generate_trial_pair)
- Network factory (make_wrapper)
- Checkpoint I/O (save_state, load_state, save_threshold, load_threshold)
- Aggregation helpers (nanmean, nanse, average_with_se)
- Slope analysis (steepest_adjacent_slope)
- Time-unit conversion constants
"""

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from prp_model.nn_wrapper import TaskNetworkWrapper
from prp_model.lca import _DEFAULTS


# ── Task definitions (paper, Fig. 13) ────────────────────────────────────────
TASK_MAP = {
    "A": (0, 0),
    "B": (1, 1),
    "C": (2, 2),
    "D": (0, 1),
    "E": (1, 0),
}

N_PATHWAYS = 3
N_FEATURES = 3


# ── Network factory ──────────────────────────────────────────────────────────

def make_wrapper(device: str = "cpu") -> TaskNetworkWrapper:
    """Create a TaskNetworkWrapper matching Musslick et al. (2023) Sim Study 3."""
    return TaskNetworkWrapper(
        stim_input_dim=N_PATHWAYS * N_FEATURES,
        task_input_dim=N_PATHWAYS ** 2,
        hidden_dim=100,
        output_dim=N_PATHWAYS * N_FEATURES,
        learning_rate=0.3,
        init_scale=0.1,
        init_task_scale=None,
        bias_offset=-2.0,
        default_weight_decay=0.0,
        device=device,
    )


# ── Trial generation ─────────────────────────────────────────────────────────

def generate_trial_pair(
    prp_pair: tuple[str, str] = ("B", "A"),
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a single PRP trial: (stim1, stim2, cue1, cue2).

    Each stimulus vector contains only the task-relevant dimension active
    (one feature unit set to 1), with all other dimensions at zero. This
    follows Musslick et al. (2023, p. 68): "we first presented the network
    with a feature from the stimulus dimension relevant to Task 1 [...] by
    activating the corresponding unit in the stimulus input layer while
    keeping all other stimulus input units inactivated."

    At the SOA, stim2 is added to stim1, resulting in exactly two active
    dimensions with activation values of 1.0 each.

    Parameters
    ----------
    prp_pair : (str, str)
        (Task1_name, Task2_name), e.g. ("B", "A").
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    stim1, stim2, cue1, cue2 : np.ndarray
        Partial stimulus vectors and one-hot task cue vectors.
    """
    rng = np.random.RandomState(seed)
    feats = rng.randint(0, N_FEATURES, size=N_PATHWAYS)

    def _make(task_name, features):
        in_dim, out_dim = TASK_MAP[task_name]

        # Stimulus: only the task-relevant dimension is active
        stim = np.zeros(N_PATHWAYS * N_FEATURES, dtype=np.float32)
        stim[in_dim * N_FEATURES + features[in_dim]] = 1.0

        # Task cue: row-major one-hot
        cue = np.zeros(N_PATHWAYS ** 2, dtype=np.float32)
        cue[in_dim * N_PATHWAYS + out_dim] = 1.0
        return stim, cue

    stim1, cue1 = _make(prp_pair[0], feats)
    stim2, cue2 = _make(prp_pair[1], feats)
    return stim1, stim2, cue1, cue2


# ── Checkpoint I/O ───────────────────────────────────────────────────────────

def save_state(wrapper: TaskNetworkWrapper, path: str | Path) -> None:
    """Save model weights to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(wrapper.model.state_dict(), path)


def load_state(path: str | Path, device: str = "cpu") -> TaskNetworkWrapper:
    """Load model weights from disk into a fresh wrapper."""
    wrapper = make_wrapper(device=device)
    wrapper.model.load_state_dict(torch.load(path, map_location=device))
    wrapper.model.eval()
    return wrapper


def save_threshold(z: float, path: str | Path) -> None:
    """Save a single LCA threshold value to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"z": float(z)}, f)


def load_threshold(path: str | Path) -> float:
    """Load a single LCA threshold value from JSON."""
    with open(path, "r") as f:
        return float(json.load(f)["z"])


# ── Aggregation helpers ──────────────────────────────────────────────────────

def nanmean(x) -> float:
    """NaN-safe mean, returning float."""
    return float(np.nanmean(np.asarray(x, float)))


def nanse(x) -> float:
    """NaN-safe standard error of the mean."""
    arr = np.asarray(x, float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return np.nan
    return float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))


def average_with_se(
    results_list: list[dict],
    keys: Sequence[str],
) -> dict:
    """
    Average sweep results across networks, computing mean ± SE per SOA.

    Assumes identical SOA grids across all entries in results_list.
    """
    soa = results_list[0]["soa"]
    out = {"soa": soa}
    for k in keys:
        out[k] = []
        out[k + "_se"] = []
        for i in range(len(soa)):
            vals = [r[k][i] for r in results_list]
            out[k].append(nanmean(vals))
            out[k + "_se"].append(nanse(vals))
    return out


# ── Slope analysis ───────────────────────────────────────────────────────────

def steepest_adjacent_slope(
    soa_steps: np.ndarray,
    rt_values: np.ndarray,
) -> dict:
    """
    Find the most-negative adjacent-pair slope in the RT vs. SOA curve.

    Both inputs should be in raw simulation units (LCA steps / dt-seconds).
    The returned slope is in units of seconds-per-second (Δ RT / Δ SOA),
    which is directly comparable to the canonical −1 prediction.

    Parameters
    ----------
    soa_steps : array-like
        SOA values in LCA steps.
    rt_values : array-like
        Mean RT values (in dt-seconds, i.e. LCA steps × dt).

    Returns
    -------
    dict with keys:
        seg           : (soa_start, soa_end) of steepest segment
        slope_s_per_s : slope in s/s (Δ RT_seconds / Δ SOA_seconds)
    """
    soa = np.asarray(soa_steps, float)
    y = np.asarray(rt_values, float)

    mask = np.isfinite(soa) & np.isfinite(y)
    soa, y = soa[mask], y[mask]
    order = np.argsort(soa)
    soa, y = soa[order], y[order]

    # RT is in seconds (completed_steps × dt + t0), SOA is in steps.
    # Convert SOA to seconds for a clean s/s slope:
    dt = _DEFAULTS["dt"]
    soa_sec = soa * dt               # steps → seconds
    dy = np.diff(y)                   # Δ RT in seconds
    dsoa = np.diff(soa_sec)           # Δ SOA in seconds
    slope_s_per_s = dy / dsoa         # dimensionless (s/s)

    i = int(np.nanargmin(slope_s_per_s))
    return {
        "seg": (float(soa[i]), float(soa[i + 1])),
        "slope_s_per_s": float(slope_s_per_s[i]),
    }


# ── Display-unit conversion ─────────────────────────────────────────────────

def steps_to_ms(steps: np.ndarray) -> np.ndarray:
    """Convert simulation steps to milliseconds.

    With the clock-faithful parameterization (dt = 0.05 s per step),
    each step is exactly dt * 1000 = 50 ms.
    """
    return np.asarray(steps, float) * _DEFAULTS["dt"] * 1000


def sim_seconds_to_ms(seconds: np.ndarray) -> np.ndarray:
    """Convert seconds to milliseconds.

    With the clock-faithful parameterization, simulation seconds ARE
    physical seconds, so the conversion is simply × 1000.
    """
    return np.asarray(seconds, float) * 1000


# --- TASK-SWITCHING TRIAL GENERATION (BIVALENT STIMULI) ─────────────────────────────

def generate_switch_trial(
    prev_task: str,
    curr_task: str,
    seed: int | None = None,
    blank_prev_stimulus: bool = True,
) -> dict:
    """
    Generate stimuli and cues for a single task-switching trial.

    Unlike PRP trials, task-switching uses BIVALENT stimuli: all three
    stimulus dimensions are active simultaneously (one feature per dimension).
    Previous and current tasks receive INDEPENDENTLY sampled stimuli
    (matching MATLAB's independent randi sampling in Transition_Analysis).

    Congruency is determined by the CURRENT stimulus: whether the active
    features in the two tasks' input dimensions are identical within that
    stimulus. This is independent of the previous stimulus.

    Parameters
    ----------
    prev_task : str
        Name of the previous task (e.g. "B").
    curr_task : str
        Name of the current task (e.g. "A").
    seed : int or None
        Random seed for stimulus sampling.
    blank_prev_stimulus : bool
        If True (default), the previous trial's stimulus is all zeros.
        If False, an independently sampled bivalent stimulus is generated.

    Returns
    -------
    dict with keys:
        stim_prev, stim_curr, cue_prev, cue_curr,
        congruent, correct_idx, resp_indices
    """
    rng = np.random.RandomState(seed)
    I = N_PATHWAYS * N_FEATURES
    T_dim = N_PATHWAYS ** 2

    # Independently sample features for current and previous stimuli
    features_curr = rng.randint(0, N_FEATURES, size=N_PATHWAYS)
    features_prev = rng.randint(0, N_FEATURES, size=N_PATHWAYS)

    def _bivalent_stim(feats):
        stim = np.zeros(I, dtype=np.float32)
        for p in range(N_PATHWAYS):
            stim[p * N_FEATURES + feats[p]] = 1.0
        return stim

    def _cue(task_name):
        in_dim, out_dim = TASK_MAP[task_name]
        cue = np.zeros(T_dim, dtype=np.float32)
        cue[in_dim * N_PATHWAYS + out_dim] = 1.0
        return cue

    stim_curr = _bivalent_stim(features_curr)
    cue_prev = _cue(prev_task)
    cue_curr = _cue(curr_task)

    if blank_prev_stimulus:
        stim_prev = np.zeros(I, dtype=np.float32)
    else:
        stim_prev = _bivalent_stim(features_prev)

    # Congruency: based on CURRENT stimulus only
    # Do the two tasks' input dimensions have the same feature in this stimulus?
    in_prev = TASK_MAP[prev_task][0]
    in_curr = TASK_MAP[curr_task][0]
    congruent = bool(features_curr[in_prev] == features_curr[in_curr])

    # Current task's correct response and output indices
    in_dim, out_dim = TASK_MAP[curr_task]
    correct_idx = int(features_curr[in_dim])
    resp_indices = list(range(out_dim * N_FEATURES, (out_dim + 1) * N_FEATURES))

    return {
        "stim_prev": stim_prev,
        "stim_curr": stim_curr,
        "cue_prev": cue_prev,
        "cue_curr": cue_curr,
        "congruent": congruent,
        "correct_idx": correct_idx,
        "resp_indices": resp_indices,
    }