#!/usr/bin/env python3
"""
Compute hidden-layer task similarities across a trained PRP-network ensemble.

Primary analysis
----------------
For each checkpoint:
1. Generate the exhaustive 27-stimulus single-task set for Tasks A-E.
2. Run ordinary static forward passes (no persistence).
3. Average the 100-dimensional hidden activation pattern across the 27
   stimuli separately for each task.
4. Pearson-correlate the five task-level mean patterns, producing one 5x5
   similarity matrix per network.
5. Average correlations across networks and report their across-network SD.

Optional diagnostic
-------------------
The script also computes correlations between task-to-hidden weight vectors
(one column of fc_task_hidden.weight per task cue). This is saved separately
and is not used as the primary thesis measure.

Run from the repository root:
    python -m scripts.analyze_task_similarity \
        --checkpoint_dir ensemble_ckpt_p09 \
        --n_networks 20 \
        --output_dir output/task_similarity

Outputs
-------
- hidden_similarity_mean_sd.pdf / .png
- hidden_similarity_sd.pdf
- hidden_pairwise_by_network.csv
- hidden_pairwise_summary.csv
- task_similarity_results.npz
- task_similarity_summary.json
- corresponding weight-diagnostic files unless --skip_weight_diagnostic
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from prp_model.training_set import generate_training_set_matlab_style
from prp_model.utils import TASK_MAP, load_state


TASKS: tuple[str, ...] = ("A", "B", "C", "D", "E")
EPS = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ensemble hidden-layer task-similarity matrices."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("ensemble_ckpt_p09"),
        help="Directory containing net_00.pt ... net_19.pt.",
    )
    parser.add_argument(
        "--checkpoint_pattern",
        type=str,
        default="net_{idx:02d}.pt",
        help="Checkpoint filename pattern. It must contain '{idx}'.",
    )
    parser.add_argument(
        "--n_networks",
        type=int,
        default=20,
        help="Number of checkpoints, starting from index 0.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/task_similarity"),
        help="Directory for figures and tabular outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device used for forward passes.",
    )
    parser.add_argument(
        "--skip_weight_diagnostic",
        action="store_true",
        help="Skip the optional task-to-hidden weight-vector analysis.",
    )
    return parser.parse_args()


def validate_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    paths = [
        args.checkpoint_dir / args.checkpoint_pattern.format(idx=idx)
        for idx in range(args.n_networks)
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:10])
        suffix = "\n  ..." if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} checkpoint(s):\n{preview}{suffix}"
        )
    return paths


def pearson_matrix(representations: np.ndarray) -> np.ndarray:
    """Correlate task representations across hidden units."""
    representations = np.asarray(representations, dtype=np.float64)
    if representations.ndim != 2:
        raise ValueError(
            f"Expected a 2D task-by-unit array, got {representations.shape}."
        )
    row_sd = representations.std(axis=1)
    if np.any(row_sd <= EPS):
        bad = [TASKS[i] for i in np.where(row_sd <= EPS)[0]]
        raise ValueError(f"Near-constant representation vector(s): {bad}")
    matrix = np.corrcoef(representations)
    if not np.isfinite(matrix).all():
        raise ValueError("Similarity matrix contains NaN or infinite values.")
    return matrix


@torch.no_grad()
def hidden_activity_representations(
    wrapper,
    X: np.ndarray,
    T: np.ndarray,
    task_labels: np.ndarray,
    device: str,
) -> np.ndarray:
    """Return one mean hidden-activation vector per task: shape [5, H]."""
    wrapper.model.eval()
    X_t = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    T_t = torch.from_numpy(T).to(device=device, dtype=torch.float32)
    _, hidden = wrapper.model(X_t, T_t)
    hidden_np = hidden.detach().cpu().numpy()

    task_means = []
    for task in TASKS:
        mask = task_labels == task
        n_patterns = int(mask.sum())
        if n_patterns != 27:
            raise ValueError(
                f"Expected 27 patterns for Task {task}, found {n_patterns}."
            )
        task_means.append(hidden_np[mask].mean(axis=0))

    return np.stack(task_means, axis=0)


def task_weight_representations(wrapper) -> np.ndarray:
    """
    Return task-to-hidden weight vectors: shape [5, H].

    nn.Linear(task_dim, hidden_dim) stores weights as [hidden_dim, task_dim],
    so the vector for a task cue is the corresponding COLUMN.
    """
    weights = (
        wrapper.model.fc_task_hidden.weight.detach().cpu().numpy()
    )
    vectors = []
    for task in TASKS:
        input_dim, output_dim = TASK_MAP[task]
        cue_index = input_dim * 3 + output_dim
        vectors.append(weights[:, cue_index])
    return np.stack(vectors, axis=0)


def fisher_mean_matrix(matrices: np.ndarray) -> np.ndarray:
    """Fisher-z mean across networks, with an exact unit diagonal."""
    clipped = np.clip(matrices, -1.0 + EPS, 1.0 - EPS)
    mean_r = np.tanh(np.arctanh(clipped).mean(axis=0))
    np.fill_diagonal(mean_r, 1.0)
    return mean_r


def matrix_statistics(matrices: np.ndarray) -> dict[str, np.ndarray]:
    if matrices.ndim != 3 or matrices.shape[1:] != (len(TASKS), len(TASKS)):
        raise ValueError(f"Unexpected matrix stack shape: {matrices.shape}")
    return {
        "mean": matrices.mean(axis=0),
        "sd": matrices.std(axis=0, ddof=1),
        "se": matrices.std(axis=0, ddof=1) / np.sqrt(matrices.shape[0]),
        "fisher_mean": fisher_mean_matrix(matrices),
    }


def pair_indices() -> Iterable[tuple[int, int]]:
    return combinations(range(len(TASKS)), 2)


def write_pairwise_by_network(
    path: Path,
    matrices: np.ndarray,
    measure: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "measure",
                "network",
                "task_1",
                "task_2",
                "pearson_r",
                "fisher_z",
            ],
        )
        writer.writeheader()
        for network_idx, matrix in enumerate(matrices):
            for i, j in pair_indices():
                r = float(matrix[i, j])
                writer.writerow(
                    {
                        "measure": measure,
                        "network": network_idx,
                        "task_1": TASKS[i],
                        "task_2": TASKS[j],
                        "pearson_r": f"{r:.10f}",
                        "fisher_z": f"{np.arctanh(np.clip(r, -1 + EPS, 1 - EPS)):.10f}",
                    }
                )


def pairwise_summary_rows(
    matrices: np.ndarray,
    measure: str,
) -> list[dict[str, float | int | str]]:
    rows = []
    n_networks = matrices.shape[0]
    for i, j in pair_indices():
        values = matrices[:, i, j]
        fisher_mean = float(
            np.tanh(np.arctanh(np.clip(values, -1 + EPS, 1 - EPS)).mean())
        )
        rows.append(
            {
                "measure": measure,
                "task_1": TASKS[i],
                "task_2": TASKS[j],
                "n_networks": n_networks,
                "mean_r": float(values.mean()),
                "sd_r": float(values.std(ddof=1)),
                "se_r": float(values.std(ddof=1) / np.sqrt(n_networks)),
                "fisher_mean_r": fisher_mean,
                "min_r": float(values.min()),
                "max_r": float(values.max()),
            }
        )
    return rows


def write_pairwise_summary(
    path: Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    fieldnames = [
        "measure",
        "task_1",
        "task_2",
        "n_networks",
        "mean_r",
        "sd_r",
        "se_r",
        "fisher_mean_r",
        "min_r",
        "max_r",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in (
                "mean_r",
                "sd_r",
                "se_r",
                "fisher_mean_r",
                "min_r",
                "max_r",
            ):
                formatted[key] = f"{float(row[key]):.10f}"
            writer.writerow(formatted)


def plot_mean_sd_matrix(
    mean_matrix: np.ndarray,
    sd_matrix: np.ndarray,
    pdf_path: Path,
    png_path: Path,
) -> None:
    """Thesis-facing heatmap: ensemble mean with across-network SD annotations."""
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(mean_matrix, vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(len(TASKS)), labels=TASKS)
    ax.set_yticks(np.arange(len(TASKS)), labels=TASKS)
    ax.set_xlabel("Task")
    ax.set_ylabel("Task")

    for i in range(len(TASKS)):
        for j in range(len(TASKS)):
            if i == j:
                label = "1.00"
            else:
                label = f"{mean_matrix[i, j]:.2f}\n±{sd_matrix[i, j]:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Pearson correlation (ensemble mean)")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sd_matrix(sd_matrix: np.ndarray, path: Path) -> None:
    """Diagnostic heatmap of across-network variability."""
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(sd_matrix, vmin=0.0)
    ax.set_xticks(np.arange(len(TASKS)), labels=TASKS)
    ax.set_yticks(np.arange(len(TASKS)), labels=TASKS)
    ax.set_xlabel("Task")
    ax.set_ylabel("Task")

    for i in range(len(TASKS)):
        for j in range(len(TASKS)):
            ax.text(
                j,
                i,
                f"{sd_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Across-network SD of Pearson r")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def print_key_pairs(
    rows: list[dict[str, float | int | str]],
    measure_label: str,
) -> None:
    lookup = {
        (str(row["task_1"]), str(row["task_2"])): row
        for row in rows
    }
    print(f"\n{measure_label}")
    print("-" * len(measure_label))
    for pair in (("A", "D"), ("B", "E"), ("A", "B"), ("A", "C")):
        row = lookup[pair]
        print(
            f"{pair[0]}-{pair[1]}: "
            f"mean r = {float(row['mean_r']): .3f}, "
            f"SD = {float(row['sd_r']):.3f}, "
            f"SE = {float(row['se_r']):.3f}, "
            f"Fisher-mean r = {float(row['fisher_mean_r']): .3f}"
        )


def serializable_stats(stats: dict[str, np.ndarray]) -> dict[str, list]:
    return {key: value.tolist() for key, value in stats.items()}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = validate_checkpoint_paths(args)

    X, T, _, meta = generate_training_set_matlab_style(
        N_pathways=3,
        N_features=3,
        tasks=TASKS,
        same_stimuli_across_tasks=True,
    )
    task_labels = np.asarray(meta["task_indices"])

    hidden_representations = []
    hidden_matrices = []
    weight_representations = []
    weight_matrices = []

    print(f"Loading {len(checkpoint_paths)} checkpoints...")
    for network_idx, checkpoint_path in enumerate(checkpoint_paths):
        wrapper = load_state(checkpoint_path, device=args.device)

        hidden_repr = hidden_activity_representations(
            wrapper, X, T, task_labels, args.device
        )
        hidden_representations.append(hidden_repr)
        hidden_matrices.append(pearson_matrix(hidden_repr))

        if not args.skip_weight_diagnostic:
            weight_repr = task_weight_representations(wrapper)
            weight_representations.append(weight_repr)
            weight_matrices.append(pearson_matrix(weight_repr))

        print(f"  [{network_idx + 1:02d}/{len(checkpoint_paths):02d}] {checkpoint_path.name}")

    hidden_representations_arr = np.stack(hidden_representations)
    hidden_matrices_arr = np.stack(hidden_matrices)
    hidden_stats = matrix_statistics(hidden_matrices_arr)
    hidden_rows = pairwise_summary_rows(hidden_matrices_arr, "hidden_activity")

    write_pairwise_by_network(
        args.output_dir / "hidden_pairwise_by_network.csv",
        hidden_matrices_arr,
        "hidden_activity",
    )
    write_pairwise_summary(
        args.output_dir / "hidden_pairwise_summary.csv",
        hidden_rows,
    )
    plot_mean_sd_matrix(
        hidden_stats["mean"],
        hidden_stats["sd"],
        args.output_dir / "hidden_similarity_mean_sd.pdf",
        args.output_dir / "hidden_similarity_mean_sd.png",
    )
    plot_sd_matrix(
        hidden_stats["sd"],
        args.output_dir / "hidden_similarity_sd.pdf",
    )

    npz_payload = {
        "tasks": np.asarray(TASKS),
        "hidden_representations": hidden_representations_arr,
        "hidden_matrices": hidden_matrices_arr,
        "hidden_mean": hidden_stats["mean"],
        "hidden_sd": hidden_stats["sd"],
        "hidden_se": hidden_stats["se"],
        "hidden_fisher_mean": hidden_stats["fisher_mean"],
    }
    summary_json: dict[str, object] = {
        "tasks": list(TASKS),
        "n_networks": len(checkpoint_paths),
        "checkpoint_dir": str(args.checkpoint_dir),
        "method": (
            "For each network and task, hidden sigmoid activations were averaged "
            "over the exhaustive 27-stimulus single-task set. The five resulting "
            "task vectors were Pearson-correlated across hidden units. Correlation "
            "matrices were then averaged across networks; SD is across networks."
        ),
        "hidden_activity": {
            "matrix_statistics": serializable_stats(hidden_stats),
            "pairwise_summary": hidden_rows,
        },
    }

    print_key_pairs(hidden_rows, "Hidden-activity similarity (primary)")

    if not args.skip_weight_diagnostic:
        weight_representations_arr = np.stack(weight_representations)
        weight_matrices_arr = np.stack(weight_matrices)
        weight_stats = matrix_statistics(weight_matrices_arr)
        weight_rows = pairwise_summary_rows(
            weight_matrices_arr, "task_to_hidden_weights"
        )

        write_pairwise_by_network(
            args.output_dir / "weight_pairwise_by_network.csv",
            weight_matrices_arr,
            "task_to_hidden_weights",
        )
        write_pairwise_summary(
            args.output_dir / "weight_pairwise_summary.csv",
            weight_rows,
        )
        plot_mean_sd_matrix(
            weight_stats["mean"],
            weight_stats["sd"],
            args.output_dir / "weight_similarity_mean_sd.pdf",
            args.output_dir / "weight_similarity_mean_sd.png",
        )
        plot_sd_matrix(
            weight_stats["sd"],
            args.output_dir / "weight_similarity_sd.pdf",
        )

        npz_payload.update(
            {
                "weight_representations": weight_representations_arr,
                "weight_matrices": weight_matrices_arr,
                "weight_mean": weight_stats["mean"],
                "weight_sd": weight_stats["sd"],
                "weight_se": weight_stats["se"],
                "weight_fisher_mean": weight_stats["fisher_mean"],
            }
        )
        summary_json["task_to_hidden_weights"] = {
            "matrix_statistics": serializable_stats(weight_stats),
            "pairwise_summary": weight_rows,
        }
        print_key_pairs(weight_rows, "Task-to-hidden weight similarity (diagnostic)")

    np.savez_compressed(
        args.output_dir / "task_similarity_results.npz",
        **npz_payload,
    )
    with (args.output_dir / "task_similarity_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary_json, file, indent=2)

    print(f"\nSaved outputs to: {args.output_dir.resolve()}")
    print("Primary thesis figure: hidden_similarity_mean_sd.pdf")
    print("Primary statistics: hidden_pairwise_summary.csv")


if __name__ == "__main__":
    main()