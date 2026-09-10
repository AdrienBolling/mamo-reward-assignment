"""`mamora-analyze`: summarize and plot the `metrics.jsonl` files of a run or a sweep.

Given a root directory, every `**/metrics.jsonl` is one run. Its config comes
from the `.hydra/config.yaml` next to it. Runs are grouped by the config keys
that vary across the sweep, except `seed`; seeds are the replicates. Outputs go
to `<root>/analysis/`:

- `summary.md`, `summary.csv`: per group, mean and 95 % bootstrap interval over
  seeds of the Premise 1 quantities: gradient geometry over the first
  "informative" iterations (dense and final gradients both non-zero), the
  iteration from which success is sustained, and final performance;
- `curves_*.png`: learning curves, cosine traces, norm ratios, Gram eigenvalues
  and per-module cosines, one line per group with the interval as a band.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

# Reference categorical palette (fixed order, colorblind-validated).
PALETTE: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
COSINE_KEYS: tuple[str, ...] = (
    "grad/cos/dense_sparse",
    "grad/cos/dense_final",
    "grad/cos/sparse_final",
)
RATIO_KEYS: tuple[str, ...] = ("grad/norm_ratio/dense_sparse", "grad/norm_ratio/dense_final")
EIGEN_KEYS: tuple[str, ...] = (
    "grad/gram_eig/0",
    "grad/gram_eig/1",
    "grad/gram_eig/2",
    "grad/gram_eig/3",
)
MODULE_KEYS: tuple[str, ...] = ("grad/encoder/cos/dense_final", "grad/actor/cos/dense_final")
CURVE_KEYS: tuple[str, ...] = ("episode/return", "episode/success", "episode/return_final")
SUCCESS_KEY = "episode/success"


@dataclass
class Run:
    path: Path
    config: dict[str, Any]
    metrics: dict[str, np.ndarray]  # each (num_iterations,), NaN where absent

    @property
    def seed(self) -> int:
        return int(self.config.get("seed", 0))


@dataclass
class Group:
    name: str
    runs: list[Run] = field(default_factory=list)


# ------------------------------------------------------------------ loading
def flatten(config: dict[Any, Any], prefix: str = "") -> dict[str, Any]:
    """Nested dict -> {"a.b.c": value}."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def load_run(metrics_path: Path) -> Run | None:
    """The run at `metrics_path`, or None (with a warning) if it has no iteration yet."""
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if not rows:
        log.warning("skipping %s: no iteration logged yet", metrics_path)
        return None
    keys = sorted({k for row in rows for k in row})
    metrics = {k: np.array([row.get(k, np.nan) for row in rows], dtype=float) for k in keys}
    config_path = metrics_path.parent / ".hydra" / "config.yaml"
    container = OmegaConf.to_container(OmegaConf.load(config_path)) if config_path.exists() else {}
    config = flatten(container) if isinstance(container, dict) else {}
    return Run(metrics_path.parent, config, metrics)


def load_runs(root: Path) -> list[Run]:
    paths = sorted(p for p in root.rglob("metrics.jsonl") if "analysis" not in p.parts)
    if not paths:
        msg = f"no metrics.jsonl under {root}"
        raise FileNotFoundError(msg)
    runs = [run for run in map(load_run, paths) if run is not None]
    if not runs:
        msg = f"no completed run under {root}"
        raise ValueError(msg)
    return runs


def filter_runs(runs: Sequence[Run], filters: Sequence[str]) -> list[Run]:
    """Keep the runs whose config matches every `key=value` filter (string comparison)."""
    pairs = [f.split("=", 1) for f in filters]
    return [run for run in runs if all(str(run.config.get(key)) == value for key, value in pairs)]


def group_runs(runs: Sequence[Run], keys: Sequence[str] | None = None) -> list[Group]:
    """Group by the config keys that vary across runs (or the given keys), ignoring seed."""
    if keys is None:
        all_keys = sorted({k for run in runs for k in run.config} - {"seed"})
        keys = [k for k in all_keys if len({repr(run.config.get(k)) for run in runs}) > 1]
    groups: dict[str, Group] = {}
    for run in runs:
        name = ", ".join(f"{k.split('.')[-1]}={run.config.get(k)}" for k in keys) or "all"
        groups.setdefault(name, Group(name)).runs.append(run)
    return list(groups.values())


# --------------------------------------------------------------- statistics
def bootstrap_interval(
    values: np.ndarray, rng: np.random.Generator, samples: int = 2000
) -> tuple[float, float, float]:
    """(mean, low, high) of a 95 % percentile bootstrap over the finite values."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = rng.choice(finite, size=(samples, finite.size), replace=True).mean(axis=1)
    return float(finite.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def time_to_success(run: Run, threshold: float, window: int) -> float:
    """First iteration from which the success rate averages >= `threshold` over `window`
    iterations (NaN-aware, so iterations without a finished episode do not count); NaN if never."""
    success = run.metrics.get(SUCCESS_KEY)
    if success is None or len(success) < window:
        return float("nan")
    for t in range(len(success) - window + 1):
        chunk = success[t : t + window]
        finite = chunk[np.isfinite(chunk)]
        if finite.size and finite.mean() >= threshold:
            return float(t)
    return float("nan")


def tail_mean(values: np.ndarray, window: int) -> float:
    finite = values[np.isfinite(values)]
    return float(finite[-window:].mean()) if finite.size else float("nan")


def informative_mask(run: Run, count: int, eps: float = 1e-12) -> np.ndarray:
    """Mask of the first `count` iterations where dense and final gradients are both non-zero.

    A channel with no reward yet has an exactly-zero gradient (zero-initialised
    channel critic), which makes its cosines 0 and its norm ratios degenerate.
    """
    dense = run.metrics.get("grad/norm/dense")
    final = run.metrics.get("grad/norm/final")
    if dense is None or final is None:
        return np.zeros(0, dtype=bool)
    informative = (dense > eps) & (final > eps)
    keep = np.flatnonzero(informative)[:count]
    mask = np.zeros_like(informative)
    mask[keep] = True
    return mask


def run_summary(run: Run, *, first: int, last: int, threshold: float) -> dict[str, float]:
    m = run.metrics
    nan = np.full(1, np.nan)
    mask = informative_mask(run, first)
    cos_df = m.get("grad/cos/dense_final", nan)
    cos_ds = m.get("grad/cos/dense_sparse", nan)
    ratio_df = m.get("grad/norm_ratio/dense_final", nan)
    ratio_ds = m.get("grad/norm_ratio/dense_sparse", nan)
    with np.errstate(all="ignore"):
        early = {
            "first_informative_iter": float(np.flatnonzero(mask)[0]) if mask.any() else np.nan,
            f"cos_dense_final@early{first}": float(cos_df[mask].mean()) if mask.any() else np.nan,
            f"frac_neg_dense_final@early{first}": (
                float((cos_df[mask] < 0).mean()) if mask.any() else np.nan
            ),
            f"cos_dense_sparse@early{first}": float(cos_ds[mask].mean()) if mask.any() else np.nan,
            # Geometric means: ratios are log-scale quantities.
            f"norm_ratio_dense_final@early{first}": (
                float(np.exp(np.log(ratio_df[mask]).mean())) if mask.any() else np.nan
            ),
            f"norm_ratio_dense_sparse@early{first}": (
                float(np.exp(np.log(ratio_ds[mask]).mean())) if mask.any() else np.nan
            ),
        }
    return {
        **early,
        "time_to_success": time_to_success(run, threshold, window=last),
        f"success@last{last}": tail_mean(m.get(SUCCESS_KEY, nan), last),
        f"return@last{last}": tail_mean(m.get("episode/return", nan), last),
    }


def summarize(
    groups: Sequence[Group], *, first: int, last: int, threshold: float, seed: int = 0
) -> tuple[list[str], list[dict[str, Any]]]:
    """Per-group rows of `mean [low, high] (n)` strings plus the numeric columns."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    for group in groups:
        per_run = [run_summary(r, first=first, last=last, threshold=threshold) for r in group.runs]
        columns = list(per_run[0])
        row: dict[str, Any] = {"group": group.name, "seeds": len(group.runs)}
        for column in columns:
            values = np.array([s[column] for s in per_run])
            mean, low, high = bootstrap_interval(values, rng)
            row[column] = mean
            row[f"{column}_low"] = low
            row[f"{column}_high"] = high
            row[f"{column}_n"] = int(np.isfinite(values).sum())
        rows.append(row)
    return columns, rows


def write_summary(rows: Sequence[dict[str, Any]], columns: Sequence[str], out: Path) -> None:
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| group | seeds | " + " | ".join(columns) + " |", "|---|---|" + "---|" * len(columns)]
    for row in rows:
        cells = [
            f"{row[c]:.3g} [{row[f'{c}_low']:.3g}, {row[f'{c}_high']:.3g}] (n={row[f'{c}_n']})"
            for c in columns
        ]
        lines.append(f"| {row['group']} | {row['seeds']} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------- figures
def _stack(runs: Iterable[Run], key: str) -> np.ndarray | None:
    series = [r.metrics[key] for r in runs if key in r.metrics]
    if not series:
        return None
    length = min(len(s) for s in series)
    return np.stack([s[:length] for s in series])


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def plot_panels(
    groups: Sequence[Group],
    keys: Sequence[str],
    out: Path,
    name: str,
    *,
    smooth: int,
    logy: bool = False,
) -> Path | None:
    """One panel per key, one line per group (mean over seeds, band = seed range)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in keys if any(k in r.metrics for g in groups for r in g.runs)]
    if not keys:
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(4.8 * len(keys), 3.4), squeeze=False)
    for ax, key in zip(axes[0], keys, strict=True):
        for color, group in zip(PALETTE, groups, strict=False):
            stacked = _stack(group.runs, key)
            if stacked is None:
                continue
            with warnings.catch_warnings():
                # All-NaN columns (no finished episode that iteration) are expected.
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(stacked, axis=0)
                low = np.nanmin(stacked, axis=0)
                high = np.nanmax(stacked, axis=0)
            valid = np.isfinite(mean)
            if not valid.any():
                continue  # no finished episode in any iteration: nothing to draw
            x = np.arange(len(mean))[valid]
            ax.plot(x, _smooth(mean[valid], smooth), color=color, linewidth=2, label=group.name)
            ax.fill_between(
                x,
                _smooth(low[valid], smooth),
                _smooth(high[valid], smooth),
                color=color,
                alpha=0.15,
                linewidth=0,
            )
        ax.set_title(key, fontsize=10)
        ax.set_xlabel("iteration")
        if logy:
            ax.set_yscale("log")
        ax.grid(color="#e6e5e1", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    if len(groups) > 1 and axes[0][0].get_legend_handles_labels()[0]:
        axes[0][0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = out / f"curves_{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------- main
def analyze(
    root: Path,
    *,
    group_keys: Sequence[str] | None = None,
    filters: Sequence[str] = (),
    first: int = 10,
    last: int = 10,
    threshold: float = 0.9,
    smooth: int = 5,
    out: Path | None = None,
) -> Path:
    """Write the summary and figures for the runs under `root`; return the output dir."""
    runs = filter_runs(load_runs(root), filters)
    if not runs:
        msg = f"no run under {root} matches {list(filters)}"
        raise ValueError(msg)
    groups = group_runs(runs, group_keys)
    out = out or root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("curves_*.png"):  # never leave figures of a previous analysis
        stale.unlink()
    columns, rows = summarize(groups, first=first, last=last, threshold=threshold)
    write_summary(rows, columns, out)
    if len(groups) > len(PALETTE):
        log.warning(
            "%d groups exceed the %d palette slots: summary only, no figures (use --filter)",
            len(groups),
            len(PALETTE),
        )
        return out
    for name, keys, logy in (
        ("learning", CURVE_KEYS, False),
        ("cosines", COSINE_KEYS, False),
        ("norm_ratios", RATIO_KEYS, True),
        ("gram_eigenvalues", EIGEN_KEYS, True),
        ("modules", MODULE_KEYS, False),
    ):
        plot_panels(groups, keys, out, name, smooth=smooth, logy=logy)
    log.info("%d runs in %d groups -> %s", len(runs), len(groups), out)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="run directory or sweep root")
    parser.add_argument(
        "--group", help="comma-separated config keys to group by (default: varying keys)"
    )
    parser.add_argument(
        "--first", type=int, default=10, help="iterations for the early-training averages"
    )
    parser.add_argument(
        "--last", type=int, default=10, help="iterations for the final-performance averages"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.9, help="episode success rate that counts as solved"
    )
    parser.add_argument(
        "--smooth", type=int, default=5, help="moving-average window for the curves"
    )
    parser.add_argument("--out", type=Path, help="output directory (default: <root>/analysis)")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="keep only runs whose config has KEY=VALUE (repeatable)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    keys = args.group.split(",") if args.group else None
    out = analyze(
        args.root,
        group_keys=keys,
        filters=args.filter,
        first=args.first,
        last=args.last,
        threshold=args.threshold,
        smooth=args.smooth,
        out=args.out,
    )
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
