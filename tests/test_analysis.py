import json

import numpy as np
import pytest

from mamora.analysis import analyze, filter_runs, group_runs, load_runs, time_to_success


def _write_run(root, name, config, success, cos_dense_final, norm_dense, norm_final):
    run = root / name
    (run / ".hydra").mkdir(parents=True)
    (run / ".hydra" / "config.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in config.items() if not isinstance(v, dict))
        + "\nenv:\n  params:\n"
        + "\n".join(f"    {k}: {v}" for k, v in config["env"]["params"].items())
        + "\n"
    )
    with (run / "metrics.jsonl").open("w") as f:
        for t in range(len(success)):
            row = {
                "iteration": t,
                "timesteps": (t + 1) * 10,
                "episode/success": success[t],
                "episode/return": 1.0,
                "grad/cos/dense_final": cos_dense_final[t],
                "grad/cos/dense_sparse": 0.0,
                "grad/norm/dense": norm_dense[t],
                "grad/norm/final": norm_final[t],
                "grad/norm_ratio/dense_final": norm_dense[t] / (norm_final[t] + 1e-8),
                "grad/norm_ratio/dense_sparse": 1.0,
            }
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def sweep(tmp_path):
    T = 20
    for seed in (0, 1):
        # Tempted runs: never solved, dense/final conflict, zero final gradient at t=0.
        _write_run(
            tmp_path,
            f"h0.3_s{seed}",
            {"seed": seed, "env": {"params": {"harvest_reward": 0.3}}},
            success=[0.0] * T,
            cos_dense_final=[0.0] + [-0.5] * (T - 1),
            norm_dense=[1.0] * T,
            norm_final=[0.0] + [0.1] * (T - 1),
        )
        # Easy runs: one lucky early episode, then sustained success from t=5.
        _write_run(
            tmp_path,
            f"h0.0_s{seed}",
            {"seed": seed, "env": {"params": {"harvest_reward": 0.0}}},
            success=[float("nan"), 1.0, 0.0, 0.0, 0.0] + [1.0] * (T - 5),
            cos_dense_final=[0.0] * T,
            norm_dense=[0.0] * T,
            norm_final=[0.1] * T,
        )
    (tmp_path / "h0.3_s9" / ".hydra").mkdir(parents=True)  # in-progress run: no metrics yet
    (tmp_path / "h0.3_s9" / "metrics.jsonl").write_text("")
    return tmp_path


def test_load_group_filter(sweep):
    runs = load_runs(sweep)
    assert len(runs) == 4  # the empty run is skipped
    groups = group_runs(runs)
    assert sorted(g.name for g in groups) == ["harvest_reward=0.0", "harvest_reward=0.3"]
    assert all(len(g.runs) == 2 for g in groups)
    assert len(filter_runs(runs, ["env.params.harvest_reward=0.3"])) == 2


def test_time_to_success_requires_a_sustained_window(sweep):
    easy = next(r for r in load_runs(sweep) if r.config["env.params.harvest_reward"] == 0.0)
    assert time_to_success(easy, 0.9, window=3) == 5.0  # the lucky episode at t=1 is ignored
    assert time_to_success(easy, 0.9, window=1) == 1.0


def test_analyze_writes_summary_and_figures(sweep):
    out = analyze(sweep, first=5, last=3, threshold=0.9)
    summary = (out / "summary.md").read_text()
    assert "harvest_reward=0.3" in summary
    for name in ("learning", "cosines", "norm_ratios"):
        assert (out / f"curves_{name}.png").exists()
    rows = (out / "summary.csv").read_text().splitlines()
    header = rows[0].split(",")
    tempted = dict(zip(header, next(r for r in rows[1:] if "0.3" in r).split(","), strict=True))
    # The zero-final-gradient iteration 0 is excluded from the early window.
    assert float(tempted["first_informative_iter"]) == 1.0
    assert np.isclose(float(tempted["cos_dense_final@early5"]), -0.5)
    assert np.isnan(float(tempted["time_to_success"]))
    assert float(tempted["success@last3"]) == 0.0


def test_analyze_rejects_empty_filter(sweep):
    with pytest.raises(ValueError, match="matches"):
        analyze(sweep, filters=["env.params.harvest_reward=9"])
