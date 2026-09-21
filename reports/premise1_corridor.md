# Premise 1 on the timescale corridor (2026-09-10)

Question: do the dense, sparse and final channels induce different, interfering learning
signals, and do the diagnostics detect it? Setup: `TimescaleCorridor` defaults
(invest_steps 5, commit_steps 3, horizon 32, milestone 1, final 10), PPO from
`conf/algo/ppo.yaml`, 195 iterations of 64 envs x 16 steps (2e5 steps), forensics every
iteration, 5 seeds per cell. Early statistics use the first 10 iterations where the dense
and final gradients are both non-zero. "Sustained success" = success rate >= 0.9 over 10
consecutive iterations. Intervals: 95 % bootstrap over seeds.

## Dense temptation sweep (invest pays 0)

| group | cos(dense, final), early | fraction negative | ‖g_dense‖/‖g_final‖ (geo. mean) | iteration of sustained success | solved seeds (success, last 10) | return, last 10 |
|---|---|---|---|---|---|---|
| harvest_reward=0.0 | n/a | n/a | n/a | 10.60 [10.00, 11.40] | 1.00 [1.00, 1.00] | 11.00 [11.00, 11.00] |
| harvest_reward=0.05 | -0.28 [-0.41, -0.15] | 0.74 [0.62, 0.84] | 3.67 [0.36, 9.61] | 12.50 [12.00, 13.50] (n=4) | 0.80 [0.40, 1.00] | 9.12 [5.36, 11.00] |
| harvest_reward=0.1 | -0.34 [-0.49, -0.18] | 0.80 [0.70, 0.88] | 15.15 [2.36, 27.94] | 13.00 [12.00, 14.00] (n=2) | 0.40 [0.00, 0.80] | 6.32 [3.20, 9.44] |
| harvest_reward=0.3 | -0.38 [-0.58, -0.23] | 0.76 [0.62, 0.90] | 75.54 [38.57, 112.51] | n/a | 0.00 [0.00, 0.00] | 9.60 [9.60, 9.60] |
| harvest_reward=1.0 | -0.09 [-0.34, 0.16] | 0.58 [0.38, 0.78] | 662.95 [338.50, 987.40] | n/a | 0.00 [0.00, 0.00] | 32.00 [32.00, 32.00] |

Optimal return is 11 (milestone 1 + final 10) for harvest <= 0.3, and 32 (harvest only)
for harvest = 1.0, where harvesting is the correct policy.

## Controls (invest pays the same dense reward as harvest)

| group | cos(dense, final), early | fraction negative | ‖g_dense‖/‖g_final‖ (geo. mean) | iteration of sustained success | solved seeds (success, last 10) | return, last 10 |
|---|---|---|---|---|---|---|
| harvest_reward=0.0, invest_reward=0.0 | n/a | n/a | n/a | 10.60 [10.00, 11.40] | 1.00 [1.00, 1.00] | 11.00 [11.00, 11.00] |
| harvest_reward=0.05, invest_reward=0.0 | -0.28 [-0.41, -0.15] | 0.74 [0.62, 0.84] | 3.67 [0.36, 9.61] | 12.50 [12.00, 13.50] (n=4) | 0.80 [0.40, 1.00] | 9.12 [5.36, 11.00] |
| harvest_reward=0.1, invest_reward=0.0 | -0.34 [-0.49, -0.18] | 0.80 [0.70, 0.88] | 15.15 [2.36, 27.94] | 13.00 [12.00, 14.00] (n=2) | 0.40 [0.00, 0.80] | 6.32 [3.20, 9.44] |
| harvest_reward=0.1, invest_reward=0.1 | -0.03 [-0.21, 0.16] | 0.50 [0.30, 0.70] | 2.01 [0.93, 3.78] | 13.25 [12.50, 14.00] (n=4) | 0.80 [0.40, 1.00] | 10.04 [7.12, 11.50] |
| harvest_reward=0.1, invest_reward=0.3 | 0.05 [-0.17, 0.27] | 0.48 [0.32, 0.62] | 4.01 [2.49, 5.53] | 18.00 [18.00, 18.00] (n=2) | 0.38 [0.00, 0.77] | 13.69 [10.60, 16.79] |
| harvest_reward=0.3, invest_reward=0.0 | -0.38 [-0.58, -0.23] | 0.76 [0.62, 0.90] | 75.54 [38.88, 112.51] | n/a | 0.00 [0.00, 0.00] | 9.60 [9.60, 9.60] |
| harvest_reward=0.3, invest_reward=0.1 | -0.31 [-0.41, -0.21] | 0.72 [0.62, 0.82] | 59.15 [22.30, 92.93] | n/a | 0.00 [0.00, 0.00] | 9.60 [9.60, 9.60] |
| harvest_reward=0.3, invest_reward=0.3 | 0.03 [-0.31, 0.32] | 0.52 [0.32, 0.74] | 14.84 [7.46, 21.32] | 22.00 [22.00, 22.00] (n=1) | 0.19 [0.00, 0.58] | 12.17 [10.60, 15.32] |
| harvest_reward=1.0, invest_reward=0.0 | -0.09 [-0.34, 0.16] | 0.58 [0.38, 0.80] | 662.95 [338.50, 1054.99] | n/a | 0.00 [0.00, 0.00] | 32.00 [32.00, 32.00] |

Note: after the milestone, `invest` still pays its dense reward while `commit` pays 0, so
the equal-reward control keeps a smaller conflict between dense and final. A clean control
needs a `commit_reward` parameter (dense paid equally for every action).

## Figures

`multirun/premise1/analysis_temptation/curves_*.png` and `multirun/premise1/analysis_all/curves_*.png`.
The early success rate of 1.0 in the learning curves is a selection artifact: before
iteration 2, only successful episodes can end inside a 16-step rollout.
