# Stage4 Experiment Report (Runbook 4)

作成日時: 2026-03-06 JST  
Source Runbook: `experiments/exp_004/runbook.md`

## 1. 実験概要

- 目的: rotation eval で seat バイアスを除去した A vs E 比較
- 比較対象:
  - A: PPO 直学習
  - E: 軽量 imitation -> PPO（交絡排除版）
- seeds: `42,43,44,45,46`
- 共通条件:
  - `selfplay.num_matches=200`
  - `training.epochs=4`
  - `evaluation.mode=rotation`
  - `evaluation.rotation_seats=[0,1,2,3]`
  - `evaluation.num_matches=50`
  - workers: `selfplay=10`, `evaluation=10`, `imitation=10`（Eのみ）
  - device: `training=cuda`, `selfplay=cpu`, `evaluation=cpu`

## 2. 対象 batch

- A: `runs/20260306_stage1_full_flat_mlp_ppo_batch_fd8b3343`
- E: `runs/20260306_stage1_full_flat_mlp_imitation_then_ppo_batch_3585de77`

両者とも `success_count=5/5`、`aggregate.eval_mode=rotation` を確認。

## 3. 最終指標（after eval, batch aggregate）

| 指標 | A (mean ± std) | E (mean ± std) | 差分 (E - A) |
|---|---:|---:|---:|
| avg_rank | 3.688 ± 0.0199 | 3.550 ± 0.0599 | -0.138 |
| avg_score | -17488.3 ± 934.0 | -15612.0 ± 1025.9 | +1876.3 |
| win_rate | 0.002624 ± 0.001218 | 0.023765 ± 0.008673 | +0.021141 |
| deal_in_rate | 0.595702 ± 0.020064 | 0.587362 ± 0.014395 | -0.008340 |

95% CI:

- A avg_rank: `[3.663326, 3.712674]`
- E avg_rank: `[3.475642, 3.624358]`
- A win_rate: `[0.001112, 0.004137]`
- E win_rate: `[0.012998, 0.034533]`

## 4. 実行時間（1 seed あたり平均）

| フェーズ | A (sec) | E (sec) |
|---|---:|---:|
| imitation | - | 45.774 |
| selfplay | 17.207 | 16.765 |
| eval_before | 257.859 | 263.378 |
| learner | 16.295 | 14.337 |
| eval | 259.175 | 264.469 |
| total | 550.536 | 604.725 |

所見:

- E は imitation 分だけ総時間が増加（+約54秒/seed）。
- rotation eval（eval_before + eval）が全体時間の大部分を占める。

## 5. 判定

Runbook 4 の 5-seed 比較では、E は A に対して全主指標で改善方向。

- `avg_rank` 改善（小さい）
- `avg_score` 改善（大きい）
- `win_rate` 改善（大きい）
- `deal_in_rate` 改善（小さい）

特に `avg_rank` と `win_rate` は CI もほぼ分離しており、E 優位傾向は強い。

## 6. 次アクション

1. この結論を確定するなら、同条件で 10 seed に拡張。
2. さらに厳密化するなら `evaluation.num_matches=100` で再検証。
