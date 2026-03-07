# Experiment Report: exp_012

作成日: 2026-03-07  
対象: `experiments/exp_012/runbook.md`

## 1. 実験概要

目的: 暫定 baseline（`lr=0.0001, epochs=4, value_loss_coef=0.25, clip_epsilon=0.2`）を固定し、`training.batch_size` を比較する。

- A: `batch_size=128`
- B: `batch_size=256`（baseline）
- C: `batch_size=512`

実行方式:
- seeds `42..46`
- seed ごとに ref run を 1 本作成
- `imitation,selfplay,eval_before` を再利用して A/B/C を分岐
- 評価は `rotation`, `num_matches=50`

## 2. 実行結果

- `exp_012` 全 run 成功
- A/B/C 再利用 run で `eval_diff.json` 生成・4指標 delta 非nullを確認
- 対応表は `experiments/exp_012/run_map.json` を正とする

## 3. 主評価（eval_before -> eval の delta）

mean ± std（seed=5）

| 条件 | Δavg_rank | Δavg_score | Δdeal_in_rate | Δwin_rate |
|---|---:|---:|---:|---:|
| A (128) | +0.0800 ± 0.0789 | -1182.7 ± 846.4 | +0.00970 ± 0.00436 | -0.01716 ± 0.00761 |
| B (256) | +0.0430 ± 0.0453 | -689.6 ± 725.1 | +0.00563 ± 0.00864 | -0.01466 ± 0.00422 |
| C (512) | +0.0840 ± 0.0552 | -898.6 ± 599.8 | +0.01106 ± 0.00296 | -0.01515 ± 0.00693 |

所見:
- 主評価順（`Δavg_rank -> Δavg_score -> Δdeal_in_rate -> Δwin_rate`）では **B(256) が最良**。
- A/C はいずれも `Δavg_rank` が大きく悪化。
- `Δavg_score` でも B が最も損失が小さい。

## 4. after 指標（eval 後）

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A (128) | 3.4570 ± 0.0769 | -14267.9 ± 957.3 | 0.04033 ± 0.00650 | 0.57822 ± 0.01257 |
| B (256) | 3.4200 ± 0.0677 | -13774.8 ± 987.9 | 0.04283 ± 0.00602 | 0.57414 ± 0.01355 |
| C (512) | 3.4610 ± 0.0466 | -13983.8 ± 666.1 | 0.04235 ± 0.00779 | 0.57957 ± 0.01129 |

所見:
- after でも B が総合的に最良。
- C は `avg_score` が A より良いが、`avg_rank` と `deal_in_rate` が悪い。

## 5. 時間（再利用 run）

1 run あたり平均（sec）

| 条件 | total | learner | eval |
|---|---:|---:|---:|
| A (128) | 288.17 | 24.25 | 263.92 |
| B (256) | 279.00 | 15.69 | 263.30 |
| C (512) | 275.09 | 11.16 | 263.94 |

補足:
- learner 時間は `batch_size` 増加で短縮（更新回数減のため）
- ただし品質面では B が最良

## 6. 結論

1. **`batch_size=256` を維持採用**（delta/after ともに最良）。
2. `128` は不採用（悪化が大きい）。
3. `512` は速度面の利点はあるが、指標悪化のため現時点では採用しない。

## 7. 次アクション

1. baseline を `batch_size=256` のまま固定し、次ノブ（`gamma` / `gae_lambda`）へ進む。  
2. 速度重視の検証が必要なら、`512` は「探索枝」として別目的で再確認する。  
3. learner 比較は引き続き reuse 前提で運用する。
