# Experiment Report: exp_049

作成日: 2026-03-15  
対象: `experiments/exp_049/runbook.md`  
目的: `policy_anchor(kl, coef=0.5) + entropy=0.0` で `10 cycle` まで改善傾向が持続するかを確認する

## 1. 実験概要

条件（1条件, 5 seeds: 42..46）:
- A: `anchor_kl_coef_05_entropy_0000_cycle10_eval100`

主設定:
- `training.multi_cycle.num_cycles=10`
- `training.multi_cycle.selfplay_matches_per_cycle=200`
- `training.multi_cycle.eval_each_cycle=true`
- `evaluation.num_matches=100`
- `training.policy_anchor.coef=0.5`
- `training.entropy_coef=0.0`

## 2. 実行結果

- batch_dir: （ローカル run）
- success: `5/5`
- driver: `completed=1, failed=0`

## 3. 主評価（after）

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| exp_049 A | **3.3455 ± 0.0143** | **-12529.6 ± 315.8** | **0.05348 ± 0.00464** | **0.57115 ± 0.00848** |

所見:
- 5 seed のばらつきは比較的小さく、特に `avg_rank` が安定。
- after 指標は `exp_048`（3cycle）よりさらに改善した。

## 4. eval_before -> eval 差分

`delta = eval.after - eval.before`（avg_rank は小さいほど良い）

| 指標 | mean ± std |
|---|---:|
| Δavg_rank | **-0.0125 ± 0.0177** |
| Δavg_score | **+100.75 ± 157.57** |
| Δwin_rate | +0.00113 ± 0.00230 |
| Δdeal_in_rate | -0.00225 ± 0.00330 |

seed別 `Δavg_rank`: `[+0.0050, -0.0175, -0.0225, +0.0100, -0.0375]`  
seed別 `Δavg_score`: `[-188.25, +155.5, +84.25, +171.75, +280.5]`

所見:
- 平均では `Δavg_rank<0` かつ `Δavg_score>0` を維持。
- 5 seed 中 3 seed で rank/score とも改善、2 seed は部分的に悪化。

## 5. cycle 推移（10 cycles）

aggregate.cycles（seed平均）

| cycle | eval avg_rank | eval avg_score | eval_diff_avg_rank |
|---:|---:|---:|---:|
| 0 | 3.3655 | -12721.10 | +0.0245 |
| 1 | 3.3680 | -12658.55 | +0.0025 |
| 2 | 3.3595 | -12656.50 | -0.0085 |
| 3 | 3.3595 | -12659.60 | -0.0000 |
| 4 | 3.3640 | -12635.30 | +0.0045 |
| 5 | 3.3590 | -12685.90 | -0.0050 |
| 6 | 3.3720 | -12894.60 | +0.0130 |
| 7 | 3.3545 | -12534.95 | -0.0175 |
| 8 | 3.3580 | -12630.35 | +0.0035 |
| 9 | 3.3455 | -12529.60 | -0.0125 |

所見:
- cycle 6 で一時的に大きく悪化するが、その後回復。
- 最終 cycle 9 は cycle 0 より良い位置まで改善。
- 「長く回すと単調に壊れ続ける」挙動は、この条件では確認されなかった。

## 6. `exp_048 B`（3cycle）との比較

参照: `experiments/exp_048/report.md`

| 指標 | exp_048 B (3cycle, 5seed) | exp_049 A (10cycle, 5seed) |
|---|---:|---:|
| after avg_rank | 3.3833 | **3.3455** |
| after avg_score | -13140.3 | **-12529.6** |
| Δavg_rank | -0.00833 | **-0.0125** |
| Δavg_score | +268.67 | +100.75 |

所見:
- after 指標（rank/score）は `exp_049` が明確に良い。
- 差分では `Δavg_score` は `exp_048 B` の方が大きいが、`exp_049` も正を維持。

## 7. 解釈

1. `anchor coef=0.5 + entropy=0.0` は、10 cycle でも平均改善を維持できた。  
2. 改善は単調ではなく、途中で悪化フェーズを挟むため、短期観測だけで判定すると誤る可能性が高い。  
3. 現時点では「改善傾向あり」と見てよく、次段の大規模検証に進む根拠が揃った。

## 8. 次アクション

1. 同一条件で `10 cycle × 20 seeds × eval=100` を実行し、統計的に確定させる。  
2. 併せて cycle 別の `eval_diff_avg_score` も出せると、回復局面の解釈がしやすくなる（現状は cycle別 rank差分のみ）。
