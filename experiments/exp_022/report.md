# Experiment Report: exp_022

作成日: 2026-03-09  
対象: `experiments/exp_022/runbook.md`

## 1. 実験概要

- 目的:
  - `shanten_delta_reward + linear_decay + scale=0.01` を固定し、`mode=both` と `mode=improve_only` を比較して、悪化側 shaping が本当に必要かを切り分ける
- 実行方式:
  - baseline は `exp_021` 条件 A を参照流用
  - 新規実行は `mode=both` / `mode=improve_only` の 2 条件
- seeds:
  - `42,43,44,45,46`
- 比較条件:
  - A: 参照 baseline（`point_delta` のみ, `exp_021` A）
  - B: `mode=both`
  - C: `mode=improve_only`
- 主評価の優先順:
  - `Δavg_rank -> Δavg_score -> Δdeal_in_rate -> Δwin_rate`

## 2. 実行結果

| 条件 | mode | batch_dir | success |
|---|---|---|---:|
| A | baseline | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_07349dda` | 5/5（参照） |
| B | both | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_d822aa82` | 5/5 |
| C | improve_only | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_1a4910d2` | 5/5 |

注記:
- `B` は実質的に `exp_021` の `scale=0.01, mode=both` と同条件であり、結果確認のため再実行された。

## 3. 主評価

mean ± std（seed=5）

| 条件 | Δavg_rank | Δavg_score | Δdeal_in_rate | Δwin_rate |
|---|---:|---:|---:|---:|
| A | +0.0833 ± 0.0312 | -753.2 ± 1050.5 | -0.0016 ± 0.0091 | -0.0168 ± 0.0060 |
| B | +0.0683 ± 0.0733 | -657.5 ± 1247.4 | -0.0034 ± 0.0072 | -0.0140 ± 0.0068 |
| C | +0.0817 ± 0.0644 | -1061.5 ± 972.8 | +0.0030 ± 0.0120 | -0.0174 ± 0.0058 |

所見:
- **B (`mode=both`) が最良**。baseline より悪化幅を小さくし、4 指標すべてで改善。
- **C (`improve_only`) は baseline を更新できない**。`Δavg_score` と `Δdeal_in_rate` はむしろ悪化した。
- したがって、少なくとも現在の `scale=0.01, linear_decay` 条件では、**悪化側 shaping も必要** と見るのが自然。

## 4. 副評価

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A | 3.4567 ± 0.0815 | -13659.2 ± 1241.9 | 0.03959 ± 0.01027 | 0.57366 ± 0.01393 |
| B | 3.4417 ± 0.0641 | -13563.5 ± 1074.2 | 0.04241 ± 0.00674 | 0.57186 ± 0.01275 |
| C | 3.4550 ± 0.0642 | -13967.5 ± 887.3 | 0.03899 ± 0.00711 | 0.57827 ± 0.01649 |

所見:
- after 指標でも **B が最良**。
- C は `avg_rank` こそ baseline 近辺だが、`avg_score / win_rate / deal_in_rate` で明確に悪い。
- `improve_only` は「褒めるだけで十分」という仮説を支持しない。

## 5. 補助観測

### 5.1 learner 診断統計

| 条件 | clip_fraction | ratio_std | new_value_mean | value_error_mean | value_error_std |
|---|---:|---:|---:|---:|---:|
| A | 0.7874 ± 0.0135 | 0.7467 ± 0.0201 | -122.5850 ± 10.1732 | 148.3684 ± 12.1589 | 683.4299 ± 53.6140 |
| B | 0.7869 ± 0.0144 | 0.7438 ± 0.0204 | -122.5827 ± 10.1702 | 148.3679 ± 12.1590 | 683.4302 ± 53.6140 |
| C | 0.7929 ± 0.0107 | 0.7616 ± 0.0271 | -122.5306 ± 10.2151 | 148.3508 ± 12.1590 | 683.4293 ± 53.6139 |

所見:
- B は baseline と同等か僅かに穏やか。
- C は `clip_fraction / ratio_std` が明確に悪化方向。
- 主評価の差は learner 診断とも整合している。

### 5.2 reward 内訳

| 条件 | total mean | total std | total p90 | total p99 | shanten_delta mean | shanten_delta std | shanten_delta p90 | shanten_delta p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | -8.940431 | 217.563450 | 0.0000 | 0.00 | 0.000000 | 0.000000 | 0.0000 | 0.00 |
| B | -8.940394 | 217.563347 | 0.0052 | 0.01 | 0.000036 | 0.004115 | 0.0055 | 0.01 |
| C | -8.939358 | 217.563370 | 0.0052 | 0.01 | 0.001073 | 0.002517 | 0.0055 | 0.01 |

所見:
- `p90/p99` は B/C で同じだが、**平均と標準偏差の形が違う**。
- C は悪化側ペナルティを消したぶん、`shanten_delta mean` が大きく正に寄り続ける。
- つまり `improve_only` は reward を「良い方向へ密にする」のではなく、**一方向の正報酬へ偏らせる** 形になっている可能性が高い。

## 6. 総合結論

1. **`mode=both` を維持採用**。`scale=0.01, linear_decay` の標準条件はこれでよい。  
2. **`mode=improve_only` は不採用**。負側 shaping を消すと、reward は一見穏やかになるどころか、正報酬バイアスが強くなり、主評価・after・learner 診断のすべてで悪化した。  
3. sparse reward 主因説は維持されるが、shaping は「改善だけ褒めればよい」ほど単純ではなく、**悪化打牌への弱い負信号も必要** と分かった。

## 7. 今回の判断

- 採用:
  - `shanten_delta_reward`
  - `schedule=linear_decay`
  - `scale=0.01`
  - **`mode=both`**
- 見送り:
  - `mode=improve_only`

## 8. 次アクション

1. reward 条件は **`linear_decay + scale=0.01 + mode=both`** を固定する。  
2. その条件を基準に、残る悪化を **value/target 側** に戻って診断する。  
3. 次の実験では、可能な限り既存 batch/run を流用して条件数を最小化する。

## 9. 実行対応表

| condition | mode | batch_dir | 備考 |
|---|---|---|---|
| A | baseline | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_07349dda` | `exp_021` A を参照流用 |
| B | both | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_d822aa82` | `exp_021` C と同条件を再実行 |
| C | improve_only | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_1a4910d2` | 新規条件 |
