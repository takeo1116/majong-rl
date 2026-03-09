# Experiment Report: exp_020

作成日: 2026-03-09  
対象: `experiments/exp_020/runbook.md`

## 1. 実験概要

- 目的:
  - reward の sparse さが PPO 悪化の主因かを、`point_delta` 単独と `shanten_delta_reward` 追加条件の比較で直接検証する
- 実行方式:
  - 3 条件とも full batch
- seeds:
  - `42,43,44,45,46`
- 比較条件:
  - A: baseline reward (`point_delta` のみ)
  - B: `point_delta + shanten_delta_reward` (`schedule=constant`)
  - C: `point_delta + shanten_delta_reward` (`schedule=linear_decay`)
- 主評価の優先順:
  - `Δavg_rank -> Δavg_score -> Δdeal_in_rate -> Δwin_rate`

## 2. 実行結果

| 条件 | run/batch | success |
|---|---|---:|
| A | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_80096099` | 5/5 |
| B | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_32343a7d` | 5/5 |
| C | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_36dd423d` | 5/5 |

注記:
- 初回 driver 実行では override 記法の不整合で B 開始前に停止した。driver を修正後、全条件を再実行して本集計に使用した。

## 3. 主評価

mean ± std（seed=5）

| 条件 | Δavg_rank | Δavg_score | Δdeal_in_rate | Δwin_rate |
|---|---:|---:|---:|---:|
| A | +0.0833 ± 0.0349 | -753.2 ± 1174.5 | -0.0016 ± 0.0102 | -0.0168 ± 0.0067 |
| B | +0.0700 ± 0.0481 | -699.8 ± 972.0 | -0.0020 ± 0.0082 | -0.0147 ± 0.0073 |
| C | +0.0683 ± 0.0819 | -657.5 ± 1394.6 | -0.0034 ± 0.0080 | -0.0140 ± 0.0076 |

所見:
- 3 条件とも PPO 後は平均で悪化するが、**B/C は A より悪化幅が一貫して小さい**。
- 主評価優先順では **C（linear_decay）> B（constant）> A（baseline）**。
- 特に C は 4 指標すべてで A を上回った。

## 4. 副評価

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A | 3.4567 ± 0.0912 | -13659.2 ± 1388.5 | 0.03959 ± 0.01149 | 0.57366 ± 0.01558 |
| B | 3.4433 ± 0.0869 | -13605.8 ± 1337.3 | 0.04171 ± 0.01311 | 0.57325 ± 0.01270 |
| C | 3.4417 ± 0.0717 | -13563.5 ± 1201.0 | 0.04241 ± 0.00753 | 0.57186 ± 0.01426 |

所見:
- after 指標でも **C が全指標で最良**。
- B も A よりやや改善しているが、C の方が一貫性が高い。
- shaping を最後まで一定で残すより、**後半で弱めて point_delta 主体へ戻す方が自然** に見える。

## 5. 補助観測

- learner 診断統計:

| 条件 | clip_fraction | ratio_std | new_value_mean | value_error_mean | value_error_std |
|---|---:|---:|---:|---:|---:|
| A | 0.7874 ± 0.0151 | 0.7468 ± 0.0225 | -122.5850 ± 11.3739 | 148.3684 ± 13.5941 | 683.4299 ± 59.9422 |
| B | 0.7889 ± 0.0172 | 0.7558 ± 0.0347 | -122.5794 ± 11.3778 | 148.3676 ± 13.5941 | 683.4304 ± 59.9422 |
| C | 0.7869 ± 0.0161 | 0.7438 ± 0.0228 | -122.5827 ± 11.3707 | 148.3679 ± 13.5941 | 683.4302 ± 59.9422 |

  所見:
  - shaping 導入で learner 診断統計は大きくは変わらない。
  - B は `ratio_std` と `clip_fraction` がむしろ少し悪化。
  - C は baseline と同等か僅かに改善で、主評価改善とも整合する。

- reward 内訳:

| 条件 | total mean | total std | total p50 | total p90 | total p99 | shanten_delta mean | shanten_delta p90 | shanten_delta p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | -8.940431 | 217.563450 | 0.0 | 0.0 | 0.0 | 0.000000 | 0.0 | 0.0 |
| B | -8.940362 | 217.563254 | 0.0 | 0.0100 | 0.0100 | 0.000068 | 0.0100 | 0.0100 |
| C | -8.940394 | 217.563347 | 0.0 | 0.0052 | 0.0100 | 0.000036 | 0.0055 | 0.0100 |

  所見:
  - `point_delta` 側の分布は A/B/C で実質不変。
  - B/C では `shanten_delta` が `p90/p99` を持ち、**`total` の p90 が 0 から正値へ上がる**。
  - つまり shaping は「平均を大きく変える」のではなく、**non-zero reward の密度を少し増やす** 方向に働いている。
  - `linear_decay` は `constant` より shaping の平均寄与が小さいが、主評価はむしろ良い。

- 教師再現率:
  - imitation は全条件で同一 (`tie_aware_best_set`)
  - `teacher_top1_match_rate`: 0.1822 ± 0.0055
  - `teacher_best_set_hit_rate`: 0.6017 ± 0.0069

## 6. 総合結論

1. **最小 shaping reward は有効**。`shanten_delta_reward` を足すと、PPO 後悪化は baseline より一貫して小さくなった。  
2. **`linear_decay` が最良**。constant より寄与量は小さいが、主評価・after 指標とも最も自然だった。  
3. したがって、「reward が sparse すぎることが PPO 悪化の主因」という仮説は**かなり強化**された。ただし、悪化が完全に消えたわけではないため、主因の一部であっても全部ではない可能性は残る。

## 7. 今回の判断

- 採用:
  - `shanten_delta_reward` 路線
  - `schedule=linear_decay`
- 保留:
  - shaping scale の最適値
  - `mode=both` が最良かどうか
- 見送り:
  - baseline reward 単独を「十分」とみなす判断
  - `constant` shaping を次の基準にすること

## 8. 次アクション

1. `linear_decay` を基準に、**scale の小規模比較**（例: `0.005 / 0.01 / 0.02`）を行う。  
2. `mode=both` と `mode=improve_only` を比較し、悪化側 shaping が必要かを確認する。  
3. shaping で改善は出たが悪化は残っているため、次段では **reward shaping を固定したうえで value/target 側の診断** に戻る。

## 9. 実行対応表

run_map はローカル管理なので、比較に必要な対応はここへ転記する。

| condition | batch_dir | 備考 |
|---|---|---|
| A | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_80096099` | baseline reward |
| B | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_32343a7d` | shanten constant |
| C | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_36dd423d` | shanten linear decay |
