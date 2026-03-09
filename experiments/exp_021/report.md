# Experiment Report: exp_021

作成日: 2026-03-09  
対象: `experiments/exp_021/runbook.md`

## 1. 実験概要

- 目的:
  - `shanten_delta_reward + linear_decay` の scale 感度を確認し、reward shaping の実用域と過剰域を切り分ける
- 実行方式:
  - 5 条件とも full batch
- seeds:
  - `42,43,44,45,46`
- 比較条件:
  - A: baseline reward (`point_delta` のみ)
  - B: `scale=0.005`
  - C: `scale=0.01`
  - D: `scale=0.02`
  - E: `scale=0.1`（極端条件）
- 主評価の優先順:
  - `Δavg_rank -> Δavg_score -> Δdeal_in_rate -> Δwin_rate`

## 2. 実行結果

| 条件 | scale | batch_dir | success |
|---|---:|---|---:|
| A | - | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_07349dda` | 5/5 |
| B | 0.005 | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_db72e880` | 5/5 |
| C | 0.01 | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_7f6d81dd` | 5/5 |
| D | 0.02 | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_6ab69317` | 5/5 |
| E | 0.1 | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_e18aa35a` | 5/5 |

## 3. 主評価

mean ± std（seed=5）

| 条件 | Δavg_rank | Δavg_score | Δdeal_in_rate | Δwin_rate |
|---|---:|---:|---:|---:|
| A | +0.0833 ± 0.0312 | -753.2 ± 1050.5 | -0.0016 ± 0.0091 | -0.0168 ± 0.0060 |
| B | +0.0717 ± 0.0488 | -975.0 ± 955.5 | +0.0040 ± 0.0071 | -0.0155 ± 0.0067 |
| C | +0.0683 ± 0.0733 | -657.5 ± 1247.4 | -0.0034 ± 0.0072 | -0.0140 ± 0.0068 |
| D | +0.0650 ± 0.0464 | -856.8 ± 892.4 | -0.0018 ± 0.0082 | -0.0148 ± 0.0040 |
| E | +0.1067 ± 0.0752 | -1292.8 ± 1485.7 | +0.0047 ± 0.0141 | -0.0205 ± 0.0065 |

所見:
- **E (`scale=0.1`) は明確に悪い**。4 指標すべてで最下位で、過剰 shaping 条件として期待どおり崩れた。
- A 比較では **C (`0.01`) と D (`0.02`) が改善域**、B (`0.005`) は効果が弱い。
- 主評価優先順だけ機械的に見ると D が最上位だが、差は小さく、`Δavg_score` と副評価では C が優勢。

## 4. 副評価

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A | 3.4567 ± 0.0815 | -13659.2 ± 1241.9 | 0.03959 ± 0.01027 | 0.57366 ± 0.01393 |
| B | 3.4450 ± 0.0649 | -13881.0 ± 983.4 | 0.04097 ± 0.01131 | 0.57930 ± 0.01308 |
| C | 3.4417 ± 0.0641 | -13563.5 ± 1074.2 | 0.04241 ± 0.00674 | 0.57186 ± 0.01275 |
| D | 3.4383 ± 0.0449 | -13762.8 ± 929.3 | 0.04161 ± 0.00799 | 0.57349 ± 0.01231 |
| E | 3.4800 ± 0.0602 | -14198.8 ± 1104.1 | 0.03593 ± 0.00672 | 0.57992 ± 0.01836 |

所見:
- **after 指標では C が最も自然**。`avg_score / win_rate / deal_in_rate` で最良。
- D は `avg_rank` だけわずかに良いが、総合では C を更新する根拠は弱い。
- E は after 指標でも明確に悪く、`point_delta` を食う過剰 shaping の兆候と解釈できる。

## 5. 補助観測

### 5.1 learner 診断統計

| 条件 | clip_fraction | ratio_std | new_value_mean | value_error_mean | value_error_std |
|---|---:|---:|---:|---:|---:|
| A | 0.7874 ± 0.0135 | 0.7467 ± 0.0201 | -122.5850 ± 10.1732 | 148.3684 ± 12.1589 | 683.4299 ± 53.6140 |
| B | 0.7840 ± 0.0172 | 0.7427 ± 0.0217 | -122.5813 ± 10.1731 | 148.3681 ± 12.1590 | 683.4300 ± 53.6140 |
| C | 0.7869 ± 0.0144 | 0.7438 ± 0.0204 | -122.5827 ± 10.1702 | 148.3679 ± 12.1590 | 683.4302 ± 53.6140 |
| D | 0.7901 ± 0.0139 | 0.7530 ± 0.0260 | -122.5833 ± 10.1742 | 148.3675 ± 12.1590 | 683.4304 ± 53.6139 |
| E | 0.7934 ± 0.0132 | 0.7630 ± 0.0219 | -122.5208 ± 10.2256 | 148.3642 ± 12.1591 | 683.4327 ± 53.6138 |

所見:
- B/C は baseline と同等か、やや穏やか。
- D から `clip_fraction / ratio_std` が悪化方向へ戻り始める。
- E は learner 診断でも最悪で、過剰 shaping の副作用が見えている。

### 5.2 reward 内訳

| 条件 | total mean | total std | total p50 | total p90 | total p99 | shanten_delta mean | shanten_delta p90 | shanten_delta p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | -8.940431 | 217.563450 | 0.0 | 0.0000 | 0.00 | 0.000000 | 0.0000 | 0.00 |
| B | -8.940413 | 217.563398 | 0.0 | 0.0026 | 0.005 | 0.000018 | 0.0028 | 0.005 |
| C | -8.940394 | 217.563347 | 0.0 | 0.0052 | 0.010 | 0.000036 | 0.0055 | 0.010 |
| D | -8.940358 | 217.563245 | 0.0 | 0.0104 | 0.020 | 0.000073 | 0.0110 | 0.020 |
| E | -8.940068 | 217.562427 | 0.0 | 0.0520 | 0.100 | 0.000363 | 0.0550 | 0.100 |

所見:
- shaping scale にほぼ比例して `shanten_delta` と `total` の `p90/p99` が増える。
- `point_delta` 自体の分布は不変で、差は純粋に shaping 由来。
- E は `total p90=0.052`, `p99=0.1` まで持ち上がっており、`point_delta` より shaping が見えすぎる領域に入っている。

## 6. 総合結論

1. **`linear_decay` shaping の実用域は `0.01〜0.02`**。  
2. **総合推奨は `scale=0.01`**。主評価で D が僅差だが、after 指標と learner 診断を含めると C の方が自然。  
3. **`scale=0.1` は不採用**。過剰 shaping により主評価・after・learner 診断のすべてが悪化した。  
4. sparse reward 主因説はさらに強化された。reward を密にしすぎると逆に崩れることも確認できたため、今後は「適度な shaping」を前提に設計を詰める段階に入った。

## 7. 今回の判断

- 採用:
  - `shanten_delta_reward`
  - `schedule=linear_decay`
  - **暫定標準 scale = `0.01`**
- 保留:
  - `mode=both` と `mode=improve_only` の比較
  - `0.02` が特定指標だけで勝つ理由の再確認
- 見送り:
  - `scale=0.1`
  - `0.005` を標準にすること

## 8. 次アクション

1. `scale=0.01` を固定して、**`mode=both` vs `mode=improve_only`** を比較する。  
2. その結果で shaping 条件を固定し、残る悪化を **value/target 側** に戻って診断する。  
3. 必要なら `0.02` は確認用に残すが、次の基準点は `0.01` でよい。

## 9. 実行対応表

run_map はローカル管理なので、比較に必要な対応はここへ転記する。

| condition | scale | batch_dir |
|---|---:|---|
| A | - | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_07349dda` |
| B | 0.005 | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_db72e880` |
| C | 0.01 | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_7f6d81dd` |
| D | 0.02 | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_6ab69317` |
| E | 0.1 | `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_e18aa35a` |
