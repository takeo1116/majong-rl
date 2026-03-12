# Experiment Report: exp_032

作成日: 2026-03-11  
対象: `experiments/exp_032/runbook.md`  
目的: post-fix baseline (`exp_031`) に対して `policy_tower_only` を 1 条件だけ追加し、PPO 後悪化を縮められるかを確認する

## 1. 実験概要

新規実行 1 条件:
- A: post-fix policy tower only
  - `model.hidden_dims=[256,128]`
  - `model.policy_tower.enabled=true`
  - `model.policy_tower.hidden_dim=128`
  - `model.value_tower.enabled=false`
  - `model.value_features.current_shanten.enabled=true`

共通固定（主要）:
- `feature_encoder.shanten_hint.enabled=true`
- `training.imitation_loss_mode=tie_aware_best_set`
- `training.imitation_value_warmstart.enabled=true`
- `training.imitation_value_warmstart.coef=0.1`
- `reward.point_delta_scale=0.0001`
- `reward.shaping.shanten_delta.enabled=true`
- `reward.shaping.shanten_delta.scale=0.01`
- `reward.shaping.shanten_delta.mode=both`
- `reward.shaping.shanten_delta.schedule.type=linear_decay`
- `training.epochs=4`
- `training.lr=1e-4`
- seeds: `42,43,44,45,46`

batch:
- A: `runs/20260311_stage1_full_flat_mlp_imitation_then_ppo_batch_5d0451b7`

比較基準:
- `exp_031` post-fix baseline

`success_count = 5/5`。

## 2. 通常評価

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| exp_031 baseline | 3.4150 ± 0.0653 | -13380.5 ± 951.2 | 0.05019 ± 0.00459 | 0.57715 ± 0.01242 |
| exp_032 policy tower only | 3.4250 ± 0.0354 | -13609.7 ± 761.7 | 0.04696 ± 0.00873 | 0.57788 ± 0.00744 |

`eval_before -> eval` の delta:

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| exp_031 baseline | +0.0783 ± 0.0371 | -1144.5 ± 463.7 | -0.00855 ± 0.00525 | +0.00676 ± 0.00953 |
| exp_032 policy tower only | +0.0783 ± 0.0507 | -977.8 ± 910.4 | -0.00646 ± 0.00814 | -0.00025 ± 0.01270 |

所見:
- after 指標は baseline を更新できなかった。
- ただし `eval_before -> eval` の悪化幅は一部で改善した。
  - `Δavg_score` は改善
  - `Δwin_rate` は改善
  - `Δdeal_in_rate` は大幅改善（ほぼ 0）
  - `Δavg_rank` は baseline と同水準
- したがって「全面的に良い」とは言えないが、PPO 後悪化の形はやや穏やかになった。

## 3. imitation 指標

mean ± std（seed=5）

| 条件 | teacher_top1_match_rate | teacher_best_set_hit_rate | imitation value_loss |
|---|---:|---:|---:|
| exp_031 baseline | 0.18316 ± 0.00459 | 0.60217 ± 0.00514 | 0.05231 ± 0.00487 |
| exp_032 policy tower only | 0.18010 ± 0.00541 | 0.59371 ± 0.00872 | 0.04875 ± 0.00491 |

所見:
- teacher 再現率は少し下がった。
- imitation value loss はやや改善。
- pre-fix 時点と同様、teacher 再現だけでは PPO 後性能を説明しない。

## 4. 主診断: 更新安定性

mean ± std（seed=5）

| 条件 | clip_fraction | ratio_std | old_value_mean | new_value_mean | value_error_mean |
|---|---:|---:|---:|---:|---:|
| exp_031 baseline | 0.09013 ± 0.01046 | 0.12834 ± 0.00659 | -0.2570 ± 0.0160 | -0.2227 ± 0.0127 | -0.03364 ± 0.00346 |
| exp_032 policy tower only | 0.09650 ± 0.00962 | 0.13608 ± 0.00635 | -0.2674 ± 0.0185 | -0.2318 ± 0.0159 | -0.03508 ± 0.00277 |

所見:
- 更新安定性は baseline よりわずかに悪化。
- pre-fix `exp_029 C` で見えていた「安定性改善」は、post-fix では再現しなかった。

## 5. 主診断: shanten_diag

mean ± std（seed=5）

| 群 | reward mean | delta_t mean | return mean | old_value mean | value_error mean | advantage mean |
|---|---:|---:|---:|---:|---:|---:|
| exp_031 improve | +0.002895 ± 0.000210 | +0.003515 ± 0.000542 | -0.22461 ± 0.01212 | -0.25346 ± 0.01473 | -0.02885 ± 0.00292 | -0.05316 ± 0.00780 |
| exp_032 improve | +0.002927 ± 0.000193 | +0.003609 ± 0.001240 | -0.23357 ± 0.01582 | -0.26270 ± 0.01739 | -0.02913 ± 0.00188 | -0.05757 ± 0.00894 |
| exp_031 same | -0.000490 ± 0.000047 | +0.002865 ± 0.000524 | -0.22426 ± 0.01310 | -0.26224 ± 0.01709 | -0.03798 ± 0.00408 | +0.04811 ± 0.00739 |
| exp_032 same | -0.000526 ± 0.000039 | +0.002728 ± 0.000700 | -0.23418 ± 0.01600 | -0.27481 ± 0.01929 | -0.04063 ± 0.00336 | +0.05385 ± 0.00468 |
| exp_031 worsen | -0.006502 ± 0.000217 | -0.002839 ± 0.000272 | -0.22058 ± 0.01172 | -0.24609 ± 0.01392 | -0.02551 ± 0.00250 | -0.09022 ± 0.01294 |
| exp_032 worsen | -0.006394 ± 0.000074 | -0.002124 ± 0.000673 | -0.22687 ± 0.01556 | -0.25226 ± 0.01774 | -0.02539 ± 0.00276 | -0.09419 ± 0.01261 |

メタ:
- `status = partial`
- `available_samples = 121437.4 ± 341.3`
- `unavailable_samples = 800.0 ± 0.0`

所見:
- reward の符号構造自体は baseline とほぼ同じで、崩れていない。
- ただし `advantage.mean` は全群で baseline よりわずかに悪化寄り。
  - improve はより負
  - same はより正
  - worsen はより負
- つまり `policy_tower_only` は `shanten_diag` の群構造を自然化しなかった。

## 6. 主診断: turn_diag

mean ± std（seed=5）

| bucket | return mean | old_value mean | new_value mean | value_update_delta mean | value_error mean | advantage mean |
|---|---:|---:|---:|---:|---:|---:|
| exp_031 early | -0.18534 ± 0.00919 | -0.16732 ± 0.01294 | -0.18530 ± 0.00964 | -0.01798 ± 0.01200 | +0.01802 ± 0.01023 | -0.57114 ± 0.11472 |
| exp_032 early | -0.17781 ± 0.01527 | -0.12874 ± 0.02335 | -0.16780 ± 0.01959 | -0.03906 ± 0.01329 | +0.04907 ± 0.01329 | -0.81445 ± 0.06284 |
| exp_031 mid | -0.19427 ± 0.01042 | -0.18866 ± 0.01136 | -0.20359 ± 0.01041 | -0.01493 ± 0.01132 | +0.00561 ± 0.00884 | -0.43356 ± 0.10410 |
| exp_032 mid | -0.19487 ± 0.01501 | -0.16518 ± 0.02087 | -0.19786 ± 0.01905 | -0.03268 ± 0.01105 | +0.02969 ± 0.01069 | -0.62670 ± 0.05274 |
| exp_031 late | -0.23044 ± 0.01351 | -0.27367 ± 0.01884 | -0.22870 ± 0.01346 | +0.04497 ± 0.00658 | -0.04323 ± 0.00541 | +0.10595 ± 0.02299 |
| exp_032 late | -0.24206 ± 0.01637 | -0.29284 ± 0.02007 | -0.23579 ± 0.01961 | +0.05705 ± 0.00275 | -0.05078 ± 0.00403 | +0.15197 ± 0.01188 |

所見:
- turn ごとの歪みは baseline より悪化。
- 特に
  - early / mid の負 advantage がより強くなり
  - late の正 advantage もより強くなった
- `policy_tower_only` は post-fix では turn 依存の偏りをむしろ増やしている。

## 7. 解釈

今回の結果は、pre-fix 系列の `exp_029 C` とはかなり違う。

1. post-fix では `policy_tower_only` の通常評価優位は再現しなかった。  
2. `eval_before -> eval` の一部悪化幅は縮んだが、after 指標は baseline を更新できなかった。  
3. `shanten_diag` と `turn_diag` の観点では、むしろ baseline より少し悪い。  
4. したがって、pre-fix で見えていた `policy_tower_only` の優位には、reward scale バグが混ざっていた可能性が高い。  

## 8. 結論

- post-fix 環境では、**`policy_tower_only` を新たな採用候補とはしない**。
- `exp_031` baseline は依然として基準条件として維持する。
- `policy_tower_only` は「悪化幅の一部を減らす可能性はあるが、総合的に baseline を更新できない」条件とみなすのが妥当。

## 9. 次アクション

1. 次の本命候補は `dual_towers` の post-fix 再検証。  
2. ただし tower 系を続ける前に、`exp_031` baseline を起点に reward/target 側へ戻る判断も有力。  
3. 少なくとも、pre-fix 系列の構造改善結果はそのまま採用せず、post-fix で再検証する方針を徹底する。
