# Experiment Report: exp_031

作成日: 2026-03-11  
対象: `experiments/exp_031/runbook.md`  
目的: CQ-0162（reward scale 経路修正）後の baseline 1 条件を再取得し、`reward / point_delta_reward / shanten_delta_reward / delta_t` の単位整合と advantage 逆転挙動を再確認する

## 1. 実験概要

新規実行 1 条件:
- A: post-fix baseline
  - `model.hidden_dims=[256,128]`
  - `model.policy_tower.enabled=false`
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
- A: `runs/20260311_stage1_full_flat_mlp_imitation_then_ppo_batch_c69c6c51`

`success_count = 5/5`。

---

## 2. 単位整合チェック（最重要）

### 2.1 self-play の reward composition

mean ± std（seed=5）

| 指標 | exp_030 (pre-fix) | exp_031 (post-fix) |
|---|---:|---:|
| `point_delta.mean` | -13.7057 ± 0.8800 | -0.000879 ± 0.000058 |
| `shanten_delta.mean` | （参考） | +0.0000367 ± 0.0000014 |
| `total.mean` | （参考） | -0.000843 ± 0.000059 |

所見:
- `point_delta` の桁が **約 1e4 縮小**し、`point_delta_scale=0.0001` と整合するレンジに入った。
- CQ-0162 の修正（reward config 注入）は有効と判断できる。

### 2.2 shanten_diag の成分レンジ

mean ± std（seed=5）

| 群 | reward mean | point_delta_reward mean | shanten_delta_reward mean |
|---|---:|---:|---:|
| improve | +0.002895 ± 0.000210 | -0.002409 ± 0.000205 | +0.005304 ± 0.000010 |
| same | -0.000490 ± 0.000047 | -0.000490 ± 0.000047 | 0.000000 ± 0.000000 |
| worsen | -0.006502 ± 0.000217 | -0.000499 ± 0.000219 | -0.006003 ± 0.000016 |

所見:
- `point_delta_reward` / `shanten_delta_reward` ともに現実的な小さなレンジに収まっている。
- exp_030 のような `-39` レベルの異常は解消。

---

## 3. 通常評価

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| exp_030 baseline (pre-fix) | 3.4450 ± 0.0594 | -13740.7 ± 1377.7 | 0.04487 ± 0.00500 | 0.58170 ± 0.01400 |
| exp_031 baseline (post-fix) | 3.4150 ± 0.0653 | -13380.5 ± 951.2 | 0.05019 ± 0.00459 | 0.57715 ± 0.01242 |

`eval_before -> eval` の delta（exp_031）:

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| exp_031 A | +0.0783 ± 0.0371 | -1144.5 ± 463.7 | -0.00855 ± 0.00525 | +0.00676 ± 0.00953 |

所見:
- after 指標は exp_030 より改善したが、`eval_before -> eval` 悪化は依然残る。
- したがって「単位バグ修正で全問題が解決した」ではない。

---

## 4. imitation 指標

mean ± std（seed=5）

| 条件 | teacher_top1_match_rate | teacher_best_set_hit_rate | imitation value_loss |
|---|---:|---:|---:|
| exp_031 A | 0.18316 ± 0.00459 | 0.60217 ± 0.00514 | 0.05231 ± 0.00487 |

所見:
- imitation value loss は exp_030（約 9e6）から大幅低下。
- ここも報酬単位修正の影響が強く出ている。

---

## 5. 主診断: 更新安定性

mean ± std（seed=5）

| 条件 | clip_fraction | ratio_std | old_value_mean | new_value_mean | value_error_mean |
|---|---:|---:|---:|---:|---:|
| exp_030 A (pre-fix) | 0.58458 ± 0.01287 | 0.66952 ± 0.07465 | -26.98 ± 4.85 | -223.61 ± 19.41 | +223.30 ± 15.34 |
| exp_031 A (post-fix) | 0.09013 ± 0.01046 | 0.12834 ± 0.00659 | -0.2570 ± 0.0160 | -0.2227 ± 0.0127 | -0.03364 ± 0.00346 |

所見:
- 更新安定性指標は全体に大幅改善。
- pre-fix の「極端なスケール起因の不安定性」は解消された。

---

## 6. 主診断: shanten_diag（post-fix）

mean ± std（seed=5）

| 群 | count | reward mean | delta_t mean | return mean | old_value mean | value_error mean | advantage mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| improve | 24771.2 ± 175.8 | +0.002895 ± 0.000210 | +0.003515 ± 0.000542 | -0.22461 ± 0.01212 | -0.25346 ± 0.01473 | -0.02885 ± 0.00292 | -0.05316 ± 0.00780 |
| same | 75560.6 ± 316.9 | -0.000490 ± 0.000047 | +0.002865 ± 0.000524 | -0.22426 ± 0.01310 | -0.26224 ± 0.01709 | -0.03798 ± 0.00408 | +0.04811 ± 0.00739 |
| worsen | 21137.4 ± 159.2 | -0.006502 ± 0.000217 | -0.002839 ± 0.000272 | -0.22058 ± 0.01172 | -0.24609 ± 0.01392 | -0.02551 ± 0.00250 | -0.09022 ± 0.01294 |

メタ:
- `status = partial`
- `available_samples = 121469.2 ± 172.6`
- `unavailable_samples = 800.0 ± 0.0`

所見:
- **pre-fix で見えていた「improve が worst」構図は崩れた。**
  - `reward.mean` は `improve > same > worsen` の順で直感的な符号になった。
- ただし `advantage.mean` は
  - improve: 負
  - worsen: さらに負
  - same: 正
  という形になっており、単純な improve正 / worsen負 には未到達。

---

## 7. 主診断: turn_diag（post-fix）

mean ± std（seed=5）

| bucket | count | return mean | old_value mean | new_value mean | value_update_delta mean | value_error mean | advantage mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| early | 10649.6 ± 7.3 | -0.18534 ± 0.00919 | -0.16732 ± 0.01294 | -0.18530 ± 0.00964 | -0.01798 ± 0.01200 | +0.01802 ± 0.01023 | -0.57114 ± 0.11472 |
| mid | 10646.0 ± 12.5 | -0.19427 ± 0.01042 | -0.18866 ± 0.01136 | -0.20359 ± 0.01041 | -0.01493 ± 0.01132 | +0.00561 ± 0.00884 | -0.43356 ± 0.10410 |
| late | 100973.6 ± 162.9 | -0.23044 ± 0.01351 | -0.27367 ± 0.01884 | -0.22870 ± 0.01346 | +0.04497 ± 0.00658 | -0.04323 ± 0.00541 | +0.10595 ± 0.02299 |

所見:
- pre-fix の巨大な misfit パターンは消失。
- ただし turn ごとの符号構造は残っており、policy 更新の歪み自体は別途分析が必要。

---

## 8. 結論

1. **報酬スケールバグ（CQ-0162）修正は有効。**
   - `point_delta_reward` の桁は正常化し、診断値の解釈可能性が回復した。

2. **exp_030 の定量解釈は pre-fix 前提で扱う必要がある。**
   - 特に `reward/delta_t/value_error/advantage` の絶対値比較は post-fix と直接比較不可。

3. **post-fix でも PPO 後悪化（`eval_before -> eval`）は残る。**
   - したがって今後の主課題は「単位バグ」ではなく「更新則とターゲット整合の問題」に戻る。

4. **逆転現象は形を変えて残存。**
   - pre-fix のような極端な improve/worsen 逆転は解消したが、`advantage` の群構造はまだ学習上の歪みを示している。

---

## 9. 次アクション

1. `exp_031` を新しい baseline（post-fix 基準）として固定する。  
2. 次実験は 1 条件追加（`policy_tower_only`）で、post-fix baseline との差分を確認する。  
3. 比較軸は以下を優先する。  
   - `eval_before -> eval`  
   - `shanten_diag` の `reward / delta_t / advantage` 群構造  
   - `turn_diag` の `advantage` と `value_error` 符号
