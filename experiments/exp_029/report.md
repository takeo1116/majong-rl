# Experiment Report: exp_029

作成日: 2026-03-11  
対象: `experiments/exp_029/runbook.md`  
目的: small model + `current_shanten=true` を baseline に、task-specific tower が `policy-value` 干渉を弱めるかを確認する

## 1. 実験概要

新規実行 4 条件:
- A: baseline
  - `policy_tower=false`
  - `value_tower=false`
- B: value tower only
  - `policy_tower=false`
  - `value_tower=true`
  - `value_tower.hidden_dim=128`
- C: policy tower only
  - `policy_tower=true`
  - `policy_tower.hidden_dim=128`
  - `value_tower=false`
- D: dual towers
  - `policy_tower=true`
  - `policy_tower.hidden_dim=128`
  - `value_tower=true`
  - `value_tower.hidden_dim=128`

共通固定:
- `feature_encoder.shanten_hint.enabled=true`
- reward shaping 標準
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.01`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
- `training.imitation_loss_mode=tie_aware_best_set`
- `training.imitation_value_warmstart.enabled=true`
- `training.imitation_value_warmstart.coef=0.1`
- `model.hidden_dims=[256,128]`
- `model.value_features.current_shanten.enabled=true`
- `training.epochs=4`
- `training.lr=1e-4`
- `training.value_loss_coef=0.25`
- seeds: `42,43,44,45,46`

batch:
- A: `runs/20260311_stage1_full_flat_mlp_imitation_then_ppo_batch_35a5049d`
- B: `runs/20260311_stage1_full_flat_mlp_imitation_then_ppo_batch_d89d3933`
- C: `runs/20260311_stage1_full_flat_mlp_imitation_then_ppo_batch_4fd392ec`
- D: `runs/20260311_stage1_full_flat_mlp_imitation_then_ppo_batch_366fc6bf`

全条件 `success_count = 5/5`。

## 2. 通常評価

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| `exp_025` 参照 | 3.3833 ± 0.0880 | -13269.2 ± 1585.1 | 0.04683 ± 0.01282 | 0.58175 ± 0.01120 |
| A baseline | 3.4450 ± 0.0594 | -13740.7 ± 1377.7 | 0.04487 ± 0.00500 | 0.58170 ± 0.01400 |
| B value tower only | 3.4900 ± 0.0469 | -14466.7 ± 547.0 | 0.03819 ± 0.00670 | 0.59647 ± 0.01142 |
| C policy tower only | 3.4133 ± 0.0684 | -13717.0 ± 1214.7 | 0.04722 ± 0.01264 | 0.58008 ± 0.00788 |
| D dual towers | 3.4400 ± 0.0785 | -13962.5 ± 1367.8 | 0.04371 ± 0.01454 | 0.58571 ± 0.01168 |

`eval_before -> eval` の delta:

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| A baseline | +0.0583 ± 0.0831 | -826.5 ± 1385.7 | -0.00758 ± 0.00338 | +0.00624 ± 0.00737 |
| B value tower only | +0.0483 ± 0.0701 | -488.8 ± 1081.7 | -0.00965 ± 0.00879 | +0.01020 ± 0.01414 |
| C policy tower only | +0.0467 ± 0.0636 | -780.5 ± 1120.8 | -0.00671 ± 0.01221 | +0.00416 ± 0.00632 |
| D dual towers | +0.0700 ± 0.0987 | -1128.2 ± 1422.7 | -0.00916 ± 0.01223 | +0.01446 ± 0.01160 |

所見:
- `current_shanten=true` の baseline A は、`exp_025` より悪化した。
- **最良は C policy tower only**。after 指標・delta 指標とも A/B/D の中で最も良い。
- B value tower only は `Δavg_rank` はわずかに最小だが、after 指標では最悪。通常評価改善には繋がっていない。
- D dual towers は更新安定性は最良だが、通常評価は baseline を明確には更新できなかった。

## 3. imitation 指標

mean ± std（seed=5）

| 条件 | teacher_top1_match_rate | teacher_best_set_hit_rate | imitation value_loss |
|---|---:|---:|---:|
| A baseline | 0.17983 ± 0.00838 | 0.58837 ± 0.00762 | 9.051e6 ± 4.546e5 |
| B value tower only | 0.20039 ± 0.02544 | 0.58279 ± 0.00506 | 9.045e6 ± 4.573e5 |
| C policy tower only | 0.18679 ± 0.01607 | 0.58457 ± 0.00695 | 9.042e6 ± 4.798e5 |
| D dual towers | 0.19573 ± 0.01282 | 0.58655 ± 0.00397 | 9.050e6 ± 4.408e5 |

所見:
- B/D は teacher top-1 を上げたが、通常評価には結びついていない。
- teacher 再現だけでは PPO 後の良し悪しを説明できない、という従来傾向は維持。

## 4. 主診断: 更新安定性

mean ± std（seed=5）

| 条件 | clip_fraction | ratio_std | value_error_mean |
|---|---:|---:|---:|
| A baseline | 0.5846 ± 0.0129 | 0.6695 ± 0.0746 | 223.30 ± 15.34 |
| B value tower only | 0.3696 ± 0.0277 | 0.5072 ± 0.1496 | 257.05 ± 16.74 |
| C policy tower only | 0.5564 ± 0.0852 | 0.5423 ± 0.1302 | 184.47 ± 14.74 |
| D dual towers | 0.3563 ± 0.0117 | 0.4015 ± 0.0501 | 187.42 ± 24.07 |

所見:
- **D dual towers は `clip_fraction` / `ratio_std` が最良**。shared trunk 干渉を減らす方向の設計意図には合っている。
- **C policy tower only も A より一貫して改善**。更新安定性改善と通常評価改善を両立した唯一の条件。
- B value tower only は更新安定性は改善したが、`value_error_mean` はむしろ悪化。value 側だけを逃がす設計は今回の small model 条件では逆効果。

## 5. 主診断: shanten_diag

mean ± std（seed=5）

| 条件 | improve adv mean | worsen adv mean | improve value_error mean | worsen value_error mean |
|---|---:|---:|---:|---:|
| A baseline | -0.0878 ± 0.0049 | +0.0599 ± 0.0050 | +297.69 ± 22.42 | +172.57 ± 12.03 |
| B value tower only | -0.0908 ± 0.0084 | +0.0609 ± 0.0040 | +341.61 ± 14.65 | +200.24 ± 13.71 |
| C policy tower only | -0.0835 ± 0.0059 | +0.0540 ± 0.0060 | +249.45 ± 20.59 | +142.42 ± 9.42 |
| D dual towers | -0.0857 ± 0.0029 | +0.0537 ± 0.0070 | +255.73 ± 30.86 | +144.36 ± 16.38 |

所見:
- 符号逆転自体は全条件で残った。
- ただし **C/D は A より improve/worsen 両群の value_error を明確に削減**した。
- **B は両群とも悪化**。value tower only は今回の仮説に沿わなかった。

## 6. 主診断: turn_diag

mean ± std（seed=5）

| 条件 | early value_error mean | mid value_error mean | late value_error mean | late advantage mean |
|---|---:|---:|---:|---:|
| A baseline | +102.30 ± 1.28 | +138.90 ± 2.06 | +245.72 ± 18.17 | -0.0264 ± 0.0014 |
| B value tower only | +123.08 ± 4.22 | +159.76 ± 6.80 | +282.81 ± 18.78 | -0.0275 ± 0.0017 |
| C policy tower only | +84.42 ± 1.75 | +114.18 ± 4.73 | +202.00 ± 17.28 | -0.0225 ± 0.0026 |
| D dual towers | +85.56 ± 8.53 | +115.42 ± 8.17 | +204.35 ± 27.75 | -0.0211 ± 0.0028 |

所見:
- `late` misfit は C/D で明確に改善した。
- B は全 bucket で悪化。巡目依存の value 推定にも悪影響。
- **turn_diag でも C と D が一貫して有望**。

## 7. 解釈

今回の 4 条件で見えたことはかなり明確。

1. **value tower only は不採用。**
   - 更新安定性は良くなるが、通常評価・`shanten_diag`・`turn_diag` がすべて悪化した。
   - 「value 側だけを逃がせば良い」という単純仮説は否定された。

2. **policy tower only は最有力。**
   - after 指標が最良。
   - `clip_fraction` / `ratio_std` も改善。
   - `shanten_diag` / `turn_diag` / global `value_error` も A より改善。
   - 現時点で、通常評価と診断の両面から最もバランスが良い。

3. **dual towers は診断改善は良いが、通常評価は policy tower only に届かない。**
   - 更新安定性は最良だが、after 指標では C を更新できない。
   - value tower を足した分だけ、まだ何か余計な自由度が入っている可能性がある。

4. **shared trunk 干渉仮説は部分的に支持。**
   - 少なくとも policy 側に task-specific tower を持たせると、更新安定性と通常評価が改善した。
   - ただし「value tower を足せばより良い」は成立しなかったので、干渉の中身は policy/value 対称ではない。

## 8. 結論

- **暫定採用候補は C policy tower only**
  - `model.policy_tower.enabled=true`
  - `model.policy_tower.hidden_dim=128`
  - `model.value_tower.enabled=false`
- **B value tower only は不採用**
- **D dual towers は保留**
  - 追加比較の対照としては残す価値があるが、現時点の主候補ではない

## 9. 次アクション

1. **C を新しい基準条件として再診断または軽い追試を行う。**
   - `exp_025` と同様の単条件診断で、`old_value/new_value/value_update_delta` を再確認する価値がある。
2. **D dual towers を続けるなら、tower hidden を小さくするなどの軽量化比較に絞る。**
3. 当面は、large model 路線や単純な PPO 弱化より、**small model + policy tower** を中心に次段を考える。
