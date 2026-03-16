# Experiment Report: exp_043

作成日: 2026-03-14  
対象: [experiments/exp_043/runbook.md](/home/takeo1116/Git/majong-rl/experiments/exp_043/runbook.md)  
目的: 高速化後に `7 conditions × 20 seeds` を一括実行し、PPO 更新強度と `discard_ukeire_hint` の有効性を評価する

## 1. 実験概要

条件:
- A: baseline_off (`lr=1e-4, epochs=2, gae=0.90, clip=0.20, hint=false`)
- B: lr7e5_off (`7e-5, 2, 0.90, 0.20, hint=false`)
- C: lr5e5_off (`5e-5, 2, 0.90, 0.20, hint=false`)
- D: weak_update_off (`7e-5, 1, 0.90, 0.15, hint=false`)
- E: weaker_update_off (`5e-5, 1, 0.85, 0.15, hint=false`)
- F: baseline_on (`1e-4, 2, 0.90, 0.20, hint=true`)
- G: weak_update_on (`7e-5, 1, 0.90, 0.15, hint=true`)

共通:
- seeds: `42..61`（20 seeds）
- imitation/self-play: `200 / 200`
- model: `[512,256] + dual towers`
- reward shaping: `both, scale=0.01`

## 2. 実行結果

| 条件 | batch | success |
|---|---|---:|
| A | `runs/20260314_stage1_full_flat_mlp_imitation_then_ppo_batch_b41896eb` | 20/20 |
| B | `runs/20260314_stage1_full_flat_mlp_imitation_then_ppo_batch_72cb3258` | 20/20 |
| C | `runs/20260314_stage1_full_flat_mlp_imitation_then_ppo_batch_2a6a8fb6` | 20/20 |
| D | `runs/20260314_stage1_full_flat_mlp_imitation_then_ppo_batch_51d93569` | 20/20 |
| E | `runs/20260314_stage1_full_flat_mlp_imitation_then_ppo_batch_026f2472` | 20/20 |
| F | `runs/20260314_stage1_full_flat_mlp_imitation_then_ppo_batch_c4ca47a3` | 20/20 |
| G | `runs/20260314_stage1_full_flat_mlp_imitation_then_ppo_batch_6c3fd62d` | 20/20 |

全条件完走。

## 3. 主評価（通常評価）

mean ± std（seed=20）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A | 3.4013 ± 0.0619 | -13126.4 | 0.0488 | 0.5742 |
| B | 3.3963 ± 0.0797 | -13390.6 | 0.0474 | 0.5753 |
| C | 3.4112 ± 0.0599 | -13397.4 | 0.0486 | 0.5750 |
| D | 3.3937 ± 0.0676 | -13229.6 | 0.0487 | 0.5760 |
| E | **3.3779 ± 0.0588** | **-13035.0** | **0.0516** | 0.5746 |
| F | 3.4071 ± 0.0582 | -13319.4 | 0.0498 | 0.5757 |
| G | 3.3971 ± 0.0615 | -13164.0 | 0.0501 | **0.5746** |

所見:
- 総合最良は **E (weaker_update_off)**。
- A 比で `avg_rank -0.0233`、`avg_score +91.4`、`win_rate +0.0028`。
- `hint=true` 条件（F/G）は、今回の 20seed でも `hint=false` 最良条件を更新できなかった。

## 4. PPO/Imitation 診断

| 条件 | teacher_top1 | teacher_best_set | imitation value_loss | clip_fraction | ratio_std | value_error_mean |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.2228 | 0.7015 | 0.0245 | 0.1483 | 0.1540 | -0.0096 |
| B | 0.2232 | 0.7018 | 0.0254 | 0.0995 | 0.1307 | -0.0057 |
| C | 0.2236 | 0.7017 | 0.0251 | 0.0748 | 0.1099 | -0.0026 |
| D | 0.2230 | 0.7017 | 0.0243 | 0.0822 | 0.1140 | -0.0043 |
| E | 0.2231 | 0.7016 | 0.0240 | 0.0668 | 0.1024 | -0.0015 |
| F | 0.2325 | 0.7078 | 0.0237 | 0.1516 | 0.1576 | -0.0104 |
| G | 0.2318 | 0.7080 | 0.0248 | 0.0830 | 0.1137 | -0.0052 |

所見:
- `hint=true`（F/G）は imitation 指標を押し上げる傾向は再確認。
- ただし最終対戦指標では E を超えず、今回も「teacher 追従改善 = 対戦改善」にはならなかった。
- E は update 強度が弱めで (`clip_fraction`, `ratio_std` 低め)、最終成績が最良。

## 5. 実行時間

run 平均（1 seed あたり）

| 条件 | total_sec | imitation_sec | selfplay_sec | eval_before_sec | learner_sec | eval_sec |
|---|---:|---:|---:|---:|---:|---:|
| A | 70.57 | 24.26 | 23.99 | 7.76 | 8.36 | 6.20 |
| B | 68.91 | 22.95 | 24.06 | 7.87 | 7.95 | 6.08 |
| C | 69.12 | 23.10 | 24.11 | 7.94 | 8.02 | 5.95 |
| D | 68.31 | 23.10 | 23.96 | 7.65 | 7.61 | 6.00 |
| E | **68.55** | 23.04 | 23.94 | 7.89 | 7.57 | 6.10 |
| F | 72.02 | 24.78 | 23.86 | 7.54 | 8.41 | 7.43 |
| G | 73.73 | 24.66 | 23.93 | 8.24 | 7.95 | 8.96 |

所見:
- `hint=true` 条件は `hint=false` 条件より平均でやや遅い（+3〜7%程度）。
- それでも旧実装比では十分高速で、今回規模（7×20）を数時間で完走できた。

## 6. 総合結論

1. 20seed 集計でも、今回の設定では **E (`lr=5e-5, epochs=1, gae=0.85, clip=0.15, hint=false`)** が最良。
2. `discard_ukeire_hint=true` は imitation 指標を改善するが、最終対戦指標では優位を示せなかった。
3. 高速化により大規模比較が可能になり、これまで 5seed で曖昧だった差が判定しやすくなった。

## 7. 今回の判断

- 採用候補:
  - E 条件（弱め更新 + hint off）
- 保留:
  - `hint=true`（採否は別条件での再検証余地あり）
- 見送り:
  - C 条件（`lr=5e-5, epochs=2`）

## 8. 次アクション

1. E を新 baseline 候補として、次の特徴量実験に進む。
2. `hint=true` は「別の reward/target 設計」で再評価する。
3. 同規模比較を今後も続けるため、集計 UX 改善（CQ 起票済み）を先に入れる。

