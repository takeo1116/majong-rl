# Experiment Report: exp_040

作成日: 2026-03-17  
対象: `experiments/exp_040/runbook.md`  
目的: `discard_ukeire_hint`, `current_shanten`, `shape_hint` を同時に有効化した feature pack が、`exp_039 B` 基準で性能と診断値を改善するかを確認する

## 1. 実験概要

比較条件:
- A reference: `exp_039 B`
  - batch: （ローカル run）
  - encoder: `shanten_hint=true`
- B new feature pack ON
  - batch: （ローカル run）
  - encoder:
    - `discard_ukeire_hint=true`
    - `current_shanten=true`
    - `shape_hint=true`

共通（主要）:
- seeds: `42,43,44,45,46`
- `selfplay.imitation_matches=200`
- `training.imitation_epochs=8`
- `selfplay.num_matches=200`
- `training.epochs=2`
- `training.gamma=0.99`
- `training.gae_lambda=0.90`
- `training.entropy_coef=0.01`
- `training.exclude_post_riichi_discards.enabled=true`

B は `success_count = 5/5`。

## 2. 通常評価

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A ref (`exp_039 B`) | 3.4317 ± 0.0718 | -13520.0 ± 1291.7 | 0.04900 ± 0.01097 | 0.58320 ± 0.00705 |
| B feature pack ON | 3.4533 ± 0.0221 | -14045.3 ± 517.1 | 0.04861 ± 0.00513 | 0.58201 ± 0.00576 |

所見:
- B は after 指標で A を更新できなかった。
- `avg_rank` と `avg_score` はともに悪化しており、feature pack 同時ONはこの設定では採用根拠が弱い。
- `deal_in_rate` はほぼ同等だが、勝率改善も確認できない。

## 3. `eval_before` と `eval_before -> eval`

`eval_before`（seed=5 mean ± std）:

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A ref | 3.4083 ± 0.0486 | -13198.8 ± 1141.2 | 0.05201 ± 0.00877 | 0.57655 ± 0.00851 |
| B feature pack ON | 3.4450 ± 0.0382 | -13391.2 ± 545.4 | 0.05356 ± 0.00448 | 0.57628 ± 0.01470 |

`eval_before -> eval` の delta:

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| A ref | +0.0233 ± 0.0640 | -321.2 ± 768.9 | -0.00301 ± 0.00552 | +0.00665 ± 0.00605 |
| B feature pack ON | +0.0083 ± 0.0264 | -654.2 ± 513.4 | -0.00496 ± 0.00454 | +0.00573 ± 0.01659 |

所見:
- `eval_before` からすでに B は `avg_rank` / `avg_score` で不利。
- PPO 後の `avg_rank` 劣化幅は B の方が小さいが、`avg_score` 劣化幅はむしろ大きい。
- つまり「特徴量で初期位置が良くなった上で PPO 後も維持できる」という形にはなっていない。

## 4. imitation 指標

| 条件 | teacher_top1_match_rate | teacher_best_set_hit_rate | imitation value_loss |
|---|---:|---:|---:|
| A ref | 0.22758 ± 0.00223 | 0.70182 ± 0.00262 | 0.02436 ± 0.00128 |
| B feature pack ON | 0.23215 ± 0.00548 | 0.70818 ± 0.00144 | 0.02401 ± 0.00109 |

所見:
- teacher 追従は B の方がやや良い。
- 特に `teacher_best_set_hit_rate` は改善している。
- それでも対戦性能は伸びていないので、今回も **teacher への適合増加が対戦強度へ変換されていない**。

## 5. 主診断: PPO/value

| 条件 | clip_fraction | ratio_std | value_error_mean | value_error_std | old_value_mean | new_value_mean |
|---|---:|---:|---:|---:|---:|---:|
| A ref | 0.13675 ± 0.00725 | 0.14886 ± 0.00352 | -0.01179 ± 0.00028 | 0.09807 ± 0.00481 | -0.21824 ± 0.00756 | -0.20591 ± 0.00747 |
| B feature pack ON | 0.14271 ± 0.00642 | 0.15109 ± 0.00392 | -0.00802 ± 0.00195 | 0.09437 ± 0.00368 | -0.18993 ± 0.01752 | -0.18138 ± 0.01578 |

所見:
- update 強度は B の方がやや強い。
- `value_error_mean/std` は B の方が改善しており、critic 側はむしろ安定寄り。
- それでも最終性能は悪化しているため、失敗の本体は「value が壊れた」ではない。

## 6. `shanten_diag` / `turn_diag`

advantage mean（seed=5 mean ± std）

### 6.1 shanten_diag

| 群 | A ref | B feature pack ON |
|---|---:|---:|
| improve | -0.04068 ± 0.00615 | -0.04748 ± 0.00306 |
| same | +0.03627 ± 0.00363 | +0.03820 ± 0.00180 |
| worsen | -0.06719 ± 0.00928 | -0.06796 ± 0.00645 |

### 6.2 turn_diag

| 群 | A ref | B feature pack ON |
|---|---:|---:|
| early | -0.63949 ± 0.05189 | -0.58715 ± 0.02479 |
| mid | -0.52047 ± 0.02490 | -0.47971 ± 0.01606 |
| late | +0.12640 ± 0.00762 | +0.11693 ± 0.00426 |

所見:
- `turn_diag` は B の方が少し穏やかで、表面的には自然に見える。
- しかし対戦性能は改善しないため、ここでも **診断の見た目の良さだけでは採用判断できない**。

## 7. reward / exclusion と実行時間

### 7.1 立直後打牌除外

| 条件 | excluded_post_riichi_discards |
|---|---:|
| A ref | 3063.2 ± 106.6 |
| B feature pack ON | 3175.6 ± 149.3 |

### 7.2 reward composition（mean）

| 条件 | point_delta | shanten_delta | total |
|---|---:|---:|---:|
| A ref | -0.001052 ± 0.000063 | +0.000033 ± 0.000001 | -0.001019 ± 0.000063 |
| B feature pack ON | -0.001143 ± 0.000091 | +0.000034 ± 0.000002 | -0.001108 ± 0.000089 |

### 7.3 実行時間

| 条件 | total_duration_sec | imitation | selfplay | eval_before | learner | eval |
|---|---:|---:|---:|---:|---:|---:|
| A ref | 1521.7 ± 23.5 | 782.0 ± 11.9 | 49.4 ± 0.8 | 339.7 ± 8.9 | 8.5 ± 0.5 | 342.1 ± 5.0 |
| B feature pack ON | 2277.1 ± 40.1 | 956.3 ± 14.1 | 543.2 ± 9.0 | 388.1 ± 18.0 | 8.7 ± 0.1 | 380.7 ± 21.5 |

所見:
- feature pack ON で **総時間は約 1.50x** に増加。
- 特に `selfplay` が **約 11.0x** に悪化しており、これが最大の副作用。
- 性能も改善しないため、当時の次アクションとして `discard_ukeire_hint` の切り分けに進む判断は妥当だった。

## 8. 解釈

1. 新特徴量パックは imitation 指標と一部の診断を改善した。  
2. しかし `eval_before` / after の主要性能には繋がらず、むしろ `avg_rank` / `avg_score` は悪化した。  
3. 加えて `selfplay` 時間が極端に増加し、実験運用上のコストも大きかった。  
4. したがって、**3特徴量を同時に導入するのではなく、どれが効いてどれが重いかを切り分ける必要がある**。

## 9. 結論

- そのままの feature pack 同時ONは **不採用**。
- 主要理由:
  - 性能改善が確認できない
  - `selfplay` の実行時間悪化が大きい
- 次アクションとして `discard_ukeire_hint` のアブレーションへ進んだ判断は正しい。
