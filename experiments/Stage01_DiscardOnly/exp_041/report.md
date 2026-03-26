# Experiment Report: exp_041

作成日: 2026-03-17  
対象: `experiments/exp_041/runbook.md`  
目的: `discard_ukeire_hint` を OFF にしたとき、`exp_040 B` で顕在化した実行時間悪化をどこまで解消できるか、また性能劣化なしで維持できるかを確認する

## 1. 実験概要

比較条件:
- A reference: `exp_040 B`
  - batch: （ローカル run）
  - encoder:
    - `discard_ukeire_hint=true`
    - `current_shanten=true`
    - `shape_hint=true`
- B new run: `discard_ukeire_hint=false`
  - batch: （ローカル run）
  - encoder:
    - `discard_ukeire_hint=false`
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

B は `success_count = 5/5`。

## 2. 主評価: 時間

mean ± std（seed=5）

| 指標 | A (`hint=true`) | B (`hint=false`) | 差分 |
|---|---:|---:|---:|
| total_duration_sec | 2277.1 ± 40.1 | 1529.0 ± 45.9 | **-32.9%** |
| imitation.duration_sec | 956.3 ± 14.1 | 786.0 ± 24.8 | **-17.8%** |
| selfplay.duration_sec | 543.2 ± 9.0 | 53.7 ± 3.0 | **-90.1%** |
| eval_before.duration_sec | 388.1 ± 18.0 | 337.6 ± 14.6 | **-13.0%** |
| learner.duration_sec | 8.7 ± 0.1 | 9.2 ± 0.3 | +5.9% |
| eval.duration_sec | 380.7 ± 21.5 | 342.4 ± 9.3 | **-10.1%** |

所見:
- 主要ボトルネックは想定どおり `discard_ukeire_hint` だった。
- 特に `selfplay` が **約 10.1x 高速化** しており、総時間も 3 分の 2 まで縮小。
- learner フェーズは元々短いため差は小さい。

## 3. 性能比較

### 3.1 通常評価

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A (`hint=true`) | 3.4533 ± 0.0221 | -14045.3 ± 517.1 | 0.04861 ± 0.00513 | 0.58201 ± 0.00576 |
| B (`hint=false`) | 3.3817 ± 0.0649 | -13380.2 ± 924.8 | 0.04761 ± 0.00318 | 0.58231 ± 0.01154 |

### 3.2 `eval_before`

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A (`hint=true`) | 3.4450 ± 0.0382 | -13391.2 ± 545.4 | 0.05356 ± 0.00448 | 0.57628 ± 0.01470 |
| B (`hint=false`) | 3.3667 ± 0.0394 | -12854.2 ± 865.6 | 0.05278 ± 0.00762 | 0.57415 ± 0.00516 |

### 3.3 `eval_before -> eval`

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| A (`hint=true`) | +0.0083 ± 0.0264 | -654.2 ± 513.4 | -0.00496 ± 0.00454 | +0.00573 ± 0.01659 |
| B (`hint=false`) | +0.0150 ± 0.0772 | -526.0 ± 1045.3 | -0.00517 ± 0.00908 | +0.00817 ± 0.00963 |

所見:
- `hint=false` は `eval_before` も after も **`avg_rank` / `avg_score` で明確に改善**。
- PPO 後 delta はほぼ同水準で、改善の主因は PPO 安定化というより **初期性能の改善**。
- つまり `discard_ukeire_hint` は「重いだけ」ではなく、この設定では性能面でもノイズ源になっていた可能性が高い。

## 4. imitation 指標

| 条件 | teacher_top1_match_rate | teacher_best_set_hit_rate | imitation value_loss |
|---|---:|---:|---:|
| A (`hint=true`) | 0.23215 ± 0.00548 | 0.70818 ± 0.00144 | 0.02401 ± 0.00109 |
| B (`hint=false`) | 0.22274 ± 0.00426 | 0.70198 ± 0.00260 | 0.02411 ± 0.00125 |

所見:
- teacher 追従は `hint=true` の方が良い。
- それでも対戦性能は `hint=false` が上回るので、ここでも **teacher 指標の改善がそのまま強さに繋がらない**。
- `exp_040` から続く傾向が再確認された。

## 5. learner 診断

| 条件 | clip_fraction | ratio_std | value_error_mean | value_error_std | old_value_mean | new_value_mean |
|---|---:|---:|---:|---:|---:|---:|
| A (`hint=true`) | 0.14271 ± 0.00642 | 0.15109 ± 0.00392 | -0.00802 ± 0.00195 | 0.09437 ± 0.00368 | -0.18993 ± 0.01752 | -0.18138 ± 0.01578 |
| B (`hint=false`) | 0.13989 ± 0.00934 | 0.14970 ± 0.00608 | -0.00889 ± 0.00144 | 0.09152 ± 0.00309 | -0.18344 ± 0.01970 | -0.17403 ± 0.01842 |

所見:
- PPO 診断は A/B で大差ない。
- `clip_fraction` と `ratio_std` は B がわずかに低く、update 強度はやや穏やか。
- ただし差は小さく、今回はやはり encoder 側の差分が主因と見るのが自然。

## 6. `shanten_diag` / `turn_diag`

advantage mean（seed=5 mean ± std）

### 6.1 shanten_diag

| 群 | A (`hint=true`) | B (`hint=false`) |
|---|---:|---:|
| improve | -0.04748 ± 0.00306 | -0.03458 ± 0.00777 |
| same | +0.03820 ± 0.00180 | +0.03569 ± 0.00338 |
| worsen | -0.06796 ± 0.00645 | -0.07183 ± 0.00613 |

### 6.2 turn_diag

| 群 | A (`hint=true`) | B (`hint=false`) |
|---|---:|---:|
| early | -0.58715 ± 0.02479 | -0.62085 ± 0.01728 |
| mid | -0.47971 ± 0.01606 | -0.49809 ± 0.02865 |
| late | +0.11693 ± 0.00426 | +0.12179 ± 0.00419 |

所見:
- `hint=false` は `worsen` をより負に寄せており、こちらの方が直感に合う。
- `turn_diag` でも early/mid の負と late の正が少し強まり、こちらも性能改善と整合的。
- `exp_040` では見えにくかった「診断と性能のズレ」が、ここではやや改善している。

## 7. reward / exclusion

### 7.1 立直後打牌除外

| 条件 | excluded_post_riichi_discards |
|---|---:|
| A (`hint=true`) | 3175.6 ± 149.3 |
| B (`hint=false`) | 2914.8 ± 231.5 |

### 7.2 reward composition（mean）

| 条件 | point_delta | shanten_delta | total |
|---|---:|---:|---:|
| A (`hint=true`) | -0.001143 ± 0.000091 | +0.000034 ± 0.000002 | -0.001108 ± 0.000089 |
| B (`hint=false`) | -0.001001 ± 0.000122 | +0.000033 ± 0.000002 | -0.000968 ± 0.000121 |

所見:
- `hint=false` の方が sample-level reward 平均も良い。
- 今回の性能改善は、速度だけでなく **自己対戦データの質の改善** とも整合している。

## 8. 解釈

1. `discard_ukeire_hint` は `exp_040` の実行時間悪化の主因だった。  
2. OFF にすると総時間が大きく縮み、特に `selfplay` はほぼ元の速度帯まで戻る。  
3. さらに、この設定では `eval_before` / after ともに改善した。  
4. したがって、少なくとも当時の実験条件では **`discard_ukeire_hint=false` を基準に戻す判断が妥当**。

## 9. 結論

- **採用候補: B (`discard_ukeire_hint=false`)**
- 理由:
  - 速度が大きく改善
  - 性能も悪化せず、むしろ改善
- `discard_ukeire_hint` は当面デフォルトから外し、必要なら将来の実装最適化後に再評価するのがよい。
