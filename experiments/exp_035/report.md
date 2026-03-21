# Experiment Report: exp_035

作成日: 2026-03-12  
対象: `experiments/exp_035/runbook.md`  
目的: 高表現力モデル + 立直後打牌除外条件で、`batch_size=512, epochs=2` により PPO 更新強度を適正化できるかを確認する

## 1. 実験概要

新規実行 1 条件:
- A: large-batch low-epoch dual towers
  - `model.hidden_dims=[512,256]`
  - `model.policy_tower.enabled=true`
  - `model.policy_tower.hidden_dim=128`
  - `model.value_tower.enabled=true`
  - `model.value_tower.hidden_dim=128`
  - `training.exclude_post_riichi_discards.enabled=true`
  - `training.batch_size=512`
  - `training.epochs=2`
  - `training.lr=0.0001`

共通固定（主要）:
- `feature_encoder.shanten_hint.enabled=true`
- `model.value_features.current_shanten.enabled=true`
- `training.imitation_loss_mode=tie_aware_best_set`
- `training.imitation_value_warmstart.enabled=true`
- `training.imitation_value_warmstart.coef=0.1`
- `reward.point_delta_scale=0.0001`
- `reward.shaping.shanten_delta.enabled=true`
- `reward.shaping.shanten_delta.scale=0.01`
- `reward.shaping.shanten_delta.mode=both`
- `reward.shaping.shanten_delta.schedule.type=linear_decay`
- seeds: `42,43,44,45,46`

batch:
- A: （ローカル run）

比較基準:
- `exp_034` high-capacity dual towers with exclusion

`success_count = 5/5`。

## 2. 通常評価

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| exp_034 high-cap dual towers | 3.4083 ± 0.0821 | -13565.8 ± 1067.0 | 0.04539 ± 0.00736 | 0.58747 ± 0.01115 |
| exp_035 batch512 epochs2 | 3.4100 ± 0.0838 | -13079.0 ± 1269.9 | 0.05033 ± 0.01109 | 0.57166 ± 0.01270 |

`eval_before -> eval` の delta:

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| exp_034 high-cap dual towers | -0.0117 ± 0.0366 | -105.3 ± 1016.0 | -0.00481 ± 0.00475 | +0.00569 ± 0.00476 |
| exp_035 batch512 epochs2 | +0.0550 ± 0.0938 | -358.0 ± 1084.2 | -0.00153 ± 0.00789 | -0.00105 ± 0.00897 |

所見:
- `eval_before -> eval` だけを見ると、`exp_034` より少し悪化。
  - `Δavg_rank` は再び正に戻った
  - `Δavg_score` もやや悪化
- ただし after 指標は **かなり改善**した。
  - `avg_score` は大幅改善
  - `win_rate` も改善
  - `deal_in_rate` は明確に改善
  - `avg_rank` はほぼ同等

つまり、**学習後の最終着地は `exp_034` より良い**。

## 3. 立直後打牌除外の実績

`post_riichi_exclusion.excluded_post_riichi_discards`:
- exp_034: `2816.4 ± 208.9`
- exp_035: `2544.2 ± 178.7`

所見:
- exclusion は今回も十分な件数で効いている。
- learner 診断は引き続き除外後サンプルに対して計算されている。

## 4. 主診断: 更新安定性

mean ± std（seed=5）

| 条件 | clip_fraction | ratio_std | value_error_mean | old_value_mean | new_value_mean |
|---|---:|---:|---:|---:|---:|
| exp_034 high-cap dual towers | 0.12041 ± 0.00551 | 0.15326 ± 0.00328 | -0.02805 ± 0.00257 | -0.22896 ± 0.00907 | -0.20022 ± 0.00731 |
| exp_035 batch512 epochs2 | 0.08796 ± 0.01068 | 0.12474 ± 0.00620 | -0.02798 ± 0.00433 | -0.22170 ± 0.01503 | -0.19276 ± 0.01099 |

所見:
- ここは **かなり改善**した。
- `clip_fraction` と `ratio_std` は `exp_034` から大きく低下し、ほぼ `exp_033` 水準まで戻った。
- `value_error_mean` はほぼ維持。

解釈:
- `batch_size↑ / epochs↓` は狙い通り **更新強度を弱める** 効果を出している。

## 5. 主診断: shanten_diag

advantage mean（seed=5 mean ± std）

| 群 | exp_034 | exp_035 |
|---|---:|---:|
| improve | -0.04597 ± 0.00788 | -0.03467 ± 0.00467 |
| same | +0.04705 ± 0.00790 | +0.04479 ± 0.00417 |
| worsen | -0.09260 ± 0.01292 | -0.10828 ± 0.01366 |

所見:
- `improve` はさらに 0 に近づいた。
- `same` も少し低下した。
- `worsen` はやや強く負に寄った。

解釈:
- 少なくとも
  - `improve < 0`
  - `same > 0`
の逆転は残るが、
- `improve` と `same` の距離は少し縮んでいる。

## 6. 主診断: turn_diag

advantage mean（seed=5 mean ± std）

| bucket | exp_034 | exp_035 |
|---|---:|---:|
| early | -0.68917 ± 0.18660 | -0.49878 ± 0.10793 |
| mid | -0.52030 ± 0.16728 | -0.36281 ± 0.09607 |
| late | +0.13113 ± 0.03829 | +0.09287 ± 0.02201 |

所見:
- ここは **はっきり改善**した。
- early / mid の過剰な負 advantage が緩和
- late の過剰な正 advantage も緩和

解釈:
- `exp_034` で増幅されていた turn 依存歪みは、やはり更新強度の問題だった可能性が高い。

## 7. 解釈

今回の結果はかなり前向きに評価できます。

1. **ユーザーの仮説は当たっていた**  
   高表現力モデルに対して、小モデル時代の更新条件は強すぎた。  
   `batch_size↑ / epochs↓` により、その不整合がかなり改善した。

2. **構造改善の芽が実際に活きた**  
   `exp_034` では「構造は良さそうだが update が強すぎる」状態だった。  
   `exp_035` では更新を弱めたことで、
   - 更新安定性
   - turn 歪み
   - after 指標
   が同時に改善した。

3. **なお完全解決ではない**  
   `same > 0 / improve < 0` はまだ残る。  
   したがって本丸はまだ全部解けていない。  
   ただし、少なくとも「構造改善 + 更新適正化」が前進方向であることはかなり強くなった。

4. **`eval_before -> eval` の解釈には注意が必要**  
   そこだけ見ると `exp_034` のほうがきれいだった。  
   しかし最終性能は `exp_035` のほうが良い。  
   つまり今後は
   - after 指標
   - 更新安定性
   - 診断指標
   を合わせて見る必要がある。

## 8. 結論

- `batch_size=512, epochs=2` は **有効**。
- 少なくとも `exp_034` の「更新が強すぎる」問題には効いている。
- 現時点では、
  - 高表現力モデル
  - `exclude_post_riichi_discards=true`
  - `batch_size=512, epochs=2`
は、かなり有望な組み合わせ。

したがって `exp_035` は
- **`exp_034` より前進**
- 今後の比較基準候補
として扱ってよい。

## 9. 次アクション

1. 次に自然なのは、この構成を基準にして
   - `lr=5e-5`
   をさらに試すこと。

2. ただし、まずは `exp_035` を新しい高表現力側の参照点として扱い、
   - `exp_033`
   - `exp_035`
   の 2 本で小モデル vs 大モデルの新基準を持つのがよい。

3. その上で、残っている
   - `same > 0 / improve < 0`
   の構造
を、target/value 側から詰めるのが自然。
