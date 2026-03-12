# Experiment Report: exp_037

作成日: 2026-03-13  
対象: [experiments/exp_037/runbook.md](/home/takeo1116/Git/majong-rl/experiments/exp_037/runbook.md)  
目的: `exp_035` を高表現力 baseline として、`gae_lambda` 短縮と imitation value warmstart 強化が PPO 安定性と after 指標を改善するかをコード変更なしで確認する

## 1. 実験概要

新規実行 4 条件:
- A: baseline replay
  - `training.gae_lambda=0.95`
  - `training.imitation_value_warmstart.coef=0.1`
- B: shorter GAE
  - `training.gae_lambda=0.90`
  - `training.imitation_value_warmstart.coef=0.1`
- C: stronger imitation value warmstart
  - `training.gae_lambda=0.95`
  - `training.imitation_value_warmstart.coef=0.3`
- D: both
  - `training.gae_lambda=0.90`
  - `training.imitation_value_warmstart.coef=0.3`

共通固定（主要）:
- `feature_encoder.shanten_hint.enabled=true`
- `model.hidden_dims=[512,256]`
- `model.value_features.current_shanten.enabled=true`
- `model.policy_tower.enabled=true`
- `model.policy_tower.hidden_dim=128`
- `model.value_tower.enabled=true`
- `model.value_tower.hidden_dim=128`
- `training.imitation_loss_mode=tie_aware_best_set`
- `reward.point_delta_scale=0.0001`
- `reward.shaping.shanten_delta.enabled=true`
- `reward.shaping.shanten_delta.scale=0.01`
- `reward.shaping.shanten_delta.mode=both`
- `reward.shaping.shanten_delta.schedule.type=linear_decay`
- `training.epochs=2`
- `training.lr=0.0001`
- `training.batch_size=512`
- `training.gamma=0.99`
- `training.exclude_post_riichi_discards.enabled=true`
- seeds: `42,43,44,45,46`

batch:
- A: `runs/20260312_stage1_full_flat_mlp_imitation_then_ppo_batch_39dc4ffb`
- B: `runs/20260312_stage1_full_flat_mlp_imitation_then_ppo_batch_2775597e`
- C: `runs/20260312_stage1_full_flat_mlp_imitation_then_ppo_batch_4b78a75b`
- D: `runs/20260312_stage1_full_flat_mlp_imitation_then_ppo_batch_1d5383df`

比較基準:
- `exp_035`

全条件 `success_count = 5/5`。

## 2. 通常評価

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| exp_035 baseline | 3.4100 ± 0.0838 | -13079.0 ± 1269.9 | 0.05033 ± 0.01109 | 0.57166 ± 0.01270 |
| A baseline replay | 3.4100 ± 0.0838 | -13079.0 ± 1269.9 | 0.05033 ± 0.01109 | 0.57166 ± 0.01270 |
| B shorter GAE | 3.4167 ± 0.0771 | -13504.3 ± 1442.0 | 0.04486 ± 0.00782 | 0.57796 ± 0.01475 |
| C stronger value warmstart | 3.4383 ± 0.0419 | -13793.5 ± 1241.9 | 0.04210 ± 0.00669 | 0.57796 ± 0.01258 |
| D both | 3.3917 ± 0.0601 | -13054.3 ± 1346.5 | 0.04619 ± 0.01231 | 0.57198 ± 0.00476 |

`eval_before -> eval` の delta:

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| exp_035 baseline | +0.0550 ± 0.0839 | -358.0 ± 969.7 | -0.00153 ± 0.00706 | -0.00105 ± 0.00802 |
| B shorter GAE | +0.0483 ± 0.0688 | -655.8 ± 795.9 | -0.00679 ± 0.00286 | +0.00609 ± 0.01146 |
| C stronger value warmstart | +0.0633 ± 0.0588 | -848.7 ± 919.8 | -0.00776 ± 0.00463 | +0.00639 ± 0.00640 |
| D both | +0.0250 ± 0.0803 | -278.5 ± 1014.9 | -0.00539 ± 0.00981 | +0.00000 ± 0.00530 |

所見:
- **D が総合で最も良い**。
  - `avg_rank` は `exp_035` を上回った
  - `avg_score` もわずかに改善
  - `deal_in_rate` はほぼ同等
- B は `turn_diag` や `value_error_mean` は良いが、after 指標が落ちる。
- C は単独では不採用。after 指標と `eval_before -> eval` の両方が悪化。
- D は `win_rate` だけ `exp_035` より少し低いが、総合では A/B/C 中で最もバランスが良い。

## 3. imitation / value warmstart 側の指標

| 条件 | imitation value_loss |
|---|---:|
| A baseline replay | 0.05071 ± 0.00764 |
| B shorter GAE | 0.03040 ± 0.00387 |
| C stronger value warmstart | 0.04728 ± 0.00633 |
| D both | 0.02900 ± 0.00341 |

所見:
- `gae_lambda=0.90` を入れた B/D で imitation `value_loss` が大きく下がっている。
- warmstart coef 単独強化（C）は、ここでは決定的な改善になっていない。

## 4. 立直後打牌除外の実績

`post_riichi_exclusion.excluded_post_riichi_discards`:
- A: `2544.2 ± 159.8`
- B: `2512.8 ± 151.2`
- C: `2509.4 ± 133.5`
- D: `2502.2 ± 161.2`

所見:
- 全条件で exclusion は十分な件数で効いている。
- 今回の比較差は exclusion 件数ではなく、GAE / warmstart 条件差とみてよい。

## 5. 主診断: 更新安定性

mean ± std（seed=5）

| 条件 | clip_fraction | ratio_std | value_error_mean |
|---|---:|---:|---:|
| A baseline replay | 0.08796 ± 0.01068 | 0.12474 ± 0.00620 | -0.02798 ± 0.00433 |
| B shorter GAE | 0.08965 ± 0.01675 | 0.12416 ± 0.00960 | -0.00552 ± 0.00200 |
| C stronger value warmstart | 0.08632 ± 0.00776 | 0.12469 ± 0.00429 | -0.02955 ± 0.00570 |
| D both | 0.09096 ± 0.01418 | 0.12621 ± 0.00669 | -0.00609 ± 0.00183 |

所見:
- `clip_fraction` / `ratio_std` は 4 条件とも大差なし。今回の差は update 強度ではなく、target/value 側の差に見える。
- `value_error_mean` は B/D がかなり 0 に近い。
- ただし、**value_error が良いだけでは after 指標は保証しない**。B は典型例。

## 6. 主診断: shanten_diag

advantage mean（seed=5 mean ± std）

| 群 | A | B | C | D |
|---|---:|---:|---:|---:|
| improve | -0.03467 ± 0.00418 | -0.01758 ± 0.00965 | -0.04619 ± 0.00724 | -0.02769 ± 0.00749 |
| same | +0.04479 ± 0.00373 | +0.04280 ± 0.00467 | +0.05593 ± 0.00390 | +0.04902 ± 0.00416 |
| worsen | -0.10828 ± 0.01221 | -0.12615 ± 0.00713 | -0.12810 ± 0.00805 | -0.13135 ± 0.00786 |

所見:
- B はこの指標だけ見ると最も「自然」に近い。
  - `improve` が 0 に近づく
  - `same` も少し下がる
- しかし B の after 指標は悪い。ここでも **群平均 shanten_diag の改善だけでは採用できない** ことが再確認された。
- C は `same` を強く押し上げており、単独では逆効果。
- D は B ほど綺麗ではないが、A よりはやや改善しつつ after 指標も維持できている。

## 7. 主診断: turn_diag

advantage mean（seed=5 mean ± std）

| bucket | A | B | C | D |
|---|---:|---:|---:|---:|
| early | -0.49879 ± 0.09654 | -0.41828 ± 0.08869 | -0.66915 ± 0.08896 | -0.53674 ± 0.08854 |
| mid | -0.36281 ± 0.08593 | -0.26657 ± 0.08300 | -0.51742 ± 0.08230 | -0.38594 ± 0.08381 |
| late | +0.09287 ± 0.01969 | +0.07385 ± 0.01863 | +0.12799 ± 0.01858 | +0.09948 ± 0.01866 |

所見:
- B は turn 依存歪みの緩和という意味では最良。
- C は明確に悪化。
- D は A と比べて少し悪く見えるが、C ほどは崩れていない。

解釈:
- `gae_lambda=0.90` は turn 依存歪みを減らす効果を持つ。
- ただし、それだけで最終性能が上がるわけではない。
- warmstart coef 強化は単独では turn 側に悪影響。

## 8. 解釈

今回の結果から言えることは次の通り。

1. **`gae_lambda=0.90` 単独は、診断を良くするが性能改善には直結しなかった**  
   B は `value_error_mean`、`shanten_diag`、`turn_diag` がかなり良い。  
   それでも after 指標は `exp_035` を下回る。  
   つまり、現状の critic / target ノイズは減っても、policy の最終着地まで押し上げるには足りない。

2. **warmstart coef=0.3 単独は悪い**  
   C は after、`eval_before -> eval`、`turn_diag` のいずれも悪化。  
   imitation value warmstart を単純に強めるだけでは、むしろ PPO 開始後のバランスを崩している可能性が高い。

3. **組み合わせ D は実用上の最良候補**  
   D は B ほど診断上きれいではないが、A より after 指標が総合でやや良い。  
   とくに `avg_rank` と `avg_score` は改善した。  
   したがって、「GAE 短縮 + ほどほどに強い critic 初期化」の組み合わせは前向きに扱える。

4. **今回の差は update 強度ではなく target/value 側の差**  
   `clip_fraction` と `ratio_std` はほぼ同水準。  
   つまり今回の結果は、まさに runbook の狙い通り `gae_lambda` / warmstart の比較になっている。

## 9. 結論

- 単独条件の採否:
  - B (`gae_lambda=0.90`): **診断改善はあるが after 悪化で不採用**
  - C (`coef=0.3`): **不採用**
- 組み合わせ条件:
  - D (`gae_lambda=0.90` + `coef=0.3`): **条件付き採用候補**

現時点では、
- `exp_035`
- `exp_037 D`
の 2 本が高表現力側の有力候補です。

ただし、`exp_037 D` の改善幅は大きくなく、`win_rate` はまだ `exp_035` を下回る。  
したがって **`exp_035` を完全に置き換えるほどではないが、次の比較基準候補としては十分に価値がある** という位置づけです。

## 10. 次アクション

1. 次に進めるなら第一候補は `exp_037 D` を基準にした追加比較。
2. ただし、`gae_lambda=0.90` の効果は見えたので、次は `0.92` や `0.93` のような中間値も候補になる。
3. warmstart coef については `0.3` 単独が悪かったため、今後振るなら
   - `0.2`
   - あるいは `D` を基準にした微調整
   を優先したい。
4. `clip_fraction` / `ratio_std` はほぼ安定なので、次は update 強度ではなく target/value 側の微調整を続けるのが自然。
