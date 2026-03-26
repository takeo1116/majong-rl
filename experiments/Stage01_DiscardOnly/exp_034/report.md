# Experiment Report: exp_034

作成日: 2026-03-12  
対象: `experiments/exp_034/runbook.md`  
目的: `exclude_post_riichi_discards=true` を維持したまま、`hidden_dims=[512,256] + dual_towers` によって PPO 後悪化と診断指標が改善するかを確認する

## 1. 実験概要

新規実行 1 条件:
- A: high-capacity dual towers with exclusion
  - `model.hidden_dims=[512,256]`
  - `model.policy_tower.enabled=true`
  - `model.policy_tower.hidden_dim=128`
  - `model.value_tower.enabled=true`
  - `model.value_tower.hidden_dim=128`
  - `training.exclude_post_riichi_discards.enabled=true`

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
- `training.epochs=4`
- `training.lr=1e-4`
- seeds: `42,43,44,45,46`

batch:
- A: （ローカル run）

比較基準:
- `exp_033` exclude post-riichi

`success_count = 5/5`。

## 2. 通常評価

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| exp_033 exclude post-riichi | 3.4117 ± 0.0582 | -13467.5 ± 903.0 | 0.04881 ± 0.01077 | 0.58610 ± 0.00566 |
| exp_034 high-cap dual towers | 3.4083 ± 0.0821 | -13565.8 ± 1067.0 | 0.04539 ± 0.00736 | 0.58747 ± 0.01115 |

`eval_before -> eval` の delta:

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| exp_033 exclude post-riichi | +0.0367 ± 0.0509 | -426.0 ± 237.3 | -0.00353 ± 0.00554 | +0.00976 ± 0.01175 |
| exp_034 high-cap dual towers | -0.0117 ± 0.0366 | -105.3 ± 1016.0 | -0.00481 ± 0.00475 | +0.00569 ± 0.00476 |

所見:
- PPO 後悪化はさらに縮んだ。
  - `Δavg_rank` は平均で負になり、むしろわずかに改善
  - `Δavg_score` もかなり 0 に近づいた
  - `Δdeal_in_rate` も改善
- ただし after 指標は大きくは伸びない。
  - `avg_rank` はわずかに改善
  - `avg_score` は悪化
  - `win_rate` は悪化
  - `deal_in_rate` もわずかに悪化

つまり、**学習後の壊れにくさは改善したが、最終性能の総合改善にはまだ届いていない**。

## 3. 立直後打牌除外の実績

`post_riichi_exclusion.excluded_post_riichi_discards`:
- exp_033: `2619.6 ± 178.0`
- exp_034: `2816.4 ± 208.9`

所見:
- exclusion は今回も十分な件数で効いている。
- exp_033 と同様、診断は除外後サンプルに対して計算されている。

## 4. 主診断: 更新安定性

mean ± std（seed=5）

| 条件 | clip_fraction | ratio_std | value_error_mean | old_value_mean | new_value_mean |
|---|---:|---:|---:|---:|---:|
| exp_033 exclude post-riichi | 0.08546 ± 0.01173 | 0.12482 ± 0.00750 | -0.03225 ± 0.00496 | -0.25166 ± 0.02474 | -0.21870 ± 0.02004 |
| exp_034 high-cap dual towers | 0.12041 ± 0.00551 | 0.15326 ± 0.00328 | -0.02805 ± 0.00257 | -0.22896 ± 0.00907 | -0.20022 ± 0.00731 |

所見:
- ここは悪化した。
- `clip_fraction` と `ratio_std` は明確に上昇。
- つまり **PPO 更新量は大きくなっている**。
- 一方で `value_error_mean` は 0 に近づいており、value fit は改善方向。

解釈:
- 表現力強化で value 側の近似は少し良くなった
- しかし policy 更新は強くなりすぎている

## 5. 主診断: shanten_diag

advantage mean（seed=5 mean ± std）

| 群 | exp_033 | exp_034 |
|---|---:|---:|
| improve | -0.05338 ± 0.01201 | -0.04597 ± 0.00788 |
| same | +0.04944 ± 0.01117 | +0.04705 ± 0.00790 |
| worsen | -0.10012 ± 0.02186 | -0.09260 ± 0.01292 |

所見:
- advantage 構造は **少し改善**した。
  - improve は 0 に近づいた
  - same はやや低下
  - worsen もやや 0 に近づいた
- ただし
  - `same > 0`
  - `improve < 0`
は依然として残っている。

解釈:
- 表現力強化は群構造の歪みを弱めた
- しかし逆転そのものを解消するには至っていない

## 6. 主診断: turn_diag

advantage mean（seed=5 mean ± std）

| bucket | exp_033 | exp_034 |
|---|---:|---:|
| early | -0.51684 ± 0.15508 | -0.68917 ± 0.18660 |
| mid | -0.39553 ± 0.14204 | -0.52030 ± 0.16728 |
| late | +0.09880 ± 0.03216 | +0.13113 ± 0.03829 |

所見:
- ここは悪化した。
- early / mid の負 advantage はより強くなり
- late の正 advantage もより強くなった

解釈:
- 表現力を上げると、群平均では少し自然化する一方で
- turn 依存の偏りはむしろ強まっている
- したがって **「価値差を学べるようになった」だけではなく、「更新が特定 turn 帯で強くなりすぎた」** 可能性がある

## 7. 解釈

今回の結果は、かなり示唆的です。

1. **表現力不足は全くの見当違いではなかった**  
   `shanten_diag` の群構造は少し改善した。  
   これは、より大きい trunk + dual towers が action/value の区別能力に一定の寄与を持つことを示している。

2. **ただし、表現力だけでは問題は解けない**  
   `same > 0 / improve < 0` は残った。  
   さらに `turn_diag` は悪化した。  
   つまり本丸は依然として target / update dynamics 側に残っている。

3. **PPO 後悪化はかなり抑えられた**  
   `eval_before -> eval` の平均差分はかなり改善した。  
   これは実用的には悪くない。

4. **最終性能に跳ねていない理由**  
   value fit 改善と群構造改善の一部は見えるが、更新量が大きくなりすぎて、turn 依存の歪みを増幅している可能性が高い。

要するに:
- `exp_034` は `exp_033` より「壊れにくい学習」には近い
- しかし「勝てる学習」にはまだ届いていない

## 8. 結論

- `hidden_dims=[512,256] + dual_towers + exclusion` は **一定の前進**。
- 具体的には
  - PPO 後悪化をかなり抑える
  - `shanten_diag` の群構造を少し改善する
- ただし
  - after 指標はまだ明確に上がらない
  - `turn_diag` は悪化
  - `clip_fraction` / `ratio_std` も悪化

したがって現時点では、
- **有望な方向ではある**
- しかし **そのまま採用確定ではない**
という評価が妥当。

## 9. 次アクション

1. 次に自然なのは、`exp_034` の構造を維持したまま **PPO 更新を少し弱める** こと。
   - `training.lr=5e-5`
   - `training.epochs=2`
   のような、過去に使った弱更新条件が第一候補。

2. もしそこでも after 指標が伸びないなら、
   - 表現力不足だけではない
   - target / turn 依存歪みのほうが主因
という判断が強くなる。

3. 今後の比較基準としては、
   - 小モデルの `exp_033`
   - 高表現力の `exp_034`
の 2 点を持っておく価値がある。
