# Experiment Report: exp_026

作成日: 2026-03-10  
対象: `experiments/exp_026/runbook.md`  
目的: `exp_025` の採用条件を基準に、モデル表現力拡大 + value current shanten により value/target の残差が改善するかを確認する

## 1. 実験概要

比較参照:
- `experiments/exp_025/report.md`

新規実行条件:
- `feature_encoder.shanten_hint.enabled=true`
- `training.imitation_loss_mode=tie_aware_best_set`
- reward:
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.01`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
- imitation:
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.1`
- model:
  - `model.hidden_dims=[512,256]`
  - `model.value_features.current_shanten.enabled=true`

共通条件:
- seeds: `42,43,44,45,46`
- phases: `imitation,selfplay,learner,eval`
- evaluation: `rotation`, `num_matches=30`
- selfplay: `num_matches=200`

## 2. 実行結果

- batch: `runs/20260310_stage1_full_flat_mlp_imitation_then_ppo_batch_1ea273a1`
- `success_count = 5/5`
- `summary.json.success=true`
- `shanten_diag` / `turn_diag` とも全 run で確認
- `summary.model_features.value_features.current_shanten.enabled=true` を確認

## 3. 通常評価

mean ± std（seed=5）

| 指標 | exp_025 | exp_026 |
|---|---:|---:|
| avg_rank | 3.3833 ± 0.0880 | 3.4300 ± 0.0509 |
| avg_score | -13269.2 ± 1585.1 | -14080.5 ± 1222.1 |
| win_rate | 0.04683 ± 0.01276 | 0.03957 ± 0.01300 |
| deal_in_rate | 0.58175 ± 0.01122 | 0.58311 ± 0.01715 |

`eval_before -> eval` の delta:

| 指標 | exp_025 | exp_026 |
|---|---:|---:|
| Δavg_rank | -0.0150 ± 0.0388 | +0.0217 ± 0.0548 |
| Δavg_score | -197.5 ± 596.9 | -591.8 ± 682.7 |
| Δwin_rate | -0.0030 ± 0.0071 | -0.0059 ± 0.0122 |
| Δdeal_in_rate | +0.0069 ± 0.0058 | +0.0069 ± 0.0129 |

所見:

1. **通常評価は `exp_025` より悪化した。**
   - `avg_rank` は悪化
   - `avg_score` も悪化
   - `Δavg_rank` は負から正へ戻った
2. `deal_in_rate` は改善しておらず、`win_rate` も下がった。

## 4. imitation 指標

mean ± std（seed=5）

| 指標 | exp_025 | exp_026 |
|---|---:|---:|
| teacher_top1_match_rate | 0.1797 ± 0.0086 | 0.1812 ± 0.0055 |
| teacher_best_set_hit_rate | 0.5876 ± 0.0067 | 0.5837 ± 0.0070 |
| imitation value_loss | 9.04e6 ± 4.78e5 | 8.98e6 ± 4.81e5 |

所見:

1. imitation teacher 再現率は大差ない。
2. `value_loss` も大きくは変わらず、joint imitation そのものの学習難度は同程度。

## 5. 主診断: shanten_diag

mean ± std（seed=5）

| 群 | 指標 | exp_025 | exp_026 |
|---|---|---:|---:|
| improve | advantage mean | -0.0877 ± 0.0052 | -0.0771 ± 0.0035 |
| improve | return mean | -326.7 ± 22.9 | -264.4 ± 19.8 |
| improve | old_value mean | -26.55 ± 4.21 | -69.83 ± 8.28 |
| improve | new_value mean | -246.60 ± 17.52 | -205.51 ± 10.41 |
| improve | value_update_delta mean | -220.06 ± 14.16 | -135.68 ± 15.36 |
| improve | value_error mean | +300.15 ± 19.50 | +194.55 ± 23.36 |
| worsen | advantage mean | +0.0608 ± 0.0039 | +0.0490 ± 0.0069 |
| worsen | return mean | -198.8 ± 14.7 | -173.9 ± 11.6 |
| worsen | old_value mean | -25.33 ± 4.03 | -66.75 ± 8.08 |
| worsen | new_value mean | -228.57 ± 14.05 | -191.39 ± 8.47 |
| worsen | value_update_delta mean | -203.25 ± 10.81 | -124.64 ± 13.51 |
| worsen | value_error mean | +173.43 ± 11.55 | +107.18 ± 13.45 |

所見:

1. **逆向き傾向は残っている。**
   - `improve.advantage.mean` は依然として負
   - `worsen.advantage.mean` は依然として正
2. ただし、**診断値自体は改善している。**
   - `improve` の負値はやや浅くなった
   - `worsen` の正値もやや小さくなった
3. **value misfit はかなり減った。**
   - `improve.value_error.mean`: `+300.15 -> +194.55`
   - `worsen.value_error.mean`: `+173.43 -> +107.18`
4. `old_value` は `return` にかなり近づいたが、なおズレは大きい。
5. `new_value` の更新量は小さくなっており、PPO 後の過剰な下方修正は緩和している。

## 6. 主診断: turn_diag

mean ± std（seed=5）

| バケット | 指標 | exp_025 | exp_026 |
|---|---|---:|---:|
| early | advantage mean | +0.1436 ± 0.0080 | +0.0973 ± 0.0119 |
| early | old_value mean | -17.86 ± 2.87 | -46.27 ± 5.55 |
| early | new_value mean | -159.76 ± 6.56 | -140.11 ± 5.89 |
| early | value_error mean | +102.30 ± 1.28 | +73.19 ± 5.85 |
| mid | advantage mean | +0.1006 ± 0.0090 | +0.0659 ± 0.0130 |
| mid | old_value mean | -19.67 ± 3.16 | -50.98 ± 6.09 |
| mid | new_value mean | -192.00 ± 11.06 | -164.47 ± 6.60 |
| mid | value_error mean | +138.90 ± 2.06 | +95.15 ± 9.12 |
| late | advantage mean | -0.0259 ± 0.0018 | -0.0172 ± 0.0025 |
| late | old_value mean | -28.82 ± 4.60 | -75.25 ± 8.85 |
| late | new_value mean | -259.92 ± 18.91 | -211.33 ± 12.07 |
| late | value_error mean | +247.45 ± 15.94 | +153.23 ± 21.34 |

所見:

1. **late misfit はかなり減った。**
   - `late.value_error.mean`: `+247.45 -> +153.23`
2. `early/mid` でも `value_error` は減っており、改善は全 turn に及んでいる。
3. `late.advantage.mean` は依然として負だが、絶対値は小さくなった。
4. `old_value` / `new_value` は全 bucket でより負に動き、return のスケールに近づいた。

## 7. learner 補助指標

mean ± std（seed=5）

| 指標 | exp_025 | exp_026 |
|---|---:|---:|
| clip_fraction | 0.5974 ± 0.0349 | 0.6502 ± 0.0354 |
| value_error_mean | 225.24 ± 14.65 | 141.22 ± 18.51 |
| ratio_std | 0.7163 ± 0.1417 | 2.6638 ± 2.6950 |

所見:

1. **global value_error_mean は大きく改善した。**
2. 一方で `clip_fraction` はやや悪化し、`ratio_std` は不安定化した。
3. つまり value の fit は改善したが、policy 更新の安定性はむしろ弱くなった可能性がある。

## 8. 解釈

今回の結果は、単純な yes/no ではない。

1. **表現力不足仮説は一定程度支持された。**
   - `shanten_diag` / `turn_diag` / global `value_error` は明確に改善
   - とくに `late` の misfit が大きく下がった
2. **しかし通常評価は改善しなかった。**
   - `avg_rank` / `avg_score` は `exp_025` より悪い
   - `Δavg_rank` も正へ戻った
3. よって、
   - 「value misfit の一部は表現力不足だった」
   - ただし「通常評価悪化の主因がそれだけだった」とは言えない
   という結論になる。
4. さらに、今回の条件は
   - hidden_dims 拡大
   - value current_shanten 有効
   を同時に入れているため、改善要因の分離はまだできていない。

## 9. 結論

1. **大きいモデル + value current_shanten で、value 診断値は改善した。**  
   `shanten_diag` と `turn_diag` の misfit は明確に減った。
2. **しかし通常評価は `exp_025` を更新できず、不採用。**
3. この結果から、  
   - value 表現力不足は残差の一因だった  
   - しかし通常評価まで決める本丸は、まだ別にある  
   と考えるのが妥当。

## 10. 次アクション

1. 次にやるべきは、今回の改善要因を分離する比較である。  
   - hidden_dims 拡大だけ
   - current_shanten だけ
   - あるいは value/target ノブ
   のどれが本当に効いたかを切る必要がある。
2. 特に、`value_error` 改善が通常評価に繋がらない理由を確認するため、  
   `clip_fraction` / `ratio_std` 側とのトレードオフも意識して比較する。
