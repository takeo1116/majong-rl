# Experiment Report: exp_025

作成日: 2026-03-10  
対象: `experiments/exp_025/runbook.md`  
目的: `exp_024` の採用候補 B 条件を単条件で再実行し、`shanten_diag` / `turn_diag` を用いて value/target の残差を診断する

## 1. 実験概要

対象条件は `exp_024` の採用候補 B をそのまま固定した。

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
  - `model.value_features.current_shanten.enabled=false`

共通条件:

- seeds: `42,43,44,45,46`
- phases: `imitation,selfplay,learner,eval`
- evaluation: `rotation`, `num_matches=30`
- selfplay: `num_matches=200`

## 2. 実行結果

- batch: （ローカル run）
- `success_count = 5/5`
- `summary.json.success=true`
- `shanten_diag` / `turn_diag` とも全 run で確認

## 3. 通常評価

mean ± std（seed=5）

| 指標 | 値 |
|---|---:|
| avg_rank | 3.3833 ± 0.0880 |
| avg_score | -13269.2 ± 1585.1 |
| win_rate | 0.04683 ± 0.01276 |
| deal_in_rate | 0.58175 ± 0.01122 |

補足:

- この run は診断用であり、通常評価自体の採否は `exp_024` の結論を維持する
- `clip_fraction = 0.5974 ± 0.0349`
- `value_error_mean = 225.24 ± 14.65`

## 4. 主診断: shanten_diag

mean ± std（seed=5）

| 群 | count | adv mean | return mean | old_value mean | new_value mean | value_update_delta mean | value_error mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| improve | 23721.2 ± 199.2 | -0.0877 ± 0.0047 | -326.7 ± 22.9 | -26.55 ± 4.21 | -246.60 ± 17.52 | -220.06 ± 14.16 | +300.15 ± 19.50 |
| same | 78261.6 ± 735.1 | +0.0103 ± 0.0012 | -244.1 ± 15.4 | -27.73 ± 4.44 | -249.73 ± 17.46 | -222.00 ± 13.88 | +216.39 ± 11.75 |
| worsen | 18805.2 ± 214.7 | +0.0608 ± 0.0039 | -198.8 ± 14.7 | -25.33 ± 4.03 | -228.57 ± 14.05 | -203.25 ± 10.81 | +173.43 ± 11.55 |

所見:

1. **逆向き傾向は継続している。**
   - `improve` 群の `advantage.mean` は依然として負
   - `worsen` 群の `advantage.mean` は依然として正

2. **return 自体にも群差がある。**
   - `improve.return.mean` は最も悪く
   - `worsen.return.mean` はそれよりかなり高い
   ので、単に value baseline だけでなく、系列 return 自体が局所ラベルとずれている可能性がある

3. **old_value は全群で return よりかなり浅い負値に留まっている。**
   - `improve`: `return -326.7` に対して `old_value -26.5`
   - `worsen`: `return -198.8` に対して `old_value -25.3`
   で、value baseline は全体に過大評価寄り

4. **PPO 更新後の `new_value` は全群で大きく負方向へ動く。**
   - `value_update_delta` は全群で約 `-200` 前後
   - 特に `improve` 群で最も強く負側へ更新されている

5. **value_error は `improve > same > worsen` の順で大きい。**
   - improve 群で value misfit が最も大きい
   - これは `improve` advantage が負に寄る現象と整合する

## 5. 主診断: turn_diag

mean ± std（seed=5）

| バケット | count | adv mean | return mean | old_value mean | new_value mean | value_update_delta mean | value_error mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| early (0-5) | 10644.0 ± 47.7 | +0.1436 ± 0.0080 | -120.17 ± 2.32 | -17.86 ± 2.87 | -159.76 ± 6.56 | -141.90 ± 5.54 | +102.30 ± 1.28 |
| mid (6-11) | 10642.8 ± 48.8 | +0.1006 ± 0.0090 | -158.57 ± 2.83 | -19.67 ± 3.16 | -192.00 ± 11.06 | -172.33 ± 8.58 | +138.90 ± 2.06 |
| late (12+) | 100301.2 ± 496.1 | -0.0259 ± 0.0018 | -276.28 ± 19.70 | -28.82 ± 4.60 | -259.92 ± 18.91 | -231.10 ± 15.19 | +247.45 ± 15.94 |

所見:

1. **advantage は序盤・中盤で正、終盤で負に反転する。**
   - `early` と `mid` は正寄り
   - `late` は明確に負寄り

2. **value_error は終盤で急増する。**
   - `early`: 約 `+102`
   - `mid`: 約 `+139`
   - `late`: 約 `+247`
   となっており、終盤で value misfit が最も強い

3. **new_value の更新量も終盤で最も大きい。**
   - `value_update_delta` は `late` で最も大きく負
   - PPO が終盤局面を強く下方修正している

## 6. imitation 指標

mean ± std（seed=5）

| 指標 | 値 |
|---|---:|
| teacher_top1_match_rate | 0.1797 ± 0.0086 |
| teacher_best_set_hit_rate | 0.5876 ± 0.0067 |
| imitation value_loss | 9.04e6 ± 4.78e5 |

## 7. 解釈

今回の結果から、少なくとも次は言える。

1. **`improve/worsen` の逆転は、単なるランダムノイズではない。**  
   群平均で一貫して逆向きであり、しかも `old_value/new_value/value_error` にも系統差がある。

2. **問題は value baseline の系統誤差と整合している。**  
   `old_value` は全群で return よりかなり浅く、`improve` 群で特に misfit が大きい。

3. **終盤ほど value misfit が強い。**  
   `turn_diag` で `late` の `value_error` と `advantage` が最も悪い。  
   これは「turn_number は入力にあるが、value が十分使えていない」可能性、または「終盤 target 自体が難しい」可能性を示す。

4. **次の本命は reward ではなく value/target 設計である。**  
   reward shaping と joint imitation で通常評価は前進したが、診断上の残差は依然として value 側に集中している。

## 8. 結論

1. `exp_024` の採用候補 B は維持する。  
2. `exp_025` により、残差の本命は **value/target 側** であることがさらに強まった。  
3. 次段では
   - `improve/worsen` 群での `return vs old_value`
   - `new_value - old_value`
   - `turn_diag` の終盤 misfit
   を前提に、value/target 改善の比較実験を組む。

## 9. 次アクション

1. B 条件を基準に、value/target 改善仮説の runbook を作る。  
2. とくに
   - 終盤 misfit を抑える方向
   - value baseline の系統誤差を減らす方向
   を優先して比較する。  
3. reward shaping の追加探索や `current_shanten` 追加比較は当面優先しない。
