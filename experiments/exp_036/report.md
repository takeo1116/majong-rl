# Experiment Report: exp_036

作成日: 2026-03-12  
対象: [experiments/exp_036/runbook.md](/home/takeo1116/Git/majong-rl/experiments/exp_036/runbook.md)  
目的: 高表現力モデル + 立直後打牌除外条件で、`batch_size=1024` と `lr` の組み合わせ最適化を確認する

## 1. 実験概要

新規実行 2 条件:
- A: `batch_size=1024`, `epochs=2`, `lr=1e-4`
- B: `batch_size=1024`, `epochs=2`, `lr=5e-5`

共通固定（主要）:
- `model.hidden_dims=[512,256]`
- `model.policy_tower.enabled=true`
- `model.policy_tower.hidden_dim=128`
- `model.value_tower.enabled=true`
- `model.value_tower.hidden_dim=128`
- `training.exclude_post_riichi_discards.enabled=true`
- `feature_encoder.shanten_hint.enabled=true`
- `model.value_features.current_shanten.enabled=true`
- `reward.point_delta_scale=0.0001`
- `reward.shaping.shanten_delta.enabled=true`
- `reward.shaping.shanten_delta.scale=0.01`
- `reward.shaping.shanten_delta.mode=both`
- `reward.shaping.shanten_delta.schedule.type=linear_decay`
- seeds: `42,43,44,45,46`

batch:
- A: `runs/20260312_stage1_full_flat_mlp_imitation_then_ppo_batch_1c05c2f0`
- B: `runs/20260312_stage1_full_flat_mlp_imitation_then_ppo_batch_8dad59b4`

比較基準:
- `exp_035` (`batch_size=512`, `epochs=2`, `lr=1e-4`)

両条件とも `success_count = 5/5`。

## 2. 通常評価

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| exp_035 baseline | 3.4100 ± 0.0838 | -13079.0 ± 1269.9 | 0.05033 ± 0.01109 | 0.57166 ± 0.01270 |
| exp_036 A: bs1024 lr1e-4 | 3.5367 ± 0.0859 | -15590.0 ± 1509.1 | 0.02200 ± 0.00980 | 0.59199 ± 0.01223 |
| exp_036 B: bs1024 lr5e-5 | 3.6450 ± 0.0356 | -16657.7 ± 1011.6 | 0.01155 ± 0.00635 | 0.59042 ± 0.00560 |

所見:
- A/B とも `exp_035` より大きく悪化。
- 特に `win_rate` が大きく崩れた。
- `batch_size=1024` は少なくとも今回の条件では強すぎる縮退を起こしている。

## 3. `eval_before -> eval`

mean ± std（seed=5）

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| exp_035 baseline | +0.0550 ± 0.0938 | -358.0 ± 1084.2 | -0.00153 ± 0.00789 | -0.00105 ± 0.00897 |
| exp_036 A: bs1024 lr1e-4 | -0.0100 ± 0.0670 | -186.2 ± 789.6 | -0.00399 ± 0.00782 | +0.00403 ± 0.00559 |
| exp_036 B: bs1024 lr5e-5 | +0.0467 ± 0.0402 | -606.7 ± 518.1 | -0.00363 ± 0.00914 | -0.00553 ± 0.00862 |

所見:
- A は `eval_before -> eval` の見かけは悪くない。
- しかし after 指標は壊れている。
- つまり今回は「PPO 後悪化幅」よりも、**学習前状態そのもの / 到達点の悪化** のほうが支配的。

## 4. 主診断: 更新安定性

mean ± std（seed=5）

| 条件 | clip_fraction | ratio_std | value_error_mean |
|---|---:|---:|---:|
| exp_035 baseline | 0.08796 ± 0.01068 | 0.12474 ± 0.00620 | -0.02798 ± 0.00433 |
| exp_036 A: bs1024 lr1e-4 | 0.02330 ± 0.00533 | 0.09113 ± 0.00386 | -0.03789 ± 0.00840 |
| exp_036 B: bs1024 lr5e-5 | 0.00116 ± 0.00144 | 0.04981 ± 0.01020 | -0.04427 ± 0.00269 |

所見:
- 更新安定性指標だけ見ると、A/B とも大幅改善。
- 特に B は更新がほぼ止まりかけているレベル。

解釈:
- `batch_size=1024` は PPO update をかなり弱める。
- しかし今回は **弱めすぎ** で、性能改善に必要な更新まで失われている可能性が高い。

## 5. 主診断: shanten_diag

advantage mean（seed=5 mean ± std）

| 群 | exp_035 | exp_036 A | exp_036 B |
|---|---:|---:|---:|
| improve | -0.03467 ± 0.00467 | +0.02768 ± 0.01136 | +0.03068 ± 0.01014 |
| same | +0.04479 ± 0.00417 | +0.03732 ± 0.00477 | +0.03217 ± 0.00456 |
| worsen | -0.10828 ± 0.01366 | -0.13216 ± 0.01484 | -0.11337 ± 0.00883 |

所見:
- A/B とも `improve > 0` を達成。
- `same` も少し下がっている。
- 群構造だけ見れば、むしろ見栄えは良い。

ただし:
- 実性能は大きく悪化している。

解釈:
- ここから分かるのは、**群平均の shanten_diag がきれいでも、全体性能は保証されない** ということ。
- 今回は表面上の advantage 構造改善より、policy/探索分布全体の悪化のほうが強い。

## 6. 主診断: turn_diag

advantage mean（seed=5 mean ± std）

| bucket | exp_035 | exp_036 A | exp_036 B |
|---|---:|---:|---:|
| early | -0.49878 ± 0.10793 | -0.70554 ± 0.14080 | -0.73451 ± 0.10567 |
| mid | -0.36281 ± 0.09607 | -0.57102 ± 0.15117 | -0.63341 ± 0.10412 |
| late | +0.09287 ± 0.02201 | +0.13410 ± 0.03035 | +0.14280 ± 0.02180 |

所見:
- turn 依存歪みは A/B とも明確に悪化。
- 特に B が最悪。

解釈:
- `batch_size=1024` は更新を弱めたが、turn 依存バイアスの補正には失敗した。
- むしろ early/mid/late の偏りを固定化した可能性がある。

## 7. 立直後打牌除外の実績

`post_riichi_exclusion.excluded_post_riichi_discards`:
- exp_035: `2544.2 ± 178.7`
- exp_036 A: `1038.2 ± 100.6`
- exp_036 B: `561.0 ± 47.5`

所見:
- A/B で exclusion 件数が大きく減っている。
- これは学習分布がかなり変わっていることを示唆する。
- ただし今回の主問題は、少なくとも learner 更新条件悪化で説明できるため、まずは `batch_size=1024` の影響を優先して解釈するのが妥当。

## 8. 解釈

今回の結果はかなり明確です。

1. **`batch_size=1024` は大きすぎる可能性が高い**
   - 更新安定性は良く見える
   - しかし性能は大きく崩れる
   - 特に B (`lr=5e-5`) は update が弱すぎる

2. **A/B の比較では A のほうがまだマシ**
   - B はほぼ全指標でさらに悪い
   - よって `batch_size=1024` 条件では `lr=5e-5` は弱すぎる

3. **`exp_035` は妥当なバランス点だった**
   - `batch_size=512, epochs=2, lr=1e-4` は
     - 更新安定性
     - turn 歪み
     - after 指標
     のバランスが最も良かった

## 9. 結論

- `exp_036 A/B` は **どちらも `exp_035` を更新できなかった**。
- 特に `batch_size=1024` は、少なくとも今回の設定では過大。
- したがって、現時点の高表現力 baseline 候補は引き続き **`exp_035`**。

## 10. 次アクション

1. `exp_035` を高表現力側の基準として維持する。
2. 次に試すなら、
   - `exp_035` を固定したまま `lr=5e-5`
   - あるいは `batch_size=768` のような中間条件
   が候補。
3. ただし今の優先順位としては、
   - これ以上 batch tuning を続けるより
   - `target/value` 側の本丸に戻る
   判断もかなり有力。
