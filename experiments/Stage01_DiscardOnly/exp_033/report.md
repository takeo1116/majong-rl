# Experiment Report: exp_033

作成日: 2026-03-12  
対象: `experiments/exp_033/runbook.md`  
目的: `training.exclude_post_riichi_discards.enabled=true` により、`same` 群の advantage 正偏りと PPO 後悪化が緩和するかを確認する

## 1. 実験概要

新規実行 1 条件:
- A: post-riichi 除外あり
  - `training.exclude_post_riichi_discards.enabled=true`

共通固定（主要）:
- `feature_encoder.shanten_hint.enabled=true`
- `model.hidden_dims=[256,128]`
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
- `exp_031` post-fix baseline

`success_count = 5/5`。

## 2. 通常評価

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| exp_031 baseline | 3.4150 ± 0.0730 | -13380.5 ± 1063.5 | 0.05019 ± 0.00513 | 0.57715 ± 0.01388 |
| exp_033 exclude post-riichi | 3.4117 ± 0.0582 | -13467.5 ± 903.0 | 0.04881 ± 0.01077 | 0.58610 ± 0.00566 |

`eval_before -> eval` の delta:

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| exp_031 baseline | +0.0783 ± 0.0415 | -1144.5 ± 518.4 | -0.00855 ± 0.00587 | +0.00676 ± 0.01066 |
| exp_033 exclude post-riichi | +0.0367 ± 0.0509 | -426.0 ± 237.3 | -0.00353 ± 0.00554 | +0.00976 ± 0.01175 |

所見:
- `eval_before -> eval` はかなり改善した。
  - `Δavg_rank` は半分以下
  - `Δavg_score` は大幅改善
  - `Δwin_rate` も改善
- ただし after 指標は全面改善ではない。
  - `avg_rank` はわずかに改善
  - `avg_score` はやや悪化
  - `win_rate` は悪化
  - `deal_in_rate` は悪化
- つまり「PPO で壊れにくくはなったが、最終性能が明確に上がったとはまだ言えない」という結果。

## 3. 立直後打牌除外の実績

`post_riichi_exclusion.excluded_post_riichi_discards`:
- mean ± std = `2619.6 ± 178.0`

補足:
- `shanten_diag.total_post_riichi_discards = 0`
- `shanten_diag.available_post_riichi_discards = 0`
- `shanten_diag.same.post_riichi_discard_ratio = 0.0`

これは矛盾ではなく、**learner 診断が除外後サンプルに対して計算されている**ため。  
今回の結果から言えるのは、
- 立直後打牌は実際に各 seed で約 2600 サンプル除外されている
- その除外後データで PPO 診断が計算されている
ということ。

## 4. 主診断: 更新安定性

mean ± std（seed=5）

| 条件 | clip_fraction | ratio_std | value_error_mean |
|---|---:|---:|---:|
| exp_031 baseline | 0.09013 ± 0.01169 | 0.12834 ± 0.00737 | -0.03365 ± 0.00387 |
| exp_033 exclude post-riichi | 0.08546 ± 0.01173 | 0.12482 ± 0.00750 | -0.03225 ± 0.00496 |

所見:
- 更新安定性はわずかに改善。
- `clip_fraction` と `ratio_std` は低下。
- `value_error_mean` もやや 0 に近づいた。
- この方向性は、PPO 後悪化幅が縮んだ事実と整合する。

## 5. 主診断: shanten_diag

advantage mean（seed=5 mean ± std）

| 群 | exp_031 | exp_033 |
|---|---:|---:|
| improve | -0.05316 ± 0.00872 | -0.05338 ± 0.01201 |
| same | +0.04811 ± 0.00826 | +0.04944 ± 0.01117 |
| worsen | -0.09022 ± 0.01447 | -0.10012 ± 0.02186 |

所見:
- 期待していた `same` 群 advantage の低下は起きていない。
- `improve` も改善していない。
- むしろ `worsen` はさらに負に寄っている。

解釈:
- 立直後打牌除外は PPO の壊れ方を弱める効果はあるが、
- `same > 0, improve < 0` という **群構造の逆転そのものは解消していない**。

## 6. 主診断: turn_diag

advantage mean（seed=5 mean ± std）

| bucket | exp_031 | exp_033 |
|---|---:|---:|
| early | -0.57114 ± 0.12826 | -0.51684 ± 0.15508 |
| mid | -0.43356 ± 0.11639 | -0.39553 ± 0.14204 |
| late | +0.10595 ± 0.02571 | +0.09880 ± 0.03216 |

所見:
- early / mid / late の歪みは全体に少し緩んだ。
- 特に
  - early の強い負 advantage
  - mid の負 advantage
  - late の正 advantage
 いずれも絶対値は縮小している。
- つまり除外は turn 依存の偏りには一定の改善を与えている。

## 7. 解釈

今回の結果から言えることは 2 つある。

1. 立直後打牌除外は **無意味ではない**  
   PPO 更新の安定性と `eval_before -> eval` の悪化幅は改善した。  
   これは「立直後打牌は learner にノイズを入れていた」という仮説をある程度支持する。

2. ただし **主因の全てではない**  
   `same` 群の正 advantage は消えず、`improve` 群の負 advantage も残った。  
   したがって、
   - 立直後打牌混入は一因ではある
   - しかし advantage 逆転の本丸はそれだけではない
という整理になる。

より厳密に言うと、
- 立直後打牌除外は「PPO が壊れやすい一部のサンプル」を取り除いた
- その結果、更新量と turn 依存歪みは少し改善した
- しかし `same / improve / worsen` の符号構造を作っている主因は別に残っている

## 8. 結論

- `exclude_post_riichi_discards` は **補助的な改善としては有望**。
- ただし、これだけで baseline を明確に更新したとは言えない。
- したがって現時点では
  - 「有効な補助策候補」
  - 「主問題の決定打ではない」
という扱いが妥当。

## 9. 次アクション

1. 今回の結果は、立直後打牌除外を完全却下するものではない。  
   以後の診断実験では `on` を使う価値がある。

2. 一方で、次の本丸は依然として
   - `same / improve / worsen` の target 構造
   - turn 依存での return / value / delta のズレ
   の分解である。

3. 次の候補は次のいずれか。
   - `exclude_post_riichi_discards=true` を前提に、さらに target/value 側診断を進める
   - あるいは baseline と exclusion の 2 条件を今後の診断実験で並行比較し、再現性を確かめる
