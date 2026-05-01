# Experiment Report: exp_022

作成日: 2026-05-01  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_022/runbook.md`
- `experiments/Stage02_CallUnlock/exp_022/run_map.json`
- `experiments/Stage02_CallUnlock/exp_021/report.md`
- `experiments/Stage02_CallUnlock/exp_020/report.md`
- `docs/CHANGE_QUEUE.md`

## 1. 要約

`exp_022` は、`exp_021` で最有力になった `B: no-anchor, lr=1e-4, clip=0.15` を 3 seed × 60 cycle で fresh に回し、30 cycle 時点の best が天井なのか、長期化で安定するのかを確認した実験である。

結論は次の通り。

- 3 seed すべて正常完了した
- `cycle00-29` は `exp_021 B` と同じ曲線を再現した
- `cycle20-29` までは 3 seed 平均で imitation より改善している
- `cycle30-39` も seed42/44 では良いが、seed43 は悪化が始まる
- `cycle40` 以降、3 seed すべてで明確に崩壊した
- `cycle50-59` は平均 avg_rank `2.985` で、imitation より大幅に悪い
- best は `2.147` と強いが、tail10 は `2.985`、final は `3.132` まで悪化した
- collapse は entropy 低下と同期している
- `ratio_mean` は外れ値で壊れており、今後は `CQ-0281` の log_ratio quantile / max_prob quantile が必要

したがって、`B no-anchor` は短期的には有望だが、現設定のまま 60 cycle まで回しても安定しない。

現時点の解釈は以下。

```text
no-anchor は改善を出す力がある。
ただし方策更新を制御する摩擦が不足しており、c40 以降で entropy collapse とともに崩壊する。
```

次は `entropy_coef` を小さく導入し、collapse を遅らせるだけでなく、良い policy 帯を維持できるかを見る。

## 2. 背景

`exp_021` では、現行 fixedRL 条件で no-anchor を比較した。

3 seed 平均では `B: no-anchor, lr=1e-4, clip=0.15` が最も良かった。

| Group | imitation | final | best | best5 | tail5 | tail10 |
|---|---:|---:|---:|---:|---:|---:|
| A anchor075 lr1e-4 clip0.15 | 2.352 | 2.360 | 2.188 | 2.311 | 2.392 | 2.355 |
| B no-anchor lr1e-4 clip0.15 | 2.352 | 2.327 | 2.148 | 2.246 | 2.288 | 2.303 |
| C no-anchor lr1e-4 clip0.10 | 2.352 | 2.355 | 2.165 | 2.257 | 2.313 | 2.303 |
| D no-anchor lr5e-5 clip0.15 | 2.330 | 2.363 | 2.190 | 2.253 | 2.361 | 2.338 |

この結果だけでは、次の問いが残っていた。

- `2.10〜2.15` 付近の best が現設定の天井なのか
- 30 cycle 以降も改善が続くのか
- 長く回せば tail も安定するのか
- 逆に entropy collapse / over-update で崩れるのか

`exp_022` は、条件を増やさず B を 60 cycle まで伸ばすことで、この問いに答えるための実験である。

## 3. 実験条件

全 seed 共通:

- config: `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- `training.multi_cycle.num_cycles = 60`
- `training.multi_cycle.selfplay_matches_per_cycle = 200`
- `training.policy_anchor.enabled = false`
- `training.policy_anchor.coef = 0.0`
- `training.lr = 0.0001`
- `training.clip_epsilon = 0.15`
- `training.value_loss_coef = 0.125`
- `model.value_hidden_dims = [128, 64]`
- `model.semantic_aux.enabled = true`
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.enabled = true`
- `training.semantic_aux.terminal_loss_coef = 0.1`
- `training.semantic_aux.yaku_loss_coef = 0.05`
- `feature_encoder.tile_presence_flags = false`
- `model.semantic_aux.tile_presence_flags_semantic_only = false`
- `selfplay.temperature = 1.0`
- expected shard semantics: `sample_semantics_version = 3`

run:

| seed | run_dir |
|---:|---|
| 42 | `runs/20260501_stage2a_exp022_Blong_noanchor_lr1e4_clip015_seed42_9d2e15a6` |
| 43 | `runs/20260501_stage2a_exp022_Blong_noanchor_lr1e4_clip015_seed43_ce37fabb` |
| 44 | `runs/20260501_stage2a_exp022_Blong_noanchor_lr1e4_clip015_seed44_34ec042d` |

## 4. 主結果

avg_rank は低いほど良い。

### 4.1 seed 別結果

| seed | imitation | final | best | best5 | best10 | tail5 | tail10 | tail20 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.365 | 2.910 | c37 2.185 | c34-c38 2.269 | c29-c38 2.307 | 2.875 | 2.785 | 2.650 |
| 43 | 2.450 | 3.285 | c09 2.145 | c06-c10 2.219 | c06-c15 2.273 | 3.177 | 3.098 | 2.947 |
| 44 | 2.240 | 3.200 | c21 2.110 | c21-c25 2.211 | c20-c29 2.233 | 3.177 | 3.072 | 2.851 |

読み:

- best は 3 seed すべてで imitation より明確に良い
- best5 / best10 も強い
- しかし tail5 / tail10 / final は大幅に悪化
- seed42 は c37 まで良いが、その後崩れる
- seed43 は c30 以降の悪化が早い
- seed44 は c30 台までは良いが、c40 以降崩れる

### 4.2 3 seed 平均

| 指標 | avg_rank | imitation差分 |
|---|---:|---:|
| imitation | 2.352 | 0.000 |
| final | 3.132 | +0.780 |
| best | 2.147 | -0.205 |
| best5 | 2.233 | -0.119 |
| best10 | 2.271 | -0.081 |
| tail5 | 3.076 | +0.725 |
| tail10 | 2.985 | +0.633 |
| tail20 | 2.816 | +0.465 |

読み:

- best / best5 / best10 では改善がある
- tail では完全に崩れている
- 60 cycle final は採用不能
- 現設定では「長く回せば安定する」ではなく「長く回すと壊れる」

## 5. cycle window

3 seed 平均:

| Window | avg_rank | imitation差分 |
|---|---:|---:|
| c00-c09 | 2.341 | -0.011 |
| c10-c19 | 2.320 | -0.032 |
| c20-c29 | 2.303 | -0.049 |
| c30-c39 | 2.343 | -0.009 |
| c40-c49 | 2.648 | +0.296 |
| c50-c59 | 2.985 | +0.633 |

読み:

- `c20-c29` は `exp_021 B tail10` と同じで、短期改善は再現した
- `c30-c39` は平均では imitation とほぼ同等まで戻るが、まだ崩壊ではない
- `c40-c49` で明確に悪化
- `c50-c59` は collapse と呼んでよい水準

seed 別:

| seed | c00-c09 | c10-c19 | c20-c29 | c30-c39 | c40-c49 | c50-c59 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.353 | 2.359 | 2.340 | 2.309 | 2.516 | 2.785 |
| 43 | 2.304 | 2.315 | 2.338 | 2.462 | 2.797 | 3.098 |
| 44 | 2.366 | 2.287 | 2.233 | 2.258 | 2.632 | 3.072 |

3 seed すべてで `c40-c49` から悪化しているため、偶然の1seed現象ではない。

## 6. win / deal-in

| seed | final win_rate | final deal_in_rate | tail10 win_rate | tail10 deal_in_rate |
|---:|---:|---:|---:|---:|
| 42 | 0.1038 | 0.2030 | 0.1364 | 0.2032 |
| 43 | 0.0639 | 0.2292 | 0.0880 | 0.2090 |
| 44 | 0.0556 | 0.2009 | 0.0698 | 0.1992 |
| avg | 0.0744 | 0.2110 | 0.0981 | 0.2038 |

読み:

- collapse 後は win_rate が大きく落ちる
- deal_in_rate は大きく爆増というより、勝てなくなる影響が目立つ
- 方策が極端に保守化/硬直化した可能性もある

## 7. PPO diagnostics

`exp_022` は `CQ-0281` 実装前の run なので、log_ratio quantile / max_prob quantile はまだない。

既存 diagnostics:

| seed | clip_avg | clip_last | clip_max | entropy_avg | entropy_last | entropy_min | ratio_last | ratio_max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.2587 | 0.3378 | 0.3389 | 0.2596 | 0.0662 | 0.0662 | 0.9589 | 1.22e10 |
| 43 | 0.2330 | 0.3115 | 0.3186 | 0.2863 | 0.1309 | 0.1309 | 1.5730 | 4.11e9 |
| 44 | 0.2604 | 0.3184 | 0.3395 | 0.2097 | 0.0262 | 0.0257 | 13.8164 | 4.94e1 |
| avg | 0.2507 | 0.3226 | 0.3323 | 0.2519 | 0.0744 | 0.0743 | 5.4495 | 5.42e9 |

読み:

- `clip_fraction` は後半で `0.31〜0.34` まで上がる
- entropy は後半で極端に低下する
- seed44 の `entropy_last=0.026` はほぼ決め打ち状態に近い
- `ratio_max` は外れ値で非常に大きくなっており、`ratio_mean` 系は診断として壊れやすい
- 今後は `CQ-0281` の `log_ratio_p95/p99`, `max_prob_p95/p99`, branch別 diagnostics を重視する

## 8. 解釈

### 8.1 no-anchor は改善力を持つ

`exp_021` と同様、`exp_022` でも `c20-c30` 付近までは imitation より良い。

特に best 平均 `2.147`、best5 平均 `2.233` は強い。

したがって no-anchor は棄却すべきではない。むしろ anchor ありより良い policy 領域に入る力はある。

### 8.2 ただし現設定のまま長期化すると崩壊する

60 cycle まで回すと、3 seed すべてで `c40` 以降に崩れた。

これは「30 cycle 時点のbest付近が天井か」を見るというより、現設定が長期安定性を欠いていることを示している。

### 8.3 collapse は entropy 低下と同期している

後半の avg_rank 悪化と同時に entropy が極端に下がった。

これは以下のループを疑わせる。

```text
no-anchor で方策が自由に動く
→ 良い領域に入る
→ 方策が尖る
→ selfplay 分布が狭くなる
→ さらに尖る
→ 誤った argmax / 低多様性に固定される
→ eval が崩壊する
```

ただし entropy 低下が原因か結果かはまだ断定できない。

### 8.4 次は entropy_coef が自然

`entropy_coef` は既に実装済みで、現 config では `0.0` である。

次に試すべきは小さい entropy bonus。

候補:

```text
B_entropy001:
  no-anchor
  lr=1e-4
  clip=0.15
  entropy_coef=0.001
  num_cycles=60

B_entropy003:
  no-anchor
  lr=1e-4
  clip=0.15
  entropy_coef=0.003
  num_cycles=60
```

期待する効果は単なる「崩壊遅延」だけではない。

- 弱い期待: c40以降の崩壊を遅らせる
- 中くらいの期待: c20-c40 の良い帯を c50-c60 まで維持する
- 強い期待: entropy collapse の自己強化ループを切り、最後まで良い policy を維持する

## 9. CQ / 実装状況

`CQ-0281` を追加し、その後実装済み。

目的:

- `ratio_mean` が外れ値で壊れる問題に対応
- `log_ratio` quantile を追加
- advantage sign / quantile を追加
- advantage × log_ratio cross stats を追加
- max_prob quantile を追加
- discard / call branch別 diagnostics を追加

`exp_022` は `CQ-0281` 実装前の run なので、これらの新 diagnostics は含まれていない。次の entropy_coef 実験から利用できる。

`CQ-0280` は未実装。

目的:

- eval worker crash 時に同じ checkpoint で eval だけ retry

`exp_022` では eval worker crash は再発しなかったため、entropy_coef 実験の前に必須ではない。

## 10. entropy_coef=0.003 seed42 probe

`exp_022` 後に、同じ `B no-anchor lr1e-4 clip0.15` 条件へ `training.entropy_coef=0.003` だけを追加した 1 seed probe を実行した。

run:

```text
runs/20260501_stage2a_entropy_probe_noanchor_ec003_seed42_2c8100e5
```

比較対象は `exp_022` seed42。

| condition | final | best | best5 | best10 | tail5 | tail10 | tail20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp022 seed42 entropy=0.0 | 2.910 | 2.185 | 2.269 | 2.307 | 2.875 | 2.785 | 2.650 |
| probe seed42 entropy=0.003 | 3.255 | 2.185 | 2.315 | 2.353 | 3.130 | 3.005 | 2.833 |

window 比較:

| condition | c00-c09 | c10-c19 | c20-c29 | c30-c39 | c40-c49 | c50-c59 |
|---|---:|---:|---:|---:|---:|---:|
| exp022 seed42 entropy=0.0 | 2.353 | 2.359 | 2.339 | 2.309 | 2.516 | 2.784 |
| probe seed42 entropy=0.003 | 2.411 | 2.369 | 2.429 | 2.470 | 2.662 | 3.005 |

diagnostics 比較:

| condition | entropy_avg | entropy_last | clip_avg | clip_last |
|---|---:|---:|---:|---:|
| exp022 seed42 entropy=0.0 | 0.2596 | 0.0662 | 0.2587 | 0.3378 |
| probe seed42 entropy=0.003 | 0.2631 | 0.0651 | 0.2621 | 0.3426 |

読み:

- `entropy_coef=0.003` は seed42 では collapse を止めなかった
- entropy は最終的に `0.065` まで落ちており、`entropy=0.0` とほぼ同じ
- `clip_fraction` も下がらず、むしろわずかに悪い
- best 一点は同じ `2.185` だが、best5 / best10 / tail はすべて悪化した
- この probe だけを見る限り、`entropy_coef=0.003` を 3 seed 実験へ昇格する根拠は弱い

`CQ-0281` により、この probe には新 diagnostics が入っている。

cycle59 の主な値:

| metric | value |
|---|---:|
| `max_prob_mean` | 0.9737 |
| `max_prob_p95` | 1.0000 |
| `log_ratio_p01` | -35.2108 |
| `log_ratio_p99` | 0.8532 |
| `ratio_p99` | 2.3472 |
| `ratio_max` | 7.89e6 |

この値から見ると、終盤は action distribution がかなり尖っている。`entropy_coef=0.003` は、少なくともこの条件では `max_prob` の飽和を防げていない。

次に entropy 系を続けるなら、単純な `0.003` 3seed よりも以下のどちらかを優先したい。

1. より強い `entropy_coef=0.01`
2. entropy ではなく update 制御側、例えば `target_kl` / early stop / lr decay / max_grad_norm 強化

## 11. 結論

`exp_022` の結論:

1. `B no-anchor lr1e-4 clip0.15` は短期的に強い
2. `cycle20-30` 付近までは imitation より改善する
3. しかし現設定のまま 60 cycle 回すと `cycle40` 以降に3seedで崩壊する
4. collapse は entropy 低下、clip_fraction 上昇、ratio outlier と同期している
5. 30cycle時点の best は「採用可能なピーク」だが、「最後まで維持できる安定解」ではない
6. 次は entropy_coef を小さく入れ、良い policy 帯を維持できるかを見る

次実験の推奨:

```text
exp_023 candidate:
  B_entropy010_60cycle seed42 probe
  or target_kl / early-stop diagnostics experiment
```

`entropy_coef=0.003` は seed42 probe で不十分だったため、同条件の 3seed 化は現時点では推奨しない。
