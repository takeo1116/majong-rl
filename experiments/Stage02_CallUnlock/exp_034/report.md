# Experiment Report: exp_034

作成日: 2026-05-10  
Stage: `Stage02_CallUnlock`

## Summary

`exp_034` では、ルール拡張へ進む前の Stage2a final baseline 候補として、`policy_lr=1x` / `value_semantic_lr=100x` / `target_kl=on` を 3seed で確認した。

結論:

- 3seed すべてで強く、seed 間のばらつきも小さい。
- `policy_lr=5x` 系で見られた大きな振動や seed 依存はかなり抑えられた。
- `final mean=2.078`, `best mean=1.998`, `tail20 mean=2.122` で、現時点の Stage2a 基準として最も扱いやすい。
- semantic eval では yaku head が大きく改善しており、`Riichi / Yakuhai / Tanyao / Pinfu` は明確に学習できている。
- terminal head は `win_menzen / win_called` はある程度拾うが、`deal_in` top1 recall は依然 0。terminal loss は分類器そのものより shared representation の補助信号として効いている可能性が高い。

Decision:

```text
Stage2a のルール拡張前 baseline は exp_034 条件を採用候補にする。
policy_lr は 1x に戻し、value_semantic_lr=100x と target_kl を維持する。
```

## Background

`exp_032` では `policy_lr=5x` が seed42 で非常に強かったが、3seed では seed44 が弱く、安定設定としては不安が残った。

`exp_033` では `target_kl` と CQ-0288/CQ-0289 後の構成で `policy_lr=5x` を再確認した。seed42 は良かったが、seed43 は序盤に大きく悪化し、最終的にも `best=2.21`, `tail20=2.323` と弱かった。

このため、`exp_034` では policy 側は保守的に `1x` へ戻し、value/semantic 側だけ `100x` を維持した。

重要な反映済み修正:

- CQ-0282: rulebase baseline action を PPO ratio に混ぜない separated PPO
- CQ-0283: `reward.point_delta_scale=0.0001`
- CQ-0285: terminal loss の weighted mean 正規化
- CQ-0286: policy / value_semantic optimizer lr group 分離
- CQ-0287: target_kl early stop
- CQ-0288: dead weight だった `semantic_proj` 削除
- CQ-0289: lr_groups の適用範囲を PPO / imitation で分離

## Conditions

共通条件:

- `policy_ratio = 1.0`
- `ppo_mode = "separated"`
- `policy_anchor.enabled = false`
- `reward.point_delta_scale = 0.0001`
- `feature_encoder.tile_presence_flags = true`
- `model.value_hidden_dims = [256, 128]`
- `training.lr_groups.enabled = true`
- `training.lr_groups.apply_to = ["ppo"]`
- `training.lr_groups.policy = 0.0001`
- `training.lr_groups.value_semantic = 0.01`
- `clip_epsilon = 0.15`
- `entropy_coef = 0.0`
- `value_loss_coef = 0.125`
- `terminal_loss_coef = 0.1`
- `yaku_loss_coef = 0.05`
- `ppo_target_kl.enabled = true`
- `ppo_target_kl.target = 0.03`
- `ppo_target_kl.stop_multiplier = 1.5`
- `ppo_target_kl.skip_minibatch_on_exceed = true`
- `multi_cycle.num_cycles = 60`
- `selfplay_matches_per_cycle = 200`
- `gradient_norms.enabled = true`

Runs:

| label | seed | run |
|---|---:|---|
| `FINAL_P1_TKL_seed42` | 42 | `runs/20260510_stage2a_exp034_final_p1_tkl_seed42_409da297` |
| `FINAL_P1_TKL_seed43` | 43 | `runs/20260510_stage2a_exp034_final_p1_tkl_seed43_441bf6ee` |
| `FINAL_P1_TKL_seed44` | 44 | `runs/20260510_stage2a_exp034_final_p1_tkl_seed44_05375a73` |

## Performance

Lower avg_rank is better.

| seed | final | best | best_cycle | best10 | tail10 | tail20 | final win | final deal-in | final avg_score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1.960 | 1.960 | 59 | 2.032 | 2.098 | 2.098 | 0.1967 | 0.1837 | 29922.0 |
| 43 | 2.180 | 2.005 | 31 | 2.046 | 2.123 | 2.136 | 0.2220 | 0.2081 | 28630.5 |
| 44 | 2.095 | 2.030 | 53 | 2.067 | 2.119 | 2.131 | 0.2168 | 0.1857 | 29966.0 |

Aggregate:

| metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| final | 2.078 | 0.091 | 1.960 | 2.180 |
| best | 1.998 | 0.029 | 1.960 | 2.030 |
| best10 | 2.048 | 0.014 | 2.032 | 2.067 |
| tail10 | 2.113 | 0.011 | 2.098 | 2.123 |
| tail20 | 2.122 | 0.017 | 2.098 | 2.136 |
| final win | 0.2118 | 0.0109 | 0.1967 | 0.2220 |
| final deal-in | 0.1925 | 0.0111 | 0.1837 | 0.2081 |
| final avg_score | 29506.2 | 619.5 | 28630.5 | 29966.0 |

## Learning Curve

10-cycle block mean:

| seed | 00-09 | 10-19 | 20-29 | 30-39 | 40-49 | 50-59 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.436 | 2.318 | 2.214 | 2.183 | 2.098 | 2.098 |
| 43 | 2.408 | 2.285 | 2.200 | 2.142 | 2.150 | 2.125 |
| 44 | 2.415 | 2.345 | 2.266 | 2.215 | 2.144 | 2.120 |

読み:

- 3seed とも序盤から終盤まで素直に改善している。
- `policy_lr=5x` のような大きな序盤悪化はない。
- seed42 は最終 cycle が best で、late drift がほぼない。
- seed43/44 も tail が `2.12-2.14` に収まっており、実験的にはかなり安定している。

## PPO Diagnostics

Final cycle の代表値。

| seed | clip_fraction | max_prob_mean | max_prob_p95 | entropy | ratio_max | log_ratio_p01 | log_ratio_p99 | approx_kl_mean | approx_kl_max | target_kl_stop | skipped/checked |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.0438 | 0.9285 | 1.0000 | 0.1679 | 9.20 | -0.279 | 0.208 | 0.0029 | 0.0276 | 0 | 0/479 |
| 43 | 0.0505 | 0.9241 | 1.0000 | 0.1823 | 7.13 | -0.321 | 0.228 | 0.0034 | 0.0222 | 0 | 0/461 |
| 44 | 0.0543 | 0.9273 | 1.0000 | 0.1797 | 5.27 | -0.341 | 0.255 | 0.0040 | 0.0207 | 0 | 0/457 |

読み:

- `clip_fraction` は `0.04-0.05` 程度で落ち着いている。
- `approx_kl_max` は target threshold `0.045` を下回っており、final cycle では target_kl stop は発動していない。
- `entropy` は低いが、`policy_lr=5x` や `P10x` ほど極端な collapse には見えない。
- `ratio_max` も一桁台で、P10x のような暴発はない。

## Semantic Evaluation

Final checkpoint (`checkpoint_cycle_59.pt`) と final selfplay shard (`cycle_59/selfplay`) で `semantic_head_eval.py` を実行した。

Output:

| seed | summary |
|---:|---|
| 42 | `experiments/Stage02_CallUnlock/exp_034/semantic_eval_seed42_final_cycle59/semantic_eval_seed42_final_cycle59_summary.md` |
| 43 | `experiments/Stage02_CallUnlock/exp_034/semantic_eval_seed43_final_cycle59/semantic_eval_seed43_final_cycle59_summary.md` |
| 44 | `experiments/Stage02_CallUnlock/exp_034/semantic_eval_seed44_final_cycle59/semantic_eval_seed44_final_cycle59_summary.md` |

### Semantic Summary

| seed | terminal_acc | yaku_micro_F1 | yaku_macro_F1 | exact_match | deal_in_AUC |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.6031 | 0.6839 | 0.1996 | 0.3339 | 0.6004 |
| 43 | 0.6033 | 0.6718 | 0.2093 | 0.3519 | 0.5553 |
| 44 | 0.6089 | 0.6859 | 0.2225 | 0.3465 | 0.5911 |

Aggregate:

| metric | mean | std |
|---|---:|---:|
| terminal_acc | 0.6051 | 0.0027 |
| yaku_micro_precision | 0.8690 | 0.0299 |
| yaku_micro_recall | 0.5598 | 0.0111 |
| yaku_micro_F1 | 0.6805 | 0.0062 |
| yaku_macro_F1 | 0.2105 | 0.0094 |
| exact_match | 0.3441 | 0.0075 |
| deal_in_AUC | 0.5823 | 0.0194 |
| deal_in_PR_AUC | 0.2067 | 0.0102 |

### Terminal Recall

| class | recall_mean | precision_mean | seed42 | seed43 | seed44 |
|---|---:|---:|---:|---:|---:|
| `win_menzen` | 0.4367 | 0.5073 | 0.5183 | 0.3761 | 0.4158 |
| `win_called` | 0.4373 | 0.4885 | 0.5973 | 0.2839 | 0.4306 |
| `draw_tenpai` | 0.1013 | 0.6391 | 0.1205 | 0.0790 | 0.1044 |
| `deal_in` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `other_non_dealin` | 0.8948 | 0.6295 | 0.8560 | 0.9228 | 0.9055 |

### Yaku Recall

| yaku | recall_mean | precision_mean | support_mean | seed42 | seed43 | seed44 |
|---|---:|---:|---:|---:|---:|---:|
| `Riichi` | 0.9898 | 0.9538 | 20199 | 0.9930 | 0.9913 | 0.9851 |
| `Yakuhai` | 0.7271 | 0.9418 | 4304 | 0.6843 | 0.7111 | 0.7860 |
| `Tanyao` | 0.3617 | 0.7896 | 5068 | 0.4515 | 0.3512 | 0.2824 |
| `Pinfu` | 0.3439 | 0.5083 | 7171 | 0.2213 | 0.3561 | 0.4542 |
| `MenzenTsumo` | 0.0003 | 0.0533 | 5150 | 0.0000 | 0.0009 | 0.0000 |
| `Ippatsu` | 0.0052 | 0.2585 | 4828 | 0.0009 | 0.0126 | 0.0022 |
| `Iipeiko` | 0.0000 | 0.0000 | 1247 | 0.0000 | 0.0000 | 0.0000 |
| `Toitoi` | 0.0000 | 0.0000 | 127 | 0.0000 | 0.0000 | 0.0000 |
| `SanshokuDoujun` | 0.0000 | 0.0000 | 387 | 0.0000 | 0.0000 | 0.0000 |

読み:

- yaku は `exp_023` 以前と比べてかなり実用的になった。
- `Riichi / Yakuhai / Tanyao / Pinfu` は明確に学習されている。
- `MenzenTsumo / Ippatsu / Iipeiko / Toitoi / Sanshoku` は依然として弱い。
- terminal は `win_menzen / win_called` はある程度拾うが、`deal_in` は top1 recall が 0。
- deal-in は AUC が平均 `0.58` あるので、完全に情報がないわけではない。分類 threshold/top1 の問題も残っている。

## Comparison To P5x

`exp_033` の `policy_lr=5x` は seed42 では良かったが、seed43 で明確に不安定だった。

```text
exp_033 P5x seed43:
final  2.34
best   2.21
best10 2.264
tail10 2.348
tail20 2.323
```

一方、`exp_034` の `policy_lr=1x` は seed43/44 でも崩れない。

```text
exp_034 P1x seed43:
final  2.18
best   2.005
best10 2.046
tail10 2.123
tail20 2.136

exp_034 P1x seed44:
final  2.095
best   2.030
best10 2.067
tail10 2.119
tail20 2.131
```

この差は大きい。  
`P5x` は上振れを引けるが、ルール拡張前の基準設定としては `P1x` の方が安全である。

## Interpretation

### 1. policy lr は上げすぎない方がよい

`P5x` は seed42 で強かったが、seed43 で崩れた。  
`P1x` は seed42/43/44 すべてで強く、tail も安定している。

したがって、現状の Stage2a では policy 更新の速度を上げるより、value/semantic 側を速くし、policy は標準 lr で追従させる方が安定する。

### 2. value/semantic lr=100x は採用価値が高い

`value_semantic_lr=0.01` は、policy を 1x に戻しても高い性能を維持した。

これは、以前の `TERM50x` 的な「terminal が shared representation を押す」効果を、より整理された optimizer group で実現できている可能性が高い。

### 3. target_kl は保険として残してよい

Final cycle では target_kl stop は発動していないが、悪い踏み込みを抑える safety として残してよい。  
今回の設定では性能を邪魔している様子はない。

### 4. semantic head は強さの全てを説明しない

yaku head はかなり改善したが、terminal head はまだ限定的である。  
それでも performance は非常に強い。

したがって、terminal auxiliary は「終局ラベルを正確に分類するため」というより、value/semantic trunk の表現を shaped する役割が大きい可能性がある。

## Decision

`exp_034` 条件を Stage2a のルール拡張前 baseline として採用する。

採用設定:

```text
policy_ratio = 1.0
ppo_mode = separated
policy_anchor = off
reward.point_delta_scale = 0.0001
tile_presence_flags = on
semantic_aux = on
terminal_loss_coef = 0.1
yaku_loss_coef = 0.05
policy_lr = 0.0001
value_semantic_lr = 0.01
lr_groups.apply_to = ["ppo"]
target_kl = on
```

次の自然なステップ:

1. `exp_034` を最終 Stage2a baseline として固定する。
2. この条件を基準に、ルール拡張へ進む。
3. ルール拡張後も、まずは `policy_lr=1x / value_semantic_lr=100x / target_kl=on` を初期設定として使う。

