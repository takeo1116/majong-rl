# Experiment Report: exp_051

作成日: 2026-03-17  
対象: [experiments/exp_051/runbook.md](/home/takeo1116/Git/majong-rl/experiments/exp_051/runbook.md)  
目的: `rule_mix(actor3+rule1) + 2段学習` が、anchor-only 条件より PPO 崩壊傾向を緩和するかを `5 seeds x 10 cycles` で pilot 確認する

## 1. 実験概要

- 条件: 1条件
- seeds: `42..46`（5 seeds）
- cycles: `10`
- eval: `rotation, num_matches=100`
- 主要設定:
- `training.policy_anchor.enabled=true`
- `training.policy_anchor.coef=0.5`
- `training.entropy_coef=0.0`
- `training.rule_mix.enabled=true`
- `training.rule_mix.policy_ratio=0.75`
- `training.rule_mix.save_baseline_actions=true`
- `training.rule_mix_learner.enabled=true`
- `training.rule_mix_learner.order=baseline_then_policy`
- `training.rule_mix_learner.baseline_imitation_epochs=1`
- `training.rule_mix_learner.policy_ppo_epochs=1`

条件:

- A: `rule_mix_policy075_two_stage_anchor05_entropy0000_cycle10_eval100`

## 2. 実行結果

- batch_dir: `runs/20260315_stage1_full_flat_mlp_imitation_then_ppo_batch_6c91c310`
- success: `5/5`
- failure: `0`

phase timing（seed平均）:

| total sec | selfplay sec | learner sec |
|---:|---:|---:|
| `764.3 ± 27.8` | `22.9 ± 1.6` | `716.6 ± 26.7` |

## 3. imitation 基準と最終 after

初期基準は各 run の `cycle0.eval_before`、最終 after は `cycle9.eval`。

| 指標 | imitation 基準 | 最終 after |
|---|---:|---:|
| avg_rank | `3.3410` | `3.3530` |
| avg_score | `-12523.6` | `-12650.3` |

最終 vs imitation 基準:
- `Δavg_rank = +0.0120`
- `Δavg_score = -126.6`

所見:
- pilot の最終着地は imitation 基準より悪い。
- ただし悪化幅は大きくなく、後述のとおり途中 cycle では改善が見える。

## 4. peak と cycle 推移

best gain と、その後の戻り幅:

| 指標 | 値 |
|---|---:|
| best rank gain | `-0.0130` |
| best score gain | `+235.9` |
| best rank cycle | `5.0` |
| best score cycle | `5.6` |
| best->final rank | `+0.0250` |
| best->final score | `-362.6` |

代表 cycle の推移:

| cycle | before avg_rank | before avg_score | after avg_rank | after avg_score | cycle内 Δavg_rank | cycle内 Δavg_score |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `3.3410` | `-12523.6` | `3.3510` | `-12720.1` | `+0.0100` | `-196.4` |
| 1 | `3.3510` | `-12720.1` | `3.3575` | `-12588.1` | `+0.0065` | `+132.0` |
| 3 | `3.3725` | `-12862.2` | `3.3875` | `-13059.1` | `+0.0150` | `-197.0` |
| 5 | `3.3795` | `-12837.0` | `3.3570` | `-12588.7` | `-0.0225` | `+248.3` |
| 7 | `3.3660` | `-12662.8` | `3.3590` | `-12716.5` | `-0.0070` | `-53.7` |
| 9 | `3.3610` | `-12931.0` | `3.3530` | `-12650.3` | `-0.0080` | `+280.7` |

所見:
- cycle 0 は悪化、cycle 1 はやや回復。
- mean best は `cycle 5` 前後。
- その後は崩れ切るわけではないが、imitation 基準を安定して超え続ける形にはならない。

## 5. imitation 基準との比較

各 cycle の after を各 seed の `cycle0.eval_before` と比較した改善 seed 数:

| cycle | rank改善 seed数 | score改善 seed数 | 両方改善 |
|---:|---:|---:|---:|
| 0 | `2/5` | `1/5` | `1/5` |
| 1 | `2/5` | `1/5` | `1/5` |
| 3 | `0/5` | `0/5` | `0/5` |
| 5 | `2/5` | `2/5` | `1/5` |
| 7 | `2/5` | `3/5` | `2/5` |
| 9 | `2/5` | `2/5` | `1/5` |

所見:
- pilot でも `5/5` でそろって改善、というほど強い結果ではない。
- ただし `cycle 5` 付近には「少数 seed で両指標改善」が見え、完全に悲観一色でもなかった。

## 6. 診断補足

cycle 全体平均の learner 診断:

| 指標 | 値 |
|---|---:|
| clip_fraction | `0.0511` |
| ratio_std | `0.0757` |
| value_error_mean | `0.0121` |
| advantage_abs_mean_before_clip | `0.4416` |
| late.value_error | `0.0113` |

actor_type_counts（cycle平均）:
- policy: `87499.1`
- baseline: `29056.2`

所見:
- self-play の分布は意図どおり actor3+rule1 相当に寄っている。
- 学習自体は数値的に不安定ではなく、失敗要因は発散より学習信号の質にあると読める。

## 7. `exp_050` / `exp_052` との位置づけ

参照:
- [experiments/exp_050/report.md](/home/takeo1116/Git/majong-rl/experiments/exp_050/report.md)
- [experiments/exp_052/report.md](/home/takeo1116/Git/majong-rl/experiments/exp_052/report.md)

比較すると:
- `exp_050`（anchor only, 20x20）は imitation 基準を平均で超えられなかった。
- `exp_051` pilot は、`cycle 5` 前後で一時的に少し良く見えたため、rule_mix を大規模検証する動機になった。
- その後の `exp_052`（20x20）では、この楽観はかなり薄まり、「少しマシだが本質解ではない」と整理された。

## 8. 結論

1. `exp_051` は pilot としては一定の前向きシグナルを出した。  
2. ただし最終着地は imitation 基準を超えず、seed を広げる前から強い改善とは言いにくかった。  
3. その意味で `exp_051` は「rule_mix は少し効くかもしれない」という探索価値を示したが、「長期改善が作れた」とまでは言えない。  
4. 後続の `exp_052` でこの楽観が修正されたことを踏まえると、`exp_051` は pilot としての役割を果たした実験と位置づけるのが自然。
