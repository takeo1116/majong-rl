# Experiment Report: exp_055

作成日: 2026-03-17  
対象: `experiments/exp_055/runbook.md`  
目的: `imitation=0 + policy_ratio=0.0 + mixed_ppo + gae=0.0` の簡約問題で、残っている plateau / 戻りが `gamma` 起因か `shanten shaping` 起因かを切り分ける

## 1. 実験概要

- 条件: 6条件
- seeds: `42..44`（3 seeds）
- cycles: `10`
- eval: `rotation, num_matches=100`
- 固定条件:
- `training.rule_mix.enabled=true`
- `training.rule_mix.policy_ratio=0.0`
- `training.rule_mix_learner.enabled=true`
- `training.rule_mix_learner.ppo_mode=mixed`
- `training.gae_lambda=0.0`
- `training.policy_anchor.enabled=false`
- `training.entropy_coef=0.0`

条件一覧:

| 条件 | gamma | shanten_delta.scale |
|---|---:|---:|
| A `gamma099_shanten0010` | `0.99` | `0.01` |
| B `gamma095_shanten0010` | `0.95` | `0.01` |
| C `gamma099_shanten0003` | `0.99` | `0.003` |
| D `gamma095_shanten0003` | `0.95` | `0.003` |
| E `gamma099_shanten0000` | `0.99` | `0.0` |
| F `gamma095_shanten0000` | `0.95` | `0.0` |

## 2. 実行結果

- 成功条件数: `6/6`
- 失敗条件数: `0`

所要時間（条件ごとの seed 平均）:

| 条件 | total sec | selfplay sec | learner sec |
|---|---:|---:|---:|
| A | `676.0 ± 4.2` | `22.1 ± 0.8` | `653.9 ± 3.4` |
| B | `687.8 ± 3.4` | `23.9 ± 0.9` | `663.9 ± 4.2` |
| C | `690.3 ± 10.0` | `24.0 ± 1.6` | `666.3 ± 9.1` |
| D | `689.9 ± 10.7` | `23.5 ± 1.1` | `666.4 ± 9.7` |
| E | `698.2 ± 12.1` | `23.1 ± 1.0` | `675.1 ± 11.5` |
| F | `690.3 ± 6.0` | `23.1 ± 0.7` | `667.2 ± 5.3` |

## 3. 初期基準と最終 after

初期基準（各 seed の `cycle0.eval_before`）は全条件共通:

| 指標 | 初期基準 |
|---|---:|
| avg_rank | `3.7058` |
| avg_score | `-18180.8` |

最終 after:

| 条件 | final avg_rank | final avg_score | final vs init |
|---|---:|---:|---:|
| A | `3.5058` | `-14470.2` | `rank -0.2000`, `score +3710.6` |
| B | `3.5008` | `-14288.0` | `rank -0.2050`, `score +3892.8` |
| C | `3.4983` | `-14289.0` | `rank -0.2075`, `score +3891.8` |
| D | `3.4933` | `-14187.8` | `rank -0.2125`, `score +3993.1` |
| E | `3.5050` | `-14488.8` | `rank -0.2008`, `score +3692.1` |
| F | `3.5067` | `-14201.3` | `rank -0.1992`, `score +3979.5` |

所見:
- 6条件すべてで final は初期基準より大きく改善。
- 最良条件は **D: `gamma=0.95, scale=0.003`**。
- `scale=0.003` は `0.01` より明確に良く、`0.0` よりもわずかに安定して強い。

## 4. peak と戻り方

| 条件 | best rank gain | best score gain | best rank cycle | best score cycle | best->final rank | best->final score |
|---|---:|---:|---:|---:|---:|---:|
| A | `-0.2650` | `+4506.9` | `4.33` | `5.33` | `+0.0650` | `-796.3` |
| B | `-0.2725` | `+4795.8` | `5.00` | `5.00` | `+0.0675` | `-903.0` |
| C | `-0.2567` | `+4366.8` | `5.67` | `5.67` | `+0.0492` | `-474.9` |
| D | `-0.2617` | `+4461.4` | `5.67` | `5.33` | `+0.0492` | `-468.3` |
| E | `-0.2467` | `+4268.2` | `5.00` | `7.00` | `+0.0458` | `-576.1` |
| F | `-0.2508` | `+4404.9` | `5.67` | `5.33` | `+0.0517` | `-425.4` |

所見:
- `gamma` を `0.95` に下げると、どの shaping 水準でも final は少し改善。
- ただし主効果としてより大きいのは `shanten_delta.scale=0.01 -> 0.003`。
- `scale=0.003` は peak を保ちながら `best->final` の戻りをかなり小さくする。

## 5. 主効果の見方

`gamma` の平均主効果:

| gamma | avg final Δrank | avg final Δscore | avg best->final score |
|---|---:|---:|---:|
| `0.99` | `-0.2028` | `+3764.8` | `-615.8` |
| `0.95` | `-0.2056` | `+3955.1` | `-598.9` |

`shanten_delta.scale` の平均主効果:

| scale | avg final Δrank | avg final Δscore | avg best->final score |
|---|---:|---:|---:|
| `0.01` | `-0.2025` | `+3801.7` | `-849.7` |
| `0.003` | `-0.2100` | `+3942.5` | `-471.6` |
| `0.0` | `-0.2000` | `+3835.8` | `-500.8` |

所見:
- `gamma=0.95` は確かに良いが、差分は中程度。
- `scale=0.003` の効果はよりはっきりしていて、特に戻りの縮小に強く効いている。
- shaping は「ゼロが最良」ではなく、**弱く残す方が良い** 可能性が高い。

## 6. teacher agreement

cycle 平均の teacher agreement:

| 条件 | action_match before | action_match after | best_set_hit before | best_set_hit after |
|---|---:|---:|---:|---:|
| A | `0.2208` | `0.2332` | `0.5012` | `0.5258` |
| B | `0.2210` | `0.2336` | `0.5038` | `0.5292` |
| C | `0.2282` | `0.2411` | `0.4935` | `0.5162` |
| D | `0.2281` | `0.2413` | `0.4961` | `0.5198` |
| E | `0.2298` | `0.2422` | `0.4902` | `0.5117` |
| F | `0.2304` | `0.2434` | `0.4927` | `0.5157` |

所見:
- 全条件で learner update 後に teacher へ近づいている。
- shaping を弱めるほど `action_match_rate` は上がる。
- 一方 `best_set_hit_rate` は `0.01` 条件が少し高い。
- 今回の性能改善は「best-set に広く入ること」より、**baseline が実際に打った action に近づくこと** とやや強く相関している。

## 7. 診断補足

cycle 全体平均の learner 診断:

| 条件 | clip_fraction | ratio_std | value_error_mean | adv_abs_before_clip | late.value_error |
|---|---:|---:|---:|---:|---:|
| A | `0.1192` | `0.0927` | `0.0112` | `0.2805` | `0.0137` |
| B | `0.1114` | `0.0910` | `0.0089` | `0.2809` | `0.0109` |
| C | `0.1254` | `0.0935` | `0.0113` | `0.2714` | `0.0143` |
| D | `0.1178` | `0.0919` | `0.0090` | `0.2714` | `0.0114` |
| E | `0.1265` | `0.0936` | `0.0114` | `0.2714` | `0.0144` |
| F | `0.1193` | `0.0921` | `0.0091` | `0.2708` | `0.0117` |

所見:
- `gamma=0.95` は `value_error_mean` と `late.value_error` を少し下げる。
- ただし今回の主因は update magnitude ではなく、reward/target の整形側に残っていたと読むのが自然。

## 8. 結論

1. `gae=0.0` の次に効いたのは `shanten shaping` の弱化だった。  
2. `gamma=0.95` も改善に寄与したが、主効果としては `scale=0.003` の方が大きかった。  
3. 今回の最良条件は **D: `gamma=0.95, shanten_delta.scale=0.003`**。  
4. したがって次段は `scale` をさらに細かく刻むより、`D` を基準に `gamma` を大胆に下げて、future bootstrap の残差を直接叩くのが自然。
