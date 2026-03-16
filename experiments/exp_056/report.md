# Experiment Report: exp_056

作成日: 2026-03-17  
対象: [experiments/exp_056/runbook.md](/home/takeo1116/Git/majong-rl/experiments/exp_056/runbook.md)  
目的: `imitation=0 + policy_ratio=0.0 + mixed_ppo + gae=0.0 + shanten_delta.scale=0.003` の簡約問題で、`gamma` を `0.75 / 0.50 / 0.25` まで大胆に下げたときに plateau / 戻りがどう変わるかを確認する

## 1. 実験概要

- 条件: 3条件
- seeds: `42..44`（3 seeds）
- cycles: `10`
- eval: `rotation, num_matches=100`
- 固定条件:
- `training.rule_mix.enabled=true`
- `training.rule_mix.policy_ratio=0.0`
- `training.rule_mix_learner.enabled=true`
- `training.rule_mix_learner.ppo_mode=mixed`
- `training.gae_lambda=0.0`
- `reward.shaping.shanten_delta.scale=0.003`
- `training.policy_anchor.enabled=false`

条件一覧:

| 条件 | gamma | scale |
|---|---:|---:|
| A `gamma075_shanten0003` | `0.75` | `0.003` |
| B `gamma050_shanten0003` | `0.50` | `0.003` |
| C `gamma025_shanten0003` | `0.25` | `0.003` |

参照条件:
- `exp_055` D: `gamma=0.95, scale=0.003`

## 2. 実行結果

- 成功条件数: `3/3`
- 失敗条件数: `0`

所要時間（条件ごとの seed 平均）:

| 条件 | total sec | selfplay sec | learner sec |
|---|---:|---:|---:|
| A | `727.4 ± 38.5` | `24.1 ± 2.2` | `703.3 ± 36.9` |
| B | `782.9 ± 20.8` | `28.3 ± 1.3` | `754.6 ± 20.1` |
| C | `760.8 ± 45.5` | `28.1 ± 3.3` | `732.8 ± 43.8` |

## 3. `exp_055` D との比較

参照条件 `gamma=0.95, scale=0.003`:

| 条件 | final avg_rank | final avg_score | best->final rank | best->final score |
|---|---:|---:|---:|---:|
| ref `gamma=0.95` | `3.4933` | `-14187.8` | `+0.0492` | `-468.3` |
| A `gamma=0.75` | `3.4233` | `-13337.2` | `+0.0108` | `-89.7` |
| B `gamma=0.50` | `3.4308` | `-13290.7` | `+0.0175` | `-183.8` |
| C `gamma=0.25` | `3.4450` | `-13616.4` | `+0.0292` | `-256.5` |

所見:
- 3 条件すべてが参照条件 `0.95` より良い。
- 特に `0.75` と `0.50` は final も peak もかなり改善し、戻りも大きく減った。
- `0.25` も `0.95` より良いが、`0.50 / 0.75` よりは少し落ちる。

## 4. 初期基準と最終 after

初期基準（各 seed の `cycle0.eval_before`）は共通:

| 指標 | 初期基準 |
|---|---:|
| avg_rank | `3.7058` |
| avg_score | `-18180.8` |

最終 after:

| 条件 | final avg_rank | final avg_score | final vs init |
|---|---:|---:|---:|
| A `0.75` | `3.4233` | `-13337.2` | `rank -0.2825`, `score +4843.6` |
| B `0.50` | `3.4308` | `-13290.7` | `rank -0.2750`, `score +4890.2` |
| C `0.25` | `3.4450` | `-13616.4` | `rank -0.2608`, `score +4564.4` |

所見:
- best rank は `0.75`、best score は `0.50`。
- 3 条件とも `exp_055` より一段強い改善を出している。

## 5. peak と戻り方

| 条件 | best rank gain | best score gain | best rank cycle | best score cycle | best->final rank | best->final score |
|---|---:|---:|---:|---:|---:|---:|
| A `0.75` | `-0.2933` | `+4933.3` | `7.33` | `7.00` | `+0.0108` | `-89.7` |
| B `0.50` | `-0.2925` | `+5074.0` | `6.00` | `6.67` | `+0.0175` | `-183.8` |
| C `0.25` | `-0.2900` | `+4820.9` | `7.67` | `7.33` | `+0.0292` | `-256.5` |

所見:
- plateau / 戻りは明確に縮小した。
- 特に `gamma=0.75` では、ほぼ peak に張り付いたまま終わっている。
- `gamma` を下げると peak cycle は `6-8` 付近へ後ろにずれ、改善が長く続く。

## 6. teacher agreement

| 条件 | action_match before | action_match after | best_set_hit before | best_set_hit after |
|---|---:|---:|---:|---:|
| ref `0.95` | `0.2281` | `0.2413` | `0.4961` | `0.5198` |
| A `0.75` | `0.2246` | `0.2384` | `0.5068` | `0.5353` |
| B `0.50` | `0.2203` | `0.2338` | `0.5106` | `0.5408` |
| C `0.25` | `0.2165` | `0.2297` | `0.5118` | `0.5429` |

所見:
- `gamma` を下げると exact action 一致率はむしろ少し下がる。
- 一方で `best_set_hit_rate` は一貫して上がる。
- 今回の改善は、baseline の一手を完全コピーしたというより、**rule teacher の良い選択肢集合により多く入るようになった** と読む方が自然。

## 7. 診断補足

| 条件 | clip_fraction | ratio_std | value_error_mean | adv_abs_before_clip | late.value_error |
|---|---:|---:|---:|---:|---:|
| ref `0.95` | `0.1178` | `0.0919` | `0.0090` | `0.2714` | `0.0114` |
| A `0.75` | `0.1007` | `0.0893` | `0.0028` | `0.2766` | `0.0039` |
| B `0.50` | `0.0928` | `0.0883` | `0.0004` | `0.2832` | `0.0007` |
| C `0.25` | `0.0888` | `0.0875` | `-0.0001` | `0.2889` | `-0.0002` |

所見:
- `gamma` を下げると `clip_fraction` と `ratio_std` は少し下がるが、ここが主因というよりは `late.value_error` の急低下が目立つ。
- `0.50` から `0.25` で `late.value_error` はさらに下がる一方、性能は少し悪化するため、「短くすればよい」ではなく最適帯があると分かる。

## 8. 解釈

今回かなり強く言えること:

1. `gae=0.0` にしても、`gamma=0.95` の future bootstrap はまだ長すぎた。  
2. `gamma=0.75` または `0.50` にすると、peak も final も改善し、しかも plateau / 戻りが大きく減る。  
3. `gamma=0.25` まで下げると改善はまだ強いが、`0.50 / 0.75` よりは少し悪くなる。  
4. よって、現状の有望帯は **`gamma=0.50〜0.75`**。

## 9. 結論

1. `exp_056` は、`gamma` が plateau / 戻りの主因の一つだったことをかなり強く支持した。  
2. `gamma=0.95` はまだ長すぎ、`0.75` または `0.50` まで下げた方が良い。  
3. ただし `0.25` は少し下げすぎの兆候もあり、最適帯は `0.50〜0.75` にある可能性が高い。  
4. 次段は、この帯の局所探索をしてから、より本番に近い setting（例: imitation あり）へ戻すのが自然。
