# Experiment Report: exp_054

作成日: 2026-03-16  
対象: [experiments/exp_054/runbook.md](/home/takeo1116/Git/majong-rl/experiments/exp_054/runbook.md)  
目的: `imitation=0 + policy_ratio=0.0 + mixed_ppo` の簡約問題で、`gae_lambda=0.0 / 0.3 / 0.6` を追加探索し、`exp_053` の `0.70 / 0.85` と合わせて有効な horizon 帯を特定する

## 1. 実験概要

- 条件: 3条件
- seeds: `42,43,44`（3 seeds）
- cycles: `10`
- eval: `rotation, num_matches=100`
- 固定条件:
- `experiment.phases=["selfplay","learner","eval"]`
- `training.rule_mix.enabled=true`
- `training.rule_mix.policy_ratio=0.0`
- `training.rule_mix_learner.enabled=true`
- `training.rule_mix_learner.ppo_mode=mixed`
- `training.policy_anchor.enabled=false`
- `training.entropy_coef=0.0`

条件一覧:

- A: `gae_000`
  - `gae_lambda=0.0`
- B: `gae_030`
  - `gae_lambda=0.3`
- C: `gae_060`
  - `gae_lambda=0.6`

比較上の注意:

- 本 report の初期基準は **各 seed の `cycle0.eval_before`**。
- `exp_053` の `gae=0.70` と `gae=0.85` を参照系列として使い、`gae sweep` として読む。

## 2. 実行結果

- 成功条件数: `3/3`
- 失敗条件数: `0`

所要時間（条件ごとの seed 平均）:

| 条件 | total sec | selfplay sec | learner sec |
|---|---:|---:|---:|
| A gae_000 | `688.1 ± 17.5` | `23.2` | `664.9` |
| B gae_030 | `682.9 ± 18.4` | `23.7` | `659.2` |
| C gae_060 | `676.0 ± 7.4` | `23.0` | `653.0` |

## 3. 初期基準と最終 after

初期基準（各 seed の `cycle0.eval_before`）は `exp_053` と同じ:

| 指標 | 初期基準 |
|---|---:|
| avg_rank | `3.7058 ± 0.0150` |
| avg_score | `-18180.8 ± 349.4` |

新規 3 条件の最終 after:

| 条件 | final avg_rank | final avg_score | final vs init |
|---|---:|---:|---:|
| A gae_000 | `3.5058 ± 0.0490` | `-14470.2 ± 872.3` | `rank -0.2000`, `score +3710.6` |
| B gae_030 | `3.5208 ± 0.0430` | `-14615.3 ± 781.3` | `rank -0.1850`, `score +3565.5` |
| C gae_060 | `3.5758 ± 0.0477` | `-16036.7 ± 642.9` | `rank -0.1300`, `score +2144.2` |

所見:

- 3 条件とも final は初期基準より大きく改善。
- しかも **低 `gae` ほど final が良い** という単調な傾向が出た。

## 4. `gae sweep` 全体比較

`exp_053` の参照系列を含めて並べる。

| gae | final avg_rank | final avg_score | final vs init |
|---|---:|---:|---:|
| `0.85` | `3.6233` | `-16764.2` | `rank -0.0825`, `score +1416.6` |
| `0.70` | `3.6125` | `-16558.4` | `rank -0.0933`, `score +1622.4` |
| `0.60` | `3.5758` | `-16036.7` | `rank -0.1300`, `score +2144.2` |
| `0.30` | `3.5208` | `-14615.3` | `rank -0.1850`, `score +3565.5` |
| `0.00` | `3.5058` | `-14470.2` | `rank -0.2000`, `score +3710.6` |

best gain と戻り幅:

| gae | best rank gain | best score gain | best->final rank | best->final score |
|---|---:|---:|---:|---:|
| `0.85` | `-0.1467` | `+2780.7` | `+0.0642` | `-1364.1` |
| `0.70` | `-0.1742` | `+3325.8` | `+0.0808` | `-1703.3` |
| `0.60` | `-0.2183` | `+3853.2` | `+0.0883` | `-1709.0` |
| `0.30` | `-0.2592` | `+4609.6` | `+0.0742` | `-1044.1` |
| `0.00` | `-0.2650` | `+4506.9` | `+0.0650` | `-796.3` |

best cycle の平均:

| gae | best rank cycle | best score cycle |
|---|---:|---:|
| `0.85` | `4.33` | `4.33` |
| `0.70` | `3.33` | `3.33` |
| `0.60` | `4.67` | `5.00` |
| `0.30` | `5.00` | `5.00` |
| `0.00` | `4.33` | `5.33` |

所見:

- この範囲では `gae` を下げるほど peak も final も改善。
- 特に `0.30` と `0.00` は peak が高いだけでなく、`best->final` の戻り幅も比較的小さい。
- 今回の sweep だけを見るなら、**最適帯は `0.0` 付近に寄っている**。

## 5. 診断値の比較

以下は各条件の **10 cycles x 3 seeds 平均** の診断値。

| gae | clip_fraction | ratio_std | value_error_mean | adv_abs_before_clip | late.value_error |
|---|---:|---:|---:|---:|---:|
| `0.85` | `0.0366` | `0.0701` | `0.0545` | `0.5580` | `0.0592` |
| `0.70` | `0.0564` | `0.0772` | `0.0327` | `0.4645` | `0.0373` |
| `0.60` | `0.0694` | `0.0814` | `0.0257` | `0.4245` | `0.0301` |
| `0.30` | `0.0996` | `0.0884` | `0.0156` | `0.3380` | `0.0189` |
| `0.00` | `0.1192` | `0.0927` | `0.0112` | `0.2805` | `0.0137` |

重要点:

- `gae` を下げるほど `clip_fraction` と `ratio_std` はむしろ上がる。
- それなのに性能は良くなる。
- 同時に `value_error_mean`, `advantage_abs_mean_before_clip`, `late.value_error` は強く下がる。

これは、今回の改善が「更新を弱めたから」ではなく、**advantage target のノイズを大きく減らしたから** であることを示唆する。

## 6. `gae=0.0` のログから見えること

`gae=0.0` の 3 seed を見ると、

- `cycle 3-5` で peak に達する
- その後 `cycle 6-9` で少し戻る

という挙動は確かに残っている。たとえば:

- seed `cf33c95b`
  - peak: `cycle 3`, `3.485 / -14682.2`
  - final: `cycle 9`, `3.575 / -15546.2`
- seed `40a1a667`
  - peak: `cycle 5`, `3.405 / -12961.8`
  - final: `cycle 9`, `3.475 / -14454.8`

ただし、この戻りは単純な数値崩壊ではない。

- `value_error_mean` は `cycle` を通じてほぼ下がり続ける
- `late.value_error` も下がり続ける
- `worsen_adv` はより負側へ寄る

つまり `gae=0.0` では、

- learner 内部の整合性はむしろ上がっている
- それでも外部評価は少し戻る

という形になっている。  
これは **不安定化というより、短期 target への自己適応による objective mismatch** を疑う方が自然。

## 7. seed 安定性

final が初期基準より良かった seed 数:

| gae | rank改善 | score改善 | 両方改善 |
|---|---:|---:|---:|
| `0.85` | `3/3` | `3/3` | `3/3` |
| `0.70` | `3/3` | `3/3` | `3/3` |
| `0.60` | `3/3` | `3/3` | `3/3` |
| `0.30` | `3/3` | `3/3` | `3/3` |
| `0.00` | `3/3` | `3/3` | `3/3` |

ただし改善幅は低 `gae` ほど明確に大きい。  
今回の結果は、3-seed とはいえ方向感としてかなり揃っている。

## 8. 解釈

今回の `gae sweep` から言えることはかなり強い。

1. 現行の簡約問題では、`gae=0.85` は明らかに長すぎる。  
2. `0.70` でもまだ長く、`0.60 -> 0.30 -> 0.00` と下げるほど改善した。  
3. つまり、現状の rule-only mixed PPO では **long-horizon credit assignment が強いノイズ源** になっている可能性が高い。  
4. 一方で `gae=0.0` でも peak 後の戻りは少し残るため、問題は完全には解けていない。

## 9. 結論

1. `gae` を下げることは、この簡約問題で最も効いたノブだった。  
2. 今回の sweep では `gae=0.0` が final 最良で、`0.3` も非常に強かった。  
3. したがって次段は、`0.0` 近傍（例: `0.0 / 0.1 / 0.2 / 0.3`）をさらに詰めるのが自然。  
4. その後、ここで見つけた低 `gae` 設定を `imitationあり` や `policy_ratio>0` のより本番に近い setting に戻して再検証する。
