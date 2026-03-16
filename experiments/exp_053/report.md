# Experiment Report: exp_053

作成日: 2026-03-16  
対象: [experiments/exp_053/runbook.md](/home/takeo1116/Git/majong-rl/experiments/exp_053/runbook.md)  
目的: `imitation=0 + policy_ratio=0.0 + mixed_ppo` の簡約問題で、`lr / clip / batch / gae` を振って「最初は上がるがその後戻る」挙動の主因が update overshoot か target 設計かを切り分ける

## 1. 実験概要

- 条件: 6条件
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

- A: `baseline_mixed_rule_only`
  - `lr=5e-5`, `clip=0.15`, `batch=512`, `gae=0.85`
- B: `low_lr`
  - `lr=2.5e-5`
- C: `low_clip`
  - `clip=0.075`
- D: `low_lr_low_clip`
  - `lr=2.5e-5`, `clip=0.075`
- E: `large_batch`
  - `batch=1024`
- F: `low_gae`
  - `gae=0.70`

重要な基準:

- 本 report の初期基準は、batch summary の top-level `eval_before` ではなく、**各 seed の `cycle0.eval_before`** を使っている。
- この簡約問題では top-level `eval_before` が途中段階の評価を指すため、初期比較には不適切。

## 2. 実行結果

- 成功条件数: `6/6`
- 失敗条件数: `0`
- 各条件とも `success_count == 3`, `failure_count == 0`

所要時間（条件ごとの seed 平均）:

| 条件 | total sec | selfplay sec | learner sec |
|---|---:|---:|---:|
| A baseline | `663.6 ± 8.4` | `22.4` | `641.2` |
| B low_lr | `674.3 ± 8.8` | `22.6` | `651.6` |
| C low_clip | `659.6 ± 7.8` | `22.3` | `637.3` |
| D low_lr_low_clip | `656.8 ± 5.1` | `22.3` | `634.5` |
| E large_batch | `652.5 ± 2.7` | `22.6` | `629.9` |
| F low_gae | `660.9 ± 3.4` | `22.2` | `638.7` |

## 3. 初期基準と最終 after

初期基準（各 seed の `cycle0.eval_before`）は全条件共通:

| 指標 | 初期基準 |
|---|---:|
| avg_rank | `3.7058 ± 0.0150` |
| avg_score | `-18180.8 ± 349.4` |

最終 after（`cycle9.eval`）との比較:

| 条件 | final avg_rank | final avg_score | final vs init |
|---|---:|---:|---:|
| A baseline | `3.6233 ± 0.0457` | `-16764.2 ± 1048.3` | `rank -0.0825`, `score +1416.6` |
| B low_lr | `3.6375 ± 0.0361` | `-17191.7 ± 751.4` | `rank -0.0683`, `score +989.2` |
| C low_clip | `3.6417 ± 0.0279` | `-17011.0 ± 610.0` | `rank -0.0642`, `score +1169.8` |
| D low_lr_low_clip | `3.6550 ± 0.0141` | `-17202.1 ± 406.0` | `rank -0.0508`, `score +978.8` |
| E large_batch | `3.6400 ± 0.0368` | `-17176.1 ± 909.1` | `rank -0.0658`, `score +1004.8` |
| F low_gae | `3.6125 ± 0.0388` | `-16558.4 ± 481.0` | `rank -0.0933`, `score +1622.4` |

所見:

- 6条件すべてで、3-seed 平均の最終値は初期基準より良い。
- ただし update を弱めた条件（`low_lr`, `low_clip`, `low_lr_low_clip`, `large_batch`）が baseline を大きく上回るわけではなかった。
- `low_gae` だけが、最終着地でもはっきりと一段上に出た。

## 4. peak と崩れ方

各 seed の最良 cycle を取って平均した `best gain` と、そこから final までの戻り幅。

| 条件 | best rank gain | best score gain | best->final rank | best->final score |
|---|---:|---:|---:|---:|
| A baseline | `-0.1467` | `+2780.7` | `+0.0642` | `-1364.1` |
| B low_lr | `-0.1333` | `+2269.5` | `+0.0650` | `-1280.3` |
| C low_clip | `-0.1117` | `+1886.8` | `+0.0475` | `-716.9` |
| D low_lr_low_clip | `-0.1025` | `+2068.1` | `+0.0517` | `-1089.3` |
| E large_batch | `-0.1250` | `+2530.3` | `+0.0592` | `-1525.6` |
| F low_gae | `-0.1742` | `+3325.8` | `+0.0808` | `-1703.3` |

best cycle の平均:

| 条件 | best rank cycle | best score cycle |
|---|---:|---:|
| A baseline | `4.33` | `4.33` |
| B low_lr | `5.33` | `5.67` |
| C low_clip | `4.33` | `3.67` |
| D low_lr_low_clip | `3.33` | `5.00` |
| E large_batch | `4.67` | `5.33` |
| F low_gae | `3.33` | `3.33` |

所見:

- update を弱める条件は、確かに `best->final` の落ち幅を少し小さくする。
- ただし、その代わり peak も低くなりがちで、**最終性能での優位にはつながらない**。
- `low_gae` は peak も最終も一番強いが、戻り自体はまだ残る。

## 5. cycle 推移の比較

代表として A baseline と F low_gae を比較する。

| cycle | A after avg_rank | A after avg_score | F after avg_rank | F after avg_score |
|---:|---:|---:|---:|---:|
| 0 | `3.6375` | `-16703.8` | `3.5842` | `-16007.9` |
| 1 | `3.6217` | `-16185.0` | `3.5650` | `-15496.5` |
| 3 | `3.6142` | `-16313.2` | `3.5517` | `-15077.8` |
| 5 | `3.6317` | `-16895.8` | `3.5583` | `-15271.0` |
| 9 | `3.6233` | `-16764.2` | `3.6125` | `-16558.4` |

対応する learner 診断:

| cycle | A clip | A ratio_std | A value_error | F clip | F ratio_std | F value_error |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0.0178` | `0.0470` | `0.0748` | `0.0211` | `0.0505` | `0.0386` |
| 1 | `0.0553` | `0.0721` | `0.0698` | `0.0899` | `0.0835` | `0.0372` |
| 3 | `0.0244` | `0.0670` | `0.0601` | `0.0787` | `0.0875` | `0.0346` |
| 5 | `0.0251` | `0.0692` | `0.0514` | `0.0437` | `0.0778` | `0.0319` |
| 9 | `0.0609` | `0.0814` | `0.0372` | `0.0564` | `0.0793` | `0.0271` |

所見:

- `low_gae` は update 指標（`clip_fraction`, `ratio_std`）を特別小さくしていない。
- それでも `value_error_mean` と後述の `late.value_error` はかなり小さい。
- このため、`low_gae` の改善は「更新が穏やかだから」ではなく、**advantage target の質が上がったから** と読むのが自然。

## 6. 診断値の比較

final cycle の診断平均:

| 条件 | clip_fraction | ratio_std | value_error_mean | adv_abs_before_clip | late.value_error |
|---|---:|---:|---:|---:|---:|
| A baseline | `0.0609` | `0.0814` | `0.0372` | `0.5725` | `0.0370` |
| B low_lr | `0.0082` | `0.0522` | `0.0378` | `0.5427` | `0.0375` |
| C low_clip | `0.0949` | `0.0455` | `0.0372` | `0.5777` | `0.0362` |
| D low_lr_low_clip | `0.0188` | `0.0324` | `0.0378` | `0.5429` | `0.0374` |
| E large_batch | `0.0228` | `0.0673` | `0.0375` | `0.5480` | `0.0372` |
| F low_gae | `0.0564` | `0.0793` | `0.0271` | `0.4696` | `0.0273` |

追加で見ると、

- A baseline の `worsen_adv_mean` は平均で `+0.0095`
- F low_gae の `worsen_adv_mean` は平均で `-0.0009`

つまり `low_gae` では、**shanten を悪化させる手に対して advantage がより自然な負方向へ寄っている**。

## 7. 解釈

今回の 6 条件比較から言えることは次の 3 点。

1. `lr / clip / batch` を調整して update magnitude を弱めても、問題は本質的には解けない。  
2. `gae_lambda=0.70` は、peak も final も最も良かった。  
3. 改善の主因は overshoot 抑制ではなく、**long-horizon の noisy な credit assignment を減らしたこと** である可能性が高い。

特に重要なのは、

- `low_lr` や `low_lr_low_clip` は `clip_fraction` と `ratio_std` をかなり下げた
- それでも `low_gae` に負けた

という点で、これは「更新が強すぎるだけ」説に対するかなり強い反証になっている。

## 8. 結論

1. この簡約問題では、`imitation=0 + rule-only mixed PPO` でも平均的には学習できる。  
2. ただし `lr / clip / batch` をいじるだけでは、「最初は上がるが戻る」問題は本質的に解けない。  
3. `gae_lambda=0.70` が最良だったことから、次の主戦場は update magnitude ではなく **advantage/target horizon**。  
4. 次段では `gae` をさらに低い帯に広げて、`0.70` より短い horizon が効くかを詰めるのが自然。
