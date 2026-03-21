# Experiment Report: exp_023

作成日: 2026-03-10  
対象: `experiments/exp_023/runbook.md`  
目的: reward shaping が learner の更新信号まで届いているかを、`shanten_diag` を使って直接診断する

## 1. 実験概要

比較条件:

- A: baseline reward
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.0`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
- B: 標準 shaping reward
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.01`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`

共通固定:

- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- `feature_encoder.shanten_hint={"enabled":true}`
- `training.imitation_loss_mode=tie_aware_best_set`
- `training.lr=0.0001`
- `training.epochs=4`
- `training.value_loss_coef=0.25`
- `training.batch_size=256`
- `training.gamma=0.99`
- `training.gae_lambda=0.95`
- `training.entropy_coef=0.01`
- `training.clip_epsilon=0.2`
- `evaluation.mode=rotation`
- `evaluation.num_matches=30`

## 2. 実行結果

| 条件 | batch_dir | success |
|---|---|---:|
| A | （ローカル run） | 5/5 |
| B | （ローカル run） | 5/5 |

両条件とも `shanten_diag.status=partial` で、`unavailable_samples` は各 run おおむね `800`。  
初回打牌など unavailable を除外したうえで、`improve/same/worsen` の 3 群が成立している。

## 3. 主診断: shanten 条件付き learner 信号

mean ± std（seed=5）

| 条件 | improve adv mean | improve adv pos ratio | worsen adv mean | worsen adv neg ratio | improve value_error mean | worsen value_error mean |
|---|---:|---:|---:|---:|---:|---:|
| A baseline | -0.0781 ± 0.0019 | 0.8171 ± 0.0063 | +0.0437 ± 0.0047 | 0.1313 ± 0.0049 | +201.76 ± 16.21 | +118.44 ± 8.62 |
| B shaping | -0.0781 ± 0.0019 | 0.8171 ± 0.0063 | +0.0437 ± 0.0047 | 0.1313 ± 0.0049 | +201.76 ± 16.21 | +118.45 ± 8.62 |

補助カウント（mean ± std）:

| 条件 | improve count | same count | worsen count | available | unavailable |
|---|---:|---:|---:|---:|---:|
| A baseline | 24754.6 ± 171.9 | 75685.6 ± 266.1 | 21125.8 ± 117.8 | 121566.0 ± 214.9 | 800.0 ± 0.0 |
| B shaping | 24754.6 ± 171.9 | 75685.6 ± 266.1 | 21125.8 ± 117.8 | 121566.0 ± 214.9 | 800.0 ± 0.0 |

所見:

1. **改善打牌群の advantage mean は依然として負**。  
   shaping を入れても `improve` 群が平均で正に転じていない。
2. **悪化打牌群の advantage mean は依然として正**。  
   `worsen` 群も平均で負になっておらず、期待する learner signal と逆方向。
3. **A/B の差はほぼゼロ**。  
   reward shaping を入れても、`shanten_diag` で見る `advantage/return/value_error` は実質的に変化していない。

この結果は、reward sparse 性の改善だけでは learner の更新信号整合が直っていないことを示す。

## 4. 副評価: eval_before -> eval

mean ± std（seed=5）

| 条件 | Δavg_rank | Δavg_score | Δdeal_in_rate | Δwin_rate |
|---|---:|---:|---:|---:|
| A baseline | +0.0833 ± 0.0312 | -753.2 ± 1050.5 | -0.0016 ± 0.0091 | -0.0168 ± 0.0060 |
| B shaping | +0.0683 ± 0.0733 | -657.5 ± 1247.4 | -0.0034 ± 0.0072 | -0.0140 ± 0.0068 |

所見:

- 標準 shaping reward の方が通常評価ではわずかにマシだが、改善幅は限定的。
- ただし今回の主目的は `shanten_diag` 診断なので、この差分は副次的に扱う。

## 5. reward 内訳

代表的な run の `reward_composition`:

| 条件 | point_delta mean | shanten_delta mean | total mean | shanten_delta p90 | shanten_delta p99 |
|---|---:|---:|---:|---:|---:|
| A baseline | -7.7538 | 0.000000 | -7.7538 | 0.0000 | 0.0000 |
| B shaping | -7.7538 | +0.000038 | -7.7538 | 0.0055 | 0.0100 |

所見:

- shaping reward 自体は B で確かに入っている。
- つまり **reward 成分は変わっているのに、shanten 条件付き learner signal はほぼ変わっていない**。

## 6. 解釈

今回の結果から言えることは次の通り。

1. reward shaping を入れても、`improve/worsen` 群で advantage の符号が期待方向へ動いていない。  
2. `value_error` も改善打牌群・悪化打牌群の両方で大きく正に偏っており、value baseline のズレが強く疑われる。  
3. したがって、現在のボトルネックは reward の sparse 性だけではなく、**value/target/advantage の品質** にある可能性が高い。  

特に重要なのは、`shanten_hint` と shaping reward により policy に必要な局所情報と即時報酬はかなり与えられているのに、  
learner が使う advantage はなお整合していない、という点である。

## 7. 結論

1. **reward shaping は必要だが、それだけでは PPO learner の更新信号整合は回復しない。**
2. **`improve` が負、`worsen` が正という `shanten_diag` の結果は、value/target 側を本命として疑う強い根拠** である。
3. 次段は reward の追加探索ではなく、**value 学習安定化仮説の検証** に進むべきである。

## 8. 次アクション

1. `value` 学習が局面価値を掴めていない仮説を検証する。  
   第一候補は「value に現在シャンテン数を直接与える」系の最小実装。
2. その前提として、必要なら value 入力拡張の CQ を起票する。  
3. reward 条件は当面 `linear_decay + scale=0.01 + mode=both` を維持し、今後の learner/value 比較では固定する。
