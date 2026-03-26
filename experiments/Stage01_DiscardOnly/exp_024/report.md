# Experiment Report: exp_024

作成日: 2026-03-10  
対象: `experiments/exp_024/runbook.md`  
目的: joint imitation と value head 専用 current shanten 特徴が、value/target 品質と PPO 後悪化を改善するかを切り分ける

## 1. 実験概要

比較条件:

- A: baseline
  - `training.imitation_value_warmstart.enabled=false`
  - `model.value_features.current_shanten.enabled=false`
- B: joint imitation small
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.1`
  - `model.value_features.current_shanten.enabled=false`
- C: joint imitation medium
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.5`
  - `model.value_features.current_shanten.enabled=false`
- D: joint imitation small + value current_shanten
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.1`
  - `model.value_features.current_shanten.enabled=true`
- E: joint imitation medium + value current_shanten
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.5`
  - `model.value_features.current_shanten.enabled=true`

共通固定:

- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- `feature_encoder.shanten_hint.enabled=true`
- `training.imitation_loss_mode=tie_aware_best_set`
- reward 条件:
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.01`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
- learner 条件:
  - `training.lr=0.0001`
  - `training.epochs=4`
  - `training.value_loss_coef=0.25`
  - `training.batch_size=256`
  - `training.gamma=0.99`
  - `training.gae_lambda=0.95`
  - `training.clip_epsilon=0.2`
- evaluation:
  - `evaluation.mode=rotation`
  - `evaluation.num_matches=30`

## 2. 実行結果

| 条件 | batch_dir | success |
|---|---|---:|
| A | （ローカル run） | 5/5 |
| B | （ローカル run） | 5/5 |
| C | （ローカル run） | 5/5 |
| D | （ローカル run） | 5/5 |
| E | （ローカル run） | 5/5 |

全条件で `summary.json.success=true`、`aggregate.eval_mode=rotation` を確認。  
joint imitation 有効条件では `summary.phase_stats.imitation.value_loss` と `summary.phase_stats.imitation.imitation_value_warmstart` を確認。  
`current_shanten` 有効条件では `summary.model_features.value_features.current_shanten.enabled=true` を確認。

## 3. 主診断: shanten 条件付き learner 信号

mean ± std（seed=5）

| 条件 | improve adv mean | improve adv pos ratio | worsen adv mean | worsen adv neg ratio | improve value_error mean | worsen value_error mean |
|---|---:|---:|---:|---:|---:|---:|
| A baseline | -0.0781 ± 0.0021 | 0.8171 ± 0.0071 | +0.0437 ± 0.0053 | 0.1313 ± 0.0055 | +201.8 ± 18.1 | +118.4 ± 9.6 |
| B joint 0.1 | -0.0877 ± 0.0052 | 0.7639 ± 0.0033 | +0.0608 ± 0.0044 | 0.1603 ± 0.0025 | +300.2 ± 21.8 | +173.4 ± 12.9 |
| C joint 0.5 | -0.0881 ± 0.0065 | 0.7631 ± 0.0046 | +0.0610 ± 0.0037 | 0.1608 ± 0.0027 | +296.8 ± 23.3 | +171.3 ± 13.1 |
| D joint 0.1 + vsh | -0.0878 ± 0.0049 | 0.7613 ± 0.0038 | +0.0599 ± 0.0050 | 0.1618 ± 0.0036 | +297.7 ± 22.4 | +172.6 ± 12.0 |
| E joint 0.5 + vsh | -0.0862 ± 0.0039 | 0.7616 ± 0.0054 | +0.0605 ± 0.0053 | 0.1616 ± 0.0036 | +294.6 ± 16.1 | +171.1 ± 8.3 |

所見:

1. **joint imitation を入れても `shanten_diag` の符号整合は回復しない。**  
   `improve` 群の advantage mean は全 joint 条件で baseline よりむしろ悪化し、`worsen` 群の advantage mean もより正寄りになった。
2. **value_error も baseline より大きい。**  
   `improve/worsen` の両群で `value_error mean` は baseline より大きく、joint imitation だけで value baseline 品質が改善したとは言えない。
3. **value current_shanten の追加効果は弱い。**  
   B→D、C→E を見ても `shanten_diag` はほぼ改善していない。

## 4. imitation 指標

mean ± std（seed=5）

| 条件 | teacher_top1 | teacher_best_set_hit | imitation value_loss |
|---|---:|---:|---:|
| A baseline | 0.1822 ± 0.0055 | 0.6017 ± 0.0069 | 0.0 ± 0.0 |
| B joint 0.1 | 0.1797 ± 0.0086 | 0.5876 ± 0.0067 | 9.04e6 ± 4.78e5 |
| C joint 0.5 | 0.1795 ± 0.0087 | 0.5874 ± 0.0069 | 9.04e6 ± 4.78e5 |
| D joint 0.1 + vsh | 0.1798 ± 0.0084 | 0.5884 ± 0.0076 | 9.05e6 ± 4.55e5 |
| E joint 0.5 + vsh | 0.1796 ± 0.0085 | 0.5884 ± 0.0076 | 9.05e6 ± 4.55e5 |

所見:

1. joint imitation は **teacher 再現率を少し下げる**。  
   `teacher_top1` / `teacher_best_set_hit_rate` は baseline が最良。
2. `coef=0.1` と `0.5` の差は小さい。  
   current shanten 有無を含めても imitation phase 指標はほぼ同等。

## 5. 副評価: eval_before -> eval

mean ± std（seed=5）

| 条件 | Δavg_rank | Δavg_score | Δdeal_in_rate | Δwin_rate |
|---|---:|---:|---:|---:|
| A baseline | +0.0683 ± 0.0819 | -657.5 ± 1394.6 | -0.0034 ± 0.0080 | -0.0140 ± 0.0076 |
| B joint 0.1 | -0.0150 ± 0.0388 | -197.5 ± 596.9 | +0.0069 ± 0.0058 | -0.0030 ± 0.0071 |
| C joint 0.5 | +0.0200 ± 0.0447 | -798.5 ± 943.4 | +0.0057 ± 0.0068 | -0.0033 ± 0.0119 |
| D joint 0.1 + vsh | +0.0583 ± 0.0831 | -826.5 ± 1385.7 | +0.0062 ± 0.0074 | -0.0076 ± 0.0034 |
| E joint 0.5 + vsh | +0.0483 ± 0.0462 | -768.2 ± 834.1 | +0.0058 ± 0.0131 | -0.0096 ± 0.0038 |

所見:

1. **通常評価では B（joint imitation 0.1）が最良。**  
   `Δavg_rank` は唯一負、`Δavg_score` も最も小さい。
2. **current shanten 追加は通常評価でも上乗せが見えない。**  
   D/E は B より悪化幅が大きい。
3. `coef=0.5` は `coef=0.1` を更新できない。  
   C/E は B より主評価が弱い。

## 6. after 指標

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A baseline | 3.4417 ± 0.0717 | -13563.5 ± 1201.0 | 0.0424 ± 0.0075 | 0.5719 ± 0.0143 |
| B joint 0.1 | 3.3833 ± 0.0880 | -13269.2 ± 1585.1 | 0.0468 ± 0.0128 | 0.5818 ± 0.0112 |
| C joint 0.5 | 3.4100 ± 0.0224 | -13691.2 ± 830.7 | 0.0462 ± 0.0059 | 0.5811 ± 0.0100 |
| D joint 0.1 + vsh | 3.4450 ± 0.0594 | -13740.7 ± 1377.7 | 0.0449 ± 0.0050 | 0.5817 ± 0.0140 |
| E joint 0.5 + vsh | 3.4350 ± 0.0208 | -13675.3 ± 590.5 | 0.0432 ± 0.0100 | 0.5800 ± 0.0099 |

所見:

1. after 指標でも **B が総合最良**。  
2. D/E は `deal_in_rate` が高止まりし、`avg_rank` / `avg_score` でも B を超えない。  

## 7. learner 補助指標

mean ± std（seed=5）

| 条件 | clip_fraction | value_error_mean | ratio_std |
|---|---:|---:|---:|
| A baseline | 0.7869 ± 0.0161 | 148.4 ± 13.6 | 0.7438 ± 0.0228 |
| B joint 0.1 | 0.5974 ± 0.0349 | 225.2 ± 14.6 | 0.7163 ± 0.1417 |
| C joint 0.5 | 0.6078 ± 0.0515 | 222.5 ± 16.0 | 1.0389 ± 0.6530 |
| D joint 0.1 + vsh | 0.5846 ± 0.0129 | 223.3 ± 15.3 | 0.6695 ± 0.0746 |
| E joint 0.5 + vsh | 0.5885 ± 0.0247 | 222.0 ± 11.3 | 0.6976 ± 0.0785 |

所見:

1. joint imitation は `clip_fraction` を大きく下げる。  
   これは PPO 更新を穏やかにしている。
2. 一方で `value_error_mean` は baseline より大きい。  
   つまり「通常評価の悪化縮小」と「`shanten_diag` / value_error の改善」は一致していない。
3. `coef=0.5` は `ratio_std` の不安定化リスクがある。  
   特に C は seed 分散が大きい。

## 8. 解釈

今回の結果は、かなりはっきりしている。

1. **joint imitation は通常評価を改善する。**  
   少なくとも `eval_before -> eval` と after 指標では、baseline より B が良い。
2. **しかし、その改善は `shanten_diag` の符号整合回復からは説明できない。**  
   `improve` は依然として負、`worsen` は依然として正で、しかも baseline より悪い。
3. **value current_shanten の追加価値は現時点では見えない。**  
   D/E は B を更新できず、`shanten_diag` も改善しない。

したがって、現時点で言えるのは:

- value warm start（joint imitation）は **通常評価の改善策としては有望**
- ただし **問題の本丸である `improve/worsen` 群の advantage 整合はまだ未解決**
- current shanten を value に 1 scalar 足すだけでは不十分

## 9. 結論

1. **暫定採用候補は B (`imitation_value_warmstart.coef=0.1`, `current_shanten=false`)**。  
   通常評価では最良で、`coef=0.5` や current shanten 追加を更新できない。
2. **value current_shanten は現段階では不採用**。  
   追加実装は入ったが、比較実験では明確な改善が見えない。
3. **joint imitation を入れても `shanten_diag` はまだ逆向き**。  
   よって、value/target 仮説は「joint imitation で解決」ではなく、まだ残っている。

## 10. 次アクション

1. B を暫定標準として固定し、`shanten_diag` がなお逆向きな理由をさらに診断する。  
2. 次段は
   - return/value の時間スケールずれ
   - shaping 対象と PPO target の整合
   - あるいは value 以外の target 設計
   を疑う。
3. current shanten 追加は、当面は優先度を下げる。  
