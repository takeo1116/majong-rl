# Experiment Report: exp_003

作成日: 2026-03-30  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_003/runbook.md`
- `experiments/Stage02_CallUnlock/exp_001/report.md`
- `experiments/Stage02_CallUnlock/exp_002/report.md`
- `experiments/Stage02_CallUnlock/exp_003/run_map.json`

## 1. 要約

`exp_003` では、`exp_002` で安定化を確認した `separated` PPO baseline の上で、
Stage02a の feature 条件を A/B/C で比較した。

比較条件:

1. A `core_minimal`
2. B `stage1style_context`
3. C `stage1style_context_plus_danger`

結論は以下。

- A/B/C の全条件で PPO は安定に完走した
- `separated` baseline は feature 比較の土台として十分安定している
- 1 seed の結果では、**B `stage1style_context` が総合的に最も有望**
- C `+ danger_mask` は悪くないが、B を明確には上回らなかった
- したがって、現時点の暫定 feature baseline 候補は **B** とするのが自然

## 2. 実験目的

`exp_001` では本来、Stage02a の feature 比較として A/B/C を比較する予定だった。
しかし実際には PPO 安定化と throughput 改善が先に必要になったため、
`exp_002` で `mixed` vs `separated` を比較し、`separated` を安定 baseline として採用した。

`exp_003` は、その stable baseline 上で、改めて A/B/C の feature 差を見るための実験である。

問いは以下。

1. `opponent_current_shanten` / `opponent_tenpai_flag` は Stage02a でも有効か
2. `danger_mask` を Stage02a に持ち込む価値があるか
3. 以後の Stage02 feature baseline として A/B/C のどれを採用すべきか

## 3. 実行条件

共通 PPO 条件:

- `training.rule_mix_learner.ppo_mode = "separated"`
- `training.rule_mix.policy_ratio = 0.25`
- `training.rule_mix_learner.baseline_sample_weight = 0.5`
- `training.policy_anchor.reference = "imitation_fixed"`
- `training.policy_anchor.coef = 0.5`
- `training.multi_cycle.num_cycles = 20`

共通 config:

- `configs/stage2a_core_minimal_separated_baseline.yaml`

実行管理:

- `experiments/Stage02_CallUnlock/exp_003/run_map.json`
- `scripts/local/stage2/exp_003_driver.py`

A は `exp_002` の既存 control run を流用し、B/C のみ新規実行した。

## 4. 対象 run

### A `core_minimal` (reused)

- source: `exp_002` の A2 `separated` control
- reused from: `exp_002/A2`

### B `stage1style_context`

- run label: `B_stage1style_context` （対応は `experiments/Stage02_CallUnlock/exp_003/run_map.json` を参照）

### C `stage1style_context_plus_danger`

- run label: `C_stage1style_context_plus_danger` （対応は `experiments/Stage02_CallUnlock/exp_003/run_map.json` を参照）

## 5. 主結果

### 最終値

| Condition | final avg_rank | final win_rate | final learner loss |
|---|---:|---:|---:|
| A core_minimal | 2.555 | 0.2312 | 0.0081 |
| B stage1style_context | 2.355 | 0.2282 | 0.0064 |
| C stage1style_context_plus_danger | 2.400 | 0.2395 | 0.0051 |

### 後半 5 cycle 平均

final 1 点だけでは評価ぶれの影響があるため、cycle 15-19 の平均も比較した。

| Condition | tail-5 avg_rank | tail-5 win_rate |
|---|---:|---:|
| A core_minimal | 2.550 | 0.2222 |
| B stage1style_context | 2.484 | 0.2320 |
| C stage1style_context_plus_danger | 2.530 | 0.2260 |

### 各条件の best point

| Condition | best avg_rank | cycle | best win_rate | cycle |
|---|---:|---:|---:|---:|
| A | 2.39 | 9 | 0.243 | 1 |
| B | 2.33 | 16 | 0.243 | 17 |
| C | 2.38 | 3 | 0.248 | 4 |

## 6. PPO 安定性

今回の重要点は、A/B/C 全条件で PPO が安定に完走したことにある。

最終 PPO diagnostics:

| Condition | ratio_mean | clip_fraction | anchor_kl_discard | anchor_kl_optional |
|---|---:|---:|---:|---:|
| A | 1.0035 | 0.2462 | 0.0688 | 0.00553 |
| B | 1.0027 | 0.2436 | 0.0737 | 0.00565 |
| C | 1.0043 | 0.2580 | 0.0666 | 0.00517 |

解釈:

- `ratio_mean` は全条件で 1.0 近傍に留まっている
- `clip_fraction` も 0.24-0.26 程度で、`mixed` のような崩れ方は見られない
- `anchor_kl_discard` は全条件で 0.07 前後に収まっている
- learner loss も全条件で低く、終盤発散は見られない

したがって、`separated` baseline は A/B/C feature 比較の足場として十分使えると判断する。

## 7. feature 比較の読み取り

### A → B

B は A に対して

- `opponent_current_shanten`
- `opponent_tenpai_flag`

を追加した条件である。

結果を見ると、

- final avg_rank: `2.555 -> 2.355` で改善
- tail-5 avg_rank: `2.550 -> 2.484` で改善
- tail-5 win_rate: `0.2222 -> 0.2320` で改善

であり、**opponent 文脈追加は Stage02a でも有望**と読める。

1 seed なので断定は避けるべきだが、少なくとも今回の run では B は A より悪化していない。
むしろ総合指標では B が最もきれいである。

### B → C

C は B に対して `danger_mask` を追加した条件である。

結果はやや混合的だった。

- final win_rate は C が最良 (`0.2395`)
- しかし final avg_rank は B の方が良い (`2.355` vs `2.400`)
- tail-5 平均でも B が C を上回る (`2.484 / 0.2320` vs `2.530 / 0.2260`)

したがって、**danger を足したことによる明確な上積みは今回は確認できなかった**。

現時点では、C は「悪くない」が「B を置き換える決定打はない」という位置づけである。

## 8. throughput 観点

A/B/C ともに imitation chunk は実験可能な時間で完了しており、
throughput 改善の成果は維持されている。

driver log の chunk timing から見ると、B/C は A よりやや重いが、研究実験として十分許容範囲である。

概況:

- A chunk total: 約 `244-248s`
- B chunk total: 約 `257-280s`
- C chunk total: 約 `265-296s`

追加特徴量で encoder / learner 負荷は少し増えるが、現時点では比較実験の障害にはならない。

## 9. 結論

今回の `exp_003` から得られた結論は以下。

1. `separated` baseline は A/B/C feature 比較を安定に回せる
2. Stage02a でも `opponent_current_shanten` / `opponent_tenpai_flag` は有望
3. `danger_mask` は 1 seed 時点では明確な上積みを示さなかった
4. 暫定 feature baseline 候補としては **B `stage1style_context` が最有力**

## 10. 次のアクション

次に自然なのは以下。

1. B を追加 seed で再確認する
2. 余力があれば C も追加 seed を回して、danger の効果が再現するかを見る
3. B 優勢が再現するなら、以後の Stage02 baseline を B へ更新する
4. その後、必要なら `mixed` parity や RL 改善量の追跡へ戻る

現時点では、**A/B/C の 1 seed 比較としては B 採用が最も自然**である。
