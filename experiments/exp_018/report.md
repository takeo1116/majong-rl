# Experiment Report: exp_018

作成日: 2026-03-08  
対象: `experiments/exp_018/runbook.md`

## 1. 実験概要

- 目的: `shanten_hint=true` / `imitation_loss_mode=tie_aware_best_set` を固定し、PPO の更新強度を弱めると `eval_before -> eval` の悪化が緩和されるかを確認する。
- 実行方式:
  - seed ごとに REF（full run）を 1 本作成
  - `imitation,selfplay,eval_before` を再利用して A/B/C を分岐
  - learner 診断統計と reward 分布を併読
- seeds: `42,43,44,45,46`
- 比較条件:
  - REF: `epochs=4`, `lr=1e-4`（参照 run）
  - A: `epochs=4`, `lr=1e-4`（baseline reuse）
  - B: `epochs=2`, `lr=1e-4`（weak-epochs）
  - C: `epochs=4`, `lr=5e-5`（weak-lr）
- 主評価の優先順:
  1. `eval_before -> eval` の `Δavg_rank`
  2. `Δavg_score`
  3. `Δdeal_in_rate`
  4. `Δwin_rate`

## 2. 実行結果

| 条件 | run/batch | success |
|---|---|---:|
| REF | （ローカル run） x5 | 5/5 |
| A | reuse run x5 | 5/5 |
| B | reuse run x5 | 5/5 |
| C | reuse run x5 | 5/5 |

注記:
- 比較の正本は `experiments/exp_018/run_map.json`。
- A/B/C は同一 seed の REF から `imitation,selfplay,eval_before` を再利用しているため、self-play 分布は条件間で共通。

## 3. 主評価

mean ± std（seed=5）

| 条件 | Δavg_rank | Δavg_score | Δdeal_in_rate | Δwin_rate |
|---|---:|---:|---:|---:|
| A (`epochs=4`, `lr=1e-4`) | +0.0683 ± 0.1076 | -1167.8 ± 1495.5 | +0.0033 ± 0.0139 | -0.0152 ± 0.0116 |
| B (`epochs=2`, `lr=1e-4`) | +0.1883 ± 0.1124 | -2302.8 ± 2278.6 | +0.0172 ± 0.0175 | -0.0279 ± 0.0177 |
| C (`epochs=4`, `lr=5e-5`) | +0.1317 ± 0.1336 | -1619.5 ± 2003.1 | +0.0057 ± 0.0106 | -0.0147 ± 0.0153 |

所見:
- 3 条件とも平均では悪化方向。
- 主評価優先順では **A が最良**。
- B は 4 指標すべてで悪化幅が大きく、`epochs=2` は不採用。
- C は `Δwin_rate` だけ A と同等だが、`Δavg_rank` と `Δavg_score` で A を更新できない。

## 4. 副評価

after 指標（mean ± std, seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| REF | 3.4567 ± 0.0912 | -13659.2 ± 1388.5 | 0.03958 ± 0.01149 | 0.57366 ± 0.01560 |
| A | 3.4417 ± 0.0888 | -14073.8 ± 1250.0 | 0.04126 ± 0.00847 | 0.57852 ± 0.01729 |
| B | 3.5617 ± 0.0895 | -15208.8 ± 1649.3 | 0.02850 ± 0.01430 | 0.59248 ± 0.02233 |
| C | 3.5050 ± 0.0816 | -14525.5 ± 1337.5 | 0.04177 ± 0.01021 | 0.58097 ± 0.01127 |

所見:
- after 指標でも A が総合最良。
- B は after でも明確に悪い。
- C は `win_rate` は A と同程度だが、`avg_rank/avg_score/deal_in_rate` は A に劣る。

## 5. 補助観測

### 5.1 learner 診断統計

mean ± std（seed=5）

| 条件 | clip_fraction | ratio_std | ratio_p99 | value_error_mean |
|---|---:|---:|---:|---:|
| REF | 0.7874 ± 0.0151 | 0.7468 ± 0.0225 | 2.8179 ± 0.1171 | 148.37 ± 13.59 |
| A | 0.7476 ± 0.0952 | 0.7137 ± 0.1199 | 2.7717 ± 0.4568 | 148.37 ± 13.59 |
| B | 0.7828 ± 0.0437 | 0.7715 ± 0.1113 | 3.0619 ± 0.5623 | 148.37 ± 13.59 |
| C | 0.7228 ± 0.0240 | 0.6487 ± 0.0497 | 2.4524 ± 0.3257 | 148.37 ± 13.59 |

所見:
- `value_error_mean` は reuse 条件で共通 self-play を使うため、A/B/C で同一。
- C は `clip_fraction` と `ratio` tail を最も下げており、**更新自体は穏やかになっている**。
- ただし、その穏和化は主評価改善に直結していない。
- B は learner 時間は短いが、`ratio_p99` が最大で、更新品質も悪い。

### 5.2 REF self-play 統計

mean ± std（seed=5）

- `policy_wins`: 83.6 ± 9.9
- `policy_deal_ins`: 56.0 ± 8.7
- `policy_draws`: 1692.8 ± 10.5
- `tsumo_count`: 27.6 ± 4.8
- `ron_count`: 56.0 ± 8.7
- `ryukyoku_count`: 1692.8 ± 10.5
- `num_rounds`: 1776.4 ± 3.3
- `total_steps`: 122366.0 ± 240.3

解釈:
- A/B/C はこの self-play を再利用しているので、条件差は learner 側だけに由来する。

### 5.3 REF reward / round_over reward 分布

`reward`（shard 全体の集計を seed 平均）

- mean: `-8.94 ± 0.82`
- std: `217.56 ± 19.95`
- p50 / p90 / p99: `0 / 0 / 0`

`round_over_reward`（round 終了サンプルのみ）

- mean: `-230.82 ± 41.33`
- std: `1390.57 ± 190.57`
- p50 / p90 / p99: `0 / 0 / 444`

解釈:
- reward は非常に sparse。
- `round_over_reward` は平均負で、分布の tail が強い。
- 「更新強度だけでなく、target の質や reward 構造も怪しい」という仮説を補強する。

### 5.4 時間

1 run あたり平均（秒）

| 条件 | total | imitation | selfplay | eval_before | learner | eval |
|---|---:|---:|---:|---:|---:|---:|
| REF | 864.29 | 119.16 | 44.20 | 344.24 | 17.66 | 339.02 |
| A | 357.63 | - | - | - | 17.90 | 339.72 |
| B | 346.15 | - | - | - | 12.32 | 333.83 |
| C | 360.61 | - | - | - | 16.70 | 343.91 |

所見:
- 実行時間の支配要因は `eval_before + eval`。
- B は learner が短くなるが、品質低下が大きく見合わない。

## 6. 総合結論

1. **baseline（A）を維持**する。今回の 3 条件では A が主評価・副評価とも最良。  
2. **`epochs=2`（B）は不採用**。悪化幅が大きく、更新強度を弱める方向として有効でない。  
3. **`lr=5e-5`（C）は保留**。learner 診断統計は改善するが、主評価では baseline を更新できない。  
4. reward は依然として sparse で、`round_over_reward` の tail も強い。PPO 悪化の主因は「更新強度だけ」ではなく、「更新方向 / target の質」も含むと考えるのが自然。  

## 7. 今回の判断

- 採用: `epochs=4`, `lr=1e-4` を baseline 維持
- 保留: `lr=5e-5` は診断用の対照条件としては有用
- 見送り: `epochs=2`

## 8. 次アクション

1. tie-aware 固定で、`A` と `C` を使った **PPO 診断 runbook** を組む。  
2. 次は learner 診断統計と reward 分布を主役にし、更新強度ではなく **target / reward 設計側の仮説** を検証する。  
3. 必要なら `eval_matches=30` のスクリーニングを維持しつつ、有望条件だけ `50` で再確認する。  

## 9. 実行対応表

run_map はローカル管理なので、比較に必要な対応をここへ転記する。

| seed | role | run_dir | source_run_dir | 備考 |
|---|---|---|---|---|
| 42 | REF | （ローカル run） |  | full run |
| 42 | A | （ローカル run） | （ローカル run） | reuse |
| 42 | B | （ローカル run） | （ローカル run） | reuse |
| 42 | C | （ローカル run） | （ローカル run） | reuse |
| 43 | REF | （ローカル run） |  | full run |
| 43 | A | （ローカル run） | （ローカル run） | reuse |
| 43 | B | （ローカル run） | （ローカル run） | reuse |
| 43 | C | （ローカル run） | （ローカル run） | reuse |
| 44 | REF | （ローカル run） |  | full run |
| 44 | A | （ローカル run） | （ローカル run） | reuse |
| 44 | B | （ローカル run） | （ローカル run） | reuse |
| 44 | C | （ローカル run） | （ローカル run） | reuse |
| 45 | REF | （ローカル run） |  | full run |
| 45 | A | （ローカル run） | （ローカル run） | reuse |
| 45 | B | （ローカル run） | （ローカル run） | reuse |
| 45 | C | （ローカル run） | （ローカル run） | reuse |
| 46 | REF | （ローカル run） |  | full run |
| 46 | A | （ローカル run） | （ローカル run） | reuse |
| 46 | B | （ローカル run） | （ローカル run） | reuse |
| 46 | C | （ローカル run） | （ローカル run） | reuse |
