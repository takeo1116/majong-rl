# Experiment Report: exp_019

作成日: 2026-03-08  
対象: `experiments/exp_019/runbook.md`

## 1. 実験概要

- 目的: `shanten_hint=true` / `imitation_loss_mode=tie_aware_best_set` を固定し、baseline と weak-lr を比較して、PPO 悪化の主因が更新強度ではなく reward / target / advantage 品質側にあるかを切り分ける。
- 実行方式:
  - seed ごとに REF（full run）を 1 本作成
  - `imitation,selfplay,eval_before` を再利用して A/B を分岐
  - `eval_before -> eval` と `ppo_diag` を併読
- seeds: `42,43,44,45,46`
- 比較条件:
  - REF: `epochs=4`, `lr=1e-4`（参照 run）
  - A: `epochs=4`, `lr=1e-4`（baseline reuse）
  - B: `epochs=4`, `lr=5e-5`（weak-lr reuse）
- 主評価の優先順:
  1. `Δavg_rank`
  2. `Δavg_score`
  3. `Δdeal_in_rate`
  4. `Δwin_rate`

## 2. 実行結果

| 条件 | run/batch | success |
|---|---|---:|
| REF | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_*` x5 | 5/5 |
| A | reuse run x5 | 5/5 |
| B | reuse run x5 | 5/5 |

注記:
- 比較の正本は `experiments/exp_019/run_map.json`。
- A/B は同一 seed の REF から `imitation,selfplay,eval_before` を再利用しているため、reward 分布・self-play 分布・`old_value/return/value_error` は条件間で共通。

## 3. 主評価

mean ± std（seed=5）

| 条件 | Δavg_rank | Δavg_score | Δdeal_in_rate | Δwin_rate |
|---|---:|---:|---:|---:|
| A (`lr=1e-4`) | +0.0683 ± 0.1076 | -1167.8 ± 1495.5 | +0.0033 ± 0.0139 | -0.0152 ± 0.0116 |
| B (`lr=5e-5`) | +0.1317 ± 0.1336 | -1619.5 ± 2003.1 | +0.0057 ± 0.0106 | -0.0147 ± 0.0153 |

所見:
- 両条件とも平均では悪化方向。
- 主評価優先順では **A が優位**。
- B は `Δwin_rate` だけ A とほぼ同等だが、`Δavg_rank` と `Δavg_score` で劣る。
- `exp_018` と同じ結論で、weak-lr は baseline を更新できなかった。

## 4. 副評価

after 指標（mean ± std, seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A | 3.4417 ± 0.0888 | -14073.8 ± 1250.0 | 0.04126 ± 0.00847 | 0.57852 ± 0.01729 |
| B | 3.5050 ± 0.0816 | -14525.5 ± 1337.5 | 0.04177 ± 0.01021 | 0.58097 ± 0.01127 |

所見:
- after 指標でも A が総合最良。
- B は `win_rate` の平均だけ僅差で上だが、`avg_rank/avg_score/deal_in_rate` はすべて悪い。
- したがって「weak-lr にすると PPO 後の方策が安定する」とは言えない。

## 5. 補助観測

### 5.1 learner 診断統計

mean ± std（seed=5）

| 条件 | clip_fraction | ratio_std | ratio_p99 | new_value_mean |
|---|---:|---:|---:|---:|
| A | 0.7476 ± 0.0952 | 0.7137 ± 0.1199 | 2.7717 ± 0.4568 | -122.63 ± 11.38 |
| B | 0.7228 ± 0.0240 | 0.6487 ± 0.0497 | 2.4524 ± 0.3257 | -104.18 ± 9.47 |

共通（A/B で同一, reuse により self-play 共通）:

- `advantage_positive_ratio`: `0.8475 ± 0.0052`
- `value_error_mean`: `148.37 ± 13.59`
- `return_mean`: `-148.99` 近辺ではなく seed ごとに変動するが条件間では同一
- `old_value_mean`: seed ごとに変動するが条件間では同一

所見:
- B は `clip_fraction` / `ratio_std` / `ratio_p99` を下げており、**更新自体は穏やか**。
- それでも主評価は改善しない。
- 一方、`advantage / return / old_value / value_error` は条件間で同一なので、今回の比較だけで target 品質問題を直接証明はできない。
- ただし少なくとも、「ratio/clip を少し穏やかにするだけでは直らない」ことは確認できた。

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
- A/B はこの self-play を共通に再利用しているので、今回の差は learner 更新に由来する。

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
- reward は依然として極端に sparse。
- round 終了サンプルでも大半は 0 で、tail が強い。
- この分布は「PPO が日常的に細かい改善信号を受け取れていない」仮説と整合する。

### 5.4 時間

1 run あたり平均（秒）

| 条件 | total | learner | eval |
|---|---:|---:|---:|
| REF | 864.29 ± 27.04 | 17.66 ± 1.67 | 339.02 ± 10.88 |
| A | 357.63 ± 16.51 | 17.90 ± 1.81 | 339.72 ± 15.64 |
| B | 360.61 ± 8.24 | 16.70 ± 1.74 | 343.91 ± 7.17 |

所見:
- B は learner が少し軽いが、品質改善には結びつかない。
- コスト差は小さいので、今回の採否は品質だけで判断してよい。

## 6. 総合結論

1. **baseline（A）を維持**する。weak-lr（B）は今回も主評価を更新できなかった。  
2. **`clip_fraction` や `ratio` tail を少し下げるだけでは PPO 悪化は解消しない**。  
3. 今回の結果は、PPO 悪化の主因が「更新強度単独」ではなく、**reward / target / advantage 品質側にもある** という仮説を補強する。  
4. ただし A/B は self-play を共通再利用しているため、今回だけで value target 問題を直接断定はしない。次段では reward / target 側に介入する診断または実装変更が必要。  

## 7. 今回の判断

- 採用: `lr=1e-4` baseline 維持
- 保留: weak-lr は learner 診断の対照条件としては有用
- 見送り: 「更新を弱めれば PPO 悪化が自然に消える」仮説

## 8. 次アクション

1. reward / target / value 側を主題にした CQ または診断実験へ進む。  
2. `clip_fraction` / `ratio` より、reward sparsity と `value_error` の扱いを優先して切り分ける。  
3. 必要なら reward 設計変更前に、現状成果物だけで読める範囲をもう一段整理する。  

## 9. 実行対応表

run_map はローカル管理なので、比較に必要な対応をここへ転記する。

| seed | role | run_dir | source_run_dir | 備考 |
|---|---|---|---|---|
| 42 | REF | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_0cd82611` |  | full run |
| 42 | A | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_6fb7a97a` | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_0cd82611` | reuse |
| 42 | B | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_f1d0702e` | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_0cd82611` | reuse |
| 43 | REF | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_79f7154e` |  | full run |
| 43 | A | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_2dc9e4bf` | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_79f7154e` | reuse |
| 43 | B | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_83e5dd70` | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_79f7154e` | reuse |
| 44 | REF | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_b47af4f1` |  | full run |
| 44 | A | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_c475dad6` | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_b47af4f1` | reuse |
| 44 | B | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_5c188216` | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_b47af4f1` | reuse |
| 45 | REF | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_42640777` |  | full run |
| 45 | A | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_e0dbd2da` | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_42640777` | reuse |
| 45 | B | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_54434f21` | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_42640777` | reuse |
| 46 | REF | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_3fabd2dc` |  | full run |
| 46 | A | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_80ee06d1` | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_3fabd2dc` | reuse |
| 46 | B | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_2dc71e6e` | `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_3fabd2dc` | reuse |
