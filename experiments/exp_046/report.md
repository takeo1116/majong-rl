# Experiment Report: exp_046

作成日: 2026-03-15  
対象: [experiments/exp_046/runbook.md](/home/takeo1116/Git/majong-rl/experiments/exp_046/runbook.md)  
目的: `policy_anchor (KL)` で PPO 後の悪化幅が縮むかを 20 seed で確認

## 1. 実験概要

条件（20 seeds, 42..61）:
- A: `policy_anchor_kl`
  - `training.policy_anchor.enabled=true`
  - `training.policy_anchor.type=kl`
  - `training.policy_anchor.coef=0.1`
  - `training.policy_anchor.reference=imitation_fixed`

比較参照:
- `exp_044 B`（turn_context on, anchorなし）
- 参考として `exp_044 E`

## 2. 実行結果

| 項目 | 結果 |
|---|---|
| batch_dir | `runs/20260315_stage1_full_flat_mlp_imitation_then_ppo_batch_5674ad5c` |
| success/failure | `20 / 0` |
| driver | `completed=1, failed=0` |

全 seed 完走。

## 3. 主評価（after）

mean ± std（seed=20）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| exp_046 A (anchor KL) | **3.3596 ± 0.0745** | **-12637.8 ± 939.3** | **0.05315 ± 0.00670** | **0.57174 ± 0.01236** |
| exp_044 B (anchorなし) | 3.3708 ± 0.0772 | -12770.6 ± 908.3 | 0.05214 ± 0.00672 | 0.57239 ± 0.01254 |
| exp_044 E (参考) | 3.3729 ± 0.0912 | -12710.0 | 0.05218 | 0.57113 |

所見:
- `exp_046 A` は `exp_044 B` 比で主要4指標すべて改善（改善幅は小〜中）。
- `avg_rank` は `-0.0113` 改善、`avg_score` は `+132.8` 改善。

## 4. eval_before → eval の悪化幅

`delta = eval.after - eval.before`（avg_rank は小さいほど良い）

| 条件 | Δavg_rank mean ± std | Δavg_score mean ± std |
|---|---:|---:|
| exp_046 A (anchor KL) | **+0.00208 ± 0.05596** | **-159.88 ± 767.27** |
| exp_044 B (anchorなし) | +0.01333 ± 0.05697 | -292.71 ± 675.26 |
| exp_044 E (参考) | +0.01542 ± 0.05696 | -232.12 ± 784.72 |

所見:
- runbook の主目的だった「PPO後悪化幅の縮小」は達成。
- 特に `Δavg_rank` が `+0.0133 -> +0.0021` まで縮小し、ほぼ横ばいに近づいた。

## 5. policy_anchor 診断

20 seed 全 run で `learner_diag.policy_anchor` が記録された。

| 指標 | mean | std |
|---|---:|---:|
| anchor_kl_mean | 0.004447 | 0.000264 |
| anchor_loss_mean | 0.004447 | 0.000264 |

補足:
- `type=kl` なので `anchor_kl_mean` と `anchor_loss_mean` は一致。
- seed 間分散は小さく、アンカー損失は安定して効いている。

## 6. Learner 安定性指標（参考）

| 条件 | clip_fraction | ratio_std | value_error_mean |
|---|---:|---:|---:|
| exp_046 A | 0.0977 ± 0.0077 | 0.0953 ± 0.0030 | -0.002249 ± 0.001285 |
| exp_044 B | 0.1203 ± 0.0256 | 0.1039 ± 0.0092 | -0.002253 ± 0.001280 |

所見:
- `clip_fraction` と `ratio_std` は `exp_046 A` の方が低く、更新がやや安定側。
- `value_error_mean` はほぼ同等。

## 7. 実行時間

`phase_timing.total.mean`:
- `exp_046 A`: `70.09s/seed`
- `exp_044 B`: `76.72s/seed`

条件差の割に速くなっているが、主因は実行時負荷や揺らぎの可能性があるため、速度改善の結論には使わない。

## 8. 結論

- `policy_anchor (KL, coef=0.1)` は、今回の設定で **有効**。
- after 性能は `exp_044 B` より改善し、かつ runbook 主目的の `eval_before->eval` 悪化幅も明確に縮小。
- 次段としては、
  1. `coef` 探索（例: 0.03 / 0.1 / 0.3）
  2. `KL vs BC` 比較
  を優先する価値が高い。
