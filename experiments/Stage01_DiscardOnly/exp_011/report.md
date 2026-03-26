# Experiment Report: exp_011

作成日: 2026-03-07  
対象: `experiments/exp_011/runbook.md`

## 1. 実験概要

目的: 暫定 baseline（`lr=0.0001, epochs=4, value_loss_coef=0.25`）を固定し、`clip_epsilon`（0.1 / 0.2 / 0.3）を再利用比較で評価する。

- seeds: `42,43,44,45,46`
- 実行方式: seed ごとに `ref` を 1 本作成し、`imitation,selfplay,eval_before` を再利用して A/B/C を分岐
- 評価: `rotation`, `evaluation.num_matches=50`

## 2. 実行結果

- 全 run 成功（`summary.json.success=true`）
- A/B/C の再利用 run で `eval/eval_diff.json` 生成と 4 指標 delta 非nullを確認
- run 対応は `experiments/exp_011/run_map.json` を正とする

## 3. 主評価（eval_before -> eval の delta）

mean ± std（seed=5）

| 条件 | clip_epsilon | Δavg_rank | Δavg_score | Δdeal_in_rate | Δwin_rate |
|---|---:|---:|---:|---:|---:|
| A | 0.1 | +0.0460 ± 0.0468 | -598.4 ± 757.4 | +0.00642 ± 0.00916 | -0.01179 ± 0.00345 |
| B | 0.2 | +0.0430 ± 0.0453 | -689.6 ± 725.1 | +0.00563 ± 0.00864 | -0.01466 ± 0.00422 |
| C | 0.3 | +0.0790 ± 0.0443 | -1172.4 ± 568.2 | +0.00755 ± 0.00609 | -0.01922 ± 0.00562 |

所見:
- C（0.3）は全体的に悪化方向が強く、最下位。
- A（0.1）と B（0.2）は僅差。
- 主評価優先順（`Δavg_rank → Δavg_score → Δdeal_in_rate → Δwin_rate`）に沿うと、
  - `Δavg_rank`: B が僅差優位
  - `Δavg_score`: A が優位
  - `Δdeal_in_rate`: B が僅差優位
  で拮抗。

## 4. after 指標（eval 後）

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A (0.1) | 3.4230 ± 0.0443 | -13683.6 ± 739.3 | 0.04571 ± 0.00184 | 0.57493 ± 0.00947 |
| B (0.2) | 3.4200 ± 0.0677 | -13774.8 ± 987.9 | 0.04283 ± 0.00602 | 0.57414 ± 0.01355 |
| C (0.3) | 3.4560 ± 0.0475 | -14257.6 ± 793.0 | 0.03828 ± 0.00590 | 0.57607 ± 0.01614 |

所見:
- after でも C は明確に不利。
- A/B はほぼ同等で、`avg_score` と `win_rate` は A、`avg_rank` と `deal_in_rate` は B が僅差で良い。

## 5. 時間（再利用 run）

1 run あたり平均（sec）

| 条件 | total | learner | eval |
|---|---:|---:|---:|
| A (0.1) | 282.85 | 17.27 | 265.58 |
| B (0.2) | 280.52 | 16.95 | 263.57 |
| C (0.3) | 280.65 | 17.95 | 262.71 |

## 6. 結論

1. `clip_epsilon=0.3` は不採用（delta/after ともに悪化）。
2. `0.1` と `0.2` は僅差で明確な一方勝ちは出ず。
3. 運用判断としては、
   - 安定維持を優先: `0.2` 継続
   - 攻めるなら: `0.1` を暫定採用して追加確認
   のどちらでも妥当。

## 7. 次アクション

1. `0.1` vs `0.2` のみを対象に、同一再利用方式で再確認（seed拡張または eval_match 増加）。
2. 次ノブ（runbook案どおり `batch_size`）へ進む場合は、baseline は一旦 `0.2` 固定で進める。
