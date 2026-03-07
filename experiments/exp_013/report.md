# Experiment Report: exp_013

作成日: 2026-03-07  
対象: `experiments/exp_013/runbook.md`

## 1. 実験概要

目的: 暫定 baseline（`lr=0.0001, epochs=4, value_loss_coef=0.25, clip_epsilon=0.2, batch_size=256`）を固定し、`training.gae_lambda` を比較する。

- A: `gae_lambda=0.90`
- B: `gae_lambda=0.95`（baseline）
- C: `gae_lambda=0.98`

実行方式:
- seeds `42..46`
- seed ごとに ref run を作成
- `imitation,selfplay,eval_before` を再利用して A/B/C を分岐
- 評価は `rotation`, `num_matches=50`

## 2. 実行結果

- 全 run 成功（`summary.json.success=true`）
- A/B/C 再利用 run で `eval/eval_diff.json` を確認
- 4 指標 delta（`avg_rank/avg_score/win_rate/deal_in_rate`）は全て非null
- 対応表は `experiments/exp_013/run_map.json` を正とする

## 3. 主評価（eval_before -> eval の delta）

mean ± std（seed=5）

| 条件 | Δavg_rank | Δavg_score | Δdeal_in_rate | Δwin_rate |
|---|---:|---:|---:|---:|
| A (0.90) | +0.0550 ± 0.0798 | -549.5 ± 805.1 | +0.00659 ± 0.00459 | -0.01352 ± 0.00726 |
| B (0.95) | +0.0430 ± 0.0453 | -689.6 ± 725.1 | +0.00563 ± 0.00864 | -0.01466 ± 0.00422 |
| C (0.98) | +0.0910 ± 0.0808 | -1063.2 ± 731.4 | +0.00932 ± 0.00814 | -0.01958 ± 0.00587 |

所見:
- C（0.98）は悪化方向が最も強く、不採用。
- A（0.90）と B（0.95）は僅差。
- 主評価優先順に従うと、
  - `Δavg_rank`: B が優位
  - `Δavg_score`: A が優位
  - `Δdeal_in_rate`: B が優位
  で、総合は **B 優勢**。

## 4. after 指標（eval 後）

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A (0.90) | 3.4320 ± 0.0797 | -13634.7 ± 973.5 | 0.04398 ± 0.00869 | 0.57510 ± 0.01546 |
| B (0.95) | 3.4200 ± 0.0677 | -13774.8 ± 987.9 | 0.04283 ± 0.00602 | 0.57414 ± 0.01355 |
| C (0.98) | 3.4680 ± 0.0789 | -14148.4 ± 791.8 | 0.03792 ± 0.00487 | 0.57783 ± 0.01113 |

所見:
- C は after 指標でも不利。
- A/B はほぼ拮抗だが、`avg_rank` と `deal_in_rate` は B、`avg_score` と `win_rate` は A が良い。

## 5. 時間（再利用 run）

1 run あたり平均（sec）

| 条件 | total | learner | eval |
|---|---:|---:|---:|
| A (0.90) | 280.49 | 16.46 | 264.03 |
| B (0.95) | 281.53 | 16.30 | 265.23 |
| C (0.98) | 283.10 | 17.58 | 265.51 |

時間差は小さく、今回の採否は品質指標優先で判断可能。

## 6. 結論

1. **`gae_lambda=0.95` 維持採用**（主評価順で総合優勢）。
2. `0.98` は不採用。
3. `0.90` は一部指標で良さがあるが、主評価順では baseline を更新する根拠は不足。

## 7. 次アクション

1. baseline は `gae_lambda=0.95` を維持。  
2. 次ノブとして `gamma` 比較に進む（`gae_lambda` は固定）。  
3. 併せて、`A(0.90)` で良かった指標（`avg_score/win_rate`）は今後の副次観測として追跡する。
