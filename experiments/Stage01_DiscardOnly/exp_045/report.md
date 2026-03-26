# Experiment Report: exp_045

作成日: 2026-03-15  
対象: `experiments/exp_045/runbook.md`  
目的: 最良近傍条件で `100 cycle` を実行し、PPO が長期で回復するか/壊れ続けるかを確認

## 1. 実験概要

- 条件: A (`long_cycle_100_seed42`)
- seed: `42`（単 seed）
- multi-cycle: `num_cycles=100`, `selfplay_matches_per_cycle=200`, `eval_each_cycle=true`
- ベース条件: `exp_044 B` 近傍
  - `turn_context=true`
  - `value_loss=mse`
  - `advantage clipなし`

## 2. 実行結果

| 項目 | 結果 |
|---|---|
| batch_dir | （ローカル run） |
| success/failure | `1 / 0` |
| run success | `true` |
| cycles | `100` |
| checkpoint_cycle_*.pt | `100 files (00..99)` |

成功判定（runbook §4）は満たした。

## 3. 最終サイクル結果（cycle 99）

- `eval_before.avg_rank = 3.6417`
- `eval.avg_rank = 3.6583`（`delta=+0.0167`）
- `eval_before.avg_score = -16863.33`
- `eval.avg_score = -16707.50`（`delta=+155.83`）
- `win_rate = 0.00346`
- `deal_in_rate = 0.57719`

最終サイクル単体では、学習直前→学習後で `avg_score` は微改善だが、`avg_rank` は悪化。

## 4. 時系列サマリ（runbook指定の3区間）

| 区間 | avg_rank | avg_score | win_rate | deal_in_rate | mean(eval_diff Δrank) |
|---|---:|---:|---:|---:|---:|
| cycle 1-20 | 3.6108 | -16322.67 | 0.01816 | 0.58534 | +0.01500 |
| cycle 21-60 | 3.6613 | -17331.83 | 0.00476 | 0.58146 | +0.00083 |
| cycle 61-100 | 3.6635 | -17236.58 | 0.00192 | 0.57840 | -0.00062 |

所見:
- **前半(1-20)で大きく悪化**し、以後は悪化速度が鈍化。
- 中盤以降は `eval_diff Δrank` の平均がほぼ0付近になり、1サイクル内の壊れ方は弱まる。
- ただし、**性能レベルは初期に戻らない**。

## 5. 代表点（cycle別）

- cycle 0: `avg_rank=3.3917`, `avg_score=-13436.67`（このrun内の最良）
- cycle 10: `avg_rank=3.6333`, `avg_score=-17120.83`
- cycle 20: `avg_rank=3.6500`, `avg_score=-17243.33`
- cycle 77: `avg_score=-18401.67`（最悪スコア）
- cycle 88: `avg_rank=3.7167`（最悪ランク）
- cycle 99: `avg_rank=3.6583`, `avg_score=-16707.50`

初期(cycle 0)→最終(cycle 99)差分:
- `avg_rank: +0.2667`（悪化）
- `avg_score: -3270.83`（悪化）
- `win_rate: -0.04899`（大幅低下）
- `deal_in_rate: -0.00565`（わずか改善）

## 6. Learner診断の推移（要点）

| 区間 | clip_fraction | ratio_std | advantage_abs_mean_before_clip |
|---|---:|---:|---:|
| cycle 1-20 | 0.0963 | 0.0931 | 0.3947 |
| cycle 21-60 | 0.0705 | 0.0824 | 0.3596 |
| cycle 61-100 | 0.0958 | 0.0896 | 0.3741 |

所見:
- 中盤で一度 `clip_fraction` / `ratio_std` が下がる（更新が穏やかになる）。
- 後半で再びやや上がるが、前半ピークほどではない。
- つまり「更新不安定で破綻し続ける」というより、**低い性能帯で安定化している**挙動に近い。

## 7. 解釈（結論）

この条件では、100 cycle まで回しても:
- 前半で急落
- その後は横ばい〜微回復
- ただし初期水準には戻らない

したがって、今回の問いに対する答えは:
- 「強化学習を続けるほど単調に壊れ続ける」ではない
- しかし「初期悪化後に明確に回復して伸びる」も確認できない
- **悪化後に低性能の定常点へ収束する傾向**が強い

## 8. 次アクション

1. 次の実験は、すでに検討中の `policy anchor (BC/KL)` 導入を優先（長期ドリフト抑制の直接対策）。
2. multi-cycle 実験は 100 cycle を毎回回すより、まず 20-30 cycle の複数 seed で再現性確認を先に行う。
3. 比較しやすくするため、`cycles` から `best_so_far` / `from_cycle0_delta` を自動算出する集約を将来的に追加検討。
