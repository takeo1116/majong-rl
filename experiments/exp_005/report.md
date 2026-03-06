# experiments/exp_005/report.md（Runbook 5 Report）

作成日: 2026-03-06  
対象 Runbook: `experiments/exp_005/runbook.md`  
目的: **E（軽量 imitation→PPO）起点で、PPO 段の悪化を抑える設定探索**

---

## 1. 実験概要

- ベース config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44`
- 評価: `rotation` (`evaluation.rotation_seats=[0,1,2,3]`)
- 評価試行数: `evaluation.num_matches=20`
- 比較条件（4条件）:
  - A: baseline
  - B: `training.lr=0.0002`
  - C: `training.lr=0.0001`
  - D: `training.clip_epsilon=0.1`

実行 batch_dir:
- A: `runs/20260306_stage1_full_flat_mlp_imitation_then_ppo_batch_37703f02`
- B: `runs/20260306_stage1_full_flat_mlp_imitation_then_ppo_batch_ea4b4789`
- C: `runs/20260306_stage1_full_flat_mlp_imitation_then_ppo_batch_a4992bbb`
- D: `runs/20260306_stage1_full_flat_mlp_imitation_then_ppo_batch_1356d5e3`

全条件で `success_count=3/3`、`aggregate.eval_mode=rotation` を確認。

---

## 2. 集計結果（after 指標）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A baseline | 3.5250 | -16012.9167 | 0.020347 | 0.593946 |
| B lr=0.0002 | 3.4458 | -14311.6667 | 0.035041 | 0.574047 |
| C lr=0.0001 | **3.4000** | **-13640.0000** | **0.046783** | **0.565742** |
| D clip=0.1 | 3.5583 | -16206.6667 | 0.021261 | 0.584847 |

所見:
- after の最終指標は **条件Cが最良**。

---

## 3. 集計結果（eval_before -> eval delta）

主評価対象（平均 delta）:

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| A baseline | +0.058333 | -1540.000000 | -0.018695 | +0.013595 |
| B lr=0.0002 | +0.033333 | -812.916667 | -0.012296 | -0.005649 |
| C lr=0.0001 | **-0.020833** | **+38.750000** | -0.003769 | **-0.008706** |
| D clip=0.1 | +0.091667 | -1733.750000 | -0.017781 | +0.004496 |

所見:
- 目的だった「PPO 段で壊さない」に最も近いのは **条件C**。
- A/B/D は `Δavg_rank > 0` かつ `Δavg_score < 0` で悪化傾向。
- D（clip縮小のみ）は今回の範囲では改善せず、むしろ悪化が大きい。

---

## 4. 条件別メモ

- A baseline:
  - E設定そのままでは、PPO後に悪化傾向が残る。
- B lr=0.0002:
  - baseline より悪化幅は縮小したが、まだ平均では悪化側。
- C lr=0.0001:
  - avg_rank/avg_score/deal_in_rate で最も安定して改善方向。
- D clip=0.1:
  - 今回は改善せず、更新抑制としては有効でなかった可能性。

---

## 5. 結論

Runbook 5 の小規模 sweep（3 seeds, eval=20）では、  
**`training.lr=0.0001`（条件C）が最有力**。

理由:
- `eval_before -> eval` の平均 delta が 4 条件中で最良
- after 指標も 4 条件中で最良

---

## 6. 次アクション（Runbook 5 次段）

1. 条件Cのみを `seeds=42..46`（5 seeds）へ拡張  
2. 同条件で `evaluation.num_matches=50` に増やして再検証  
3. 同時に `summary.json.phase_stats.selfplay`（`policy_wins`, `policy_deal_ins` など）を確認し、悪化要因分解を行う

---

## 付録: delta 生値（参考）

- A `Δavg_rank`: `[+0.0125, +0.0500, +0.1125]`
- B `Δavg_rank`: `[+0.0125, +0.0875, +0.0000]`
- C `Δavg_rank`: `[-0.0875, -0.0500, +0.0750]`
- D `Δavg_rank`: `[+0.0125, +0.2250, +0.0375]`

