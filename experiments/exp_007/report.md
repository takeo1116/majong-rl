# experiments/exp_007/report.md（Runbook 7 Report）

作成日: 2026-03-06  
対象 Runbook: `experiments/exp_007/runbook.md`  
目的: `training.lr=0.0001` が「良い更新」か「壊れるのが遅いだけ」かを、epochs 軸で切り分ける

---

## 1. 実験条件

- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44`
- 共通固定:
  - `imitation.num_workers=10`
  - `selfplay.imitation_matches=25`
  - `training.imitation_epochs=4`
  - `selfplay.num_matches=200`
  - `selfplay.num_workers=10`
  - `selfplay.policy_ratio=1.0`
  - `selfplay.save_baseline_actions=false`
  - `evaluation.mode=rotation`
  - `evaluation.rotation_seats=[0,1,2,3]`
  - `evaluation.num_matches=20`
  - `evaluation.num_workers=10`
  - `training.lr=0.0001`
  - `training.device=cuda`
  - `selfplay.inference_device=cpu`
  - `evaluation.inference_device=cpu`

条件別 batch_dir:
- A (`training.epochs=2`): `runs/20260306_stage1_full_flat_mlp_imitation_then_ppo_batch_1725e4e6`
- B (`training.epochs=4`): `runs/20260306_stage1_full_flat_mlp_imitation_then_ppo_batch_9ae344fd`
- C (`training.epochs=8`): `runs/20260306_stage1_full_flat_mlp_imitation_then_ppo_batch_fa89f533`
- D (`training.epochs=4`, `training.entropy_coef=0.005`): `runs/20260306_stage1_full_flat_mlp_imitation_then_ppo_batch_80335d72`

全条件: `success_count=3/3`

---

## 2. 結果サマリ（after 指標）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A (epochs=2) | 3.4000 ± 0.0750 | -13573.3 ± 431.9 | 0.045295 ± 0.011374 | 0.571812 ± 0.003926 |
| B (epochs=4) | 3.4000 ± 0.0760 | -13640.0 ± 1260.1 | 0.046783 ± 0.009346 | 0.565742 ± 0.014747 |
| C (epochs=8) | 3.4708 ± 0.0191 | -15225.4 ± 80.5 | 0.032429 ± 0.008068 | 0.583515 ± 0.012643 |
| D (epochs=4 + entropy低下) | 3.4292 ± 0.0688 | -14187.1 ± 770.5 | 0.045466 ± 0.015528 | 0.586153 ± 0.019903 |

所見:
- after 指標は **A/B が最良帯**。
- C は全指標で悪化し、明確に過更新寄り。
- D は rank/score は C より良いが、deal-in が最悪。

---

## 3. 主評価（eval_before -> eval の平均差分）

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| A (epochs=2) | -0.020833 | +105.416667 | -0.005257 | -0.002636 |
| B (epochs=4) | -0.020833 | +38.750000 | -0.003769 | -0.008706 |
| C (epochs=8) | +0.050000 | -1546.666667 | -0.018123 | +0.009067 |
| D (epochs=4 + entropy低下) | +0.000000 | -471.250000 | -0.006382 | +0.011938 |

解釈:
- **A/B は delta が改善方向**（少なくとも悪化は抑制）。
- **C は delta が明確に悪化**し、「壊れるのが遅いだけ」仮説を支持。
- **D は entropy を下げるとむしろ悪化**（特に deal-in）。

---

## 4. フェーズ時間（1 seed 平均, 秒）

| 条件 | imitation | selfplay | eval_before | learner | eval | total |
|---|---:|---:|---:|---:|---:|---:|
| A | 44.56 | 17.99 | 114.28 | 10.82 | 110.28 | 297.93 |
| B | 45.37 | 18.51 | 115.34 | 15.33 | 111.21 | 305.75 |
| C | 43.72 | 17.30 | 110.19 | 28.97 | 106.41 | 306.59 |
| D | 42.85 | 16.92 | 110.53 | 13.76 | 106.75 | 290.81 |

所見:
- epochs 増加で learner 時間は増える（B→C）。
- 全体時間は eval が支配的で、条件差は小さい。

---

## 5. 結論

1. `training.lr=0.0001` 固定での最適域は **epochs=2〜4**。  
2. `epochs=8` は性能・deltaともに悪化し、「遅延悪化」挙動が観測された。  
3. `entropy_coef=0.005` は今回条件では有効でなく、優先度を下げてよい。  

Runbook 7 の問いに対する回答:
- low-lr は完全に「遅いだけ」ではなく、**適切な更新量（epochs）では改善方向が出る**。  
- ただし更新量を増やしすぎると、**結局壊れる**。

---

## 6. 注意点（解釈上）

- 今回は `evaluation.num_matches=20`, `seeds=3` の小規模探索。  
- A/B はほぼ同等で、厳密な優劣判定には分散が大きい。  
- 次段で結論固定するなら、候補を A/B に絞って `5 seeds × eval50` で再確認が妥当。
