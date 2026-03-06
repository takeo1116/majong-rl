# experiments/exp_008/report.md（Runbook 8 Report）

作成日: 2026-03-06  
対象 Runbook: `experiments/exp_008/runbook.md`  
目的: `training.lr=0.0001` 固定時の `training.epochs=2` vs `training.epochs=4` を 5 seeds / rotation eval 50 で決着する

---

## 1. 実験条件

- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
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
  - `evaluation.num_matches=50`
  - `evaluation.num_workers=10`
  - `training.lr=0.0001`
  - `training.device=cuda`
  - `selfplay.inference_device=cpu`
  - `evaluation.inference_device=cpu`

条件別 batch_dir:
- A (`training.epochs=2`): `runs/20260306_stage1_full_flat_mlp_imitation_then_ppo_batch_19478a92`
- B (`training.epochs=4`): `runs/20260306_stage1_full_flat_mlp_imitation_then_ppo_batch_0f1f5ec4`

成功率:
- A: `5/5`
- B: `5/5`

---

## 2. 結果比較

### 2.1 after 指標（aggregate mean ± std）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A (epochs=2) | 3.4340 ± 0.0627 | -13735.9 ± 460.5 | 0.045537 ± 0.004786 | 0.577521 ± 0.008138 |
| B (epochs=4) | 3.4010 ± 0.0457 | -13382.4 ± 636.8 | 0.048883 ± 0.006199 | 0.570317 ± 0.002372 |

所見:
- after 指標は **B が全項目で A を上回る**。

### 2.2 eval_before -> eval 差分（runs[].eval_diff の平均）

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| A (epochs=2) | +0.0570 | -650.7 | -0.0119588 | +0.0090062 |
| B (epochs=4) | +0.0240 | -297.2 | -0.0086132 | +0.0018028 |

所見:
- 主評価の delta でも **B が A より明確に良い**。
- どちらも平均では悪化方向だが、悪化幅は B が小さい。

---

## 3. self-play 統計（参考）

両条件で self-play 設定が同一のため、`summary.json.phase_stats.selfplay` の平均値はほぼ同一。

- `policy_wins`: 30.2
- `policy_deal_ins`: 22.4
- `policy_draws`: 1758.8
- `tsumo_count`: 7.8
- `ron_count`: 22.4
- `ryukyoku_count`: 1758.8
- `num_rounds`: 1789.0

解釈:
- 今回の A/B 差は self-play 分布差というより、PPO 更新量（epochs 差）に由来すると見るのが自然。

---

## 4. 所要時間（1 seed 平均, 秒）

| 条件 | imitation | selfplay | eval_before | learner | eval | total |
|---|---:|---:|---:|---:|---:|---:|
| A (epochs=2) | 44.82 | 17.44 | 262.45 | 10.39 | 260.22 | 595.31 |
| B (epochs=4) | 46.66 | 18.64 | 271.73 | 16.40 | 270.82 | 624.24 |

所見:
- B は learner 時間が増えるが、総時間差は約 +29 秒/seed（+4.9%）。
- 指標改善幅を考えると許容可能なコスト増。

---

## 5. 結論

Runbook 8 の結論:

1. `lr=0.0001` 固定では、**標準候補は `training.epochs=4`**。  
2. `epochs=2` は「より保守的」だったが、今回の 5 seed / eval50 では保守性の優位は確認されず、性能・deltaともに B に劣後。  
3. 次段探索は `lr=0.0001, epochs=4` を baseline として進めるのが妥当。

---

## 6. Runbook 7 との整合

- Runbook 7（3 seeds / eval20）では A/B がほぼ同等に見えた。  
- Runbook 8（5 seeds / eval50）で差が拡大し、**B 優位**が確認された。  

解釈:
- Runbook 7 は探索として有効だったが、A/B の最終判定には分散が大きかった。
- Runbook 8 はその判定を確定させる役割を果たした。
