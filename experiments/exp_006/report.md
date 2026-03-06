# experiments/exp_006/report.md（Runbook 6 Report）

作成日: 2026-03-06  
対象 Runbook: `experiments/exp_006/runbook.md`  
目的: **Runbook 5 最有力条件 C（`training.lr=0.0001`）を 5 seeds / rotation eval 50 で確証取り**

---

## 1. 実験条件

- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- 固定:
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
  - `training.epochs=4`
  - `training.lr=0.0001`
  - `training.device=cuda`
  - `selfplay.inference_device=cpu`
  - `evaluation.inference_device=cpu`

batch_dir:
- `runs/20260306_stage1_full_flat_mlp_imitation_then_ppo_batch_6892dd9e`

成功率:
- `5/5`（100%）
- `aggregate.eval_mode = rotation`

---

## 2. Runbook6 単体結果（5 seeds）

### 2.1 after 指標（aggregate）

- `avg_rank`: **3.4010 ± 0.0457**
- `avg_score`: **-13382.4 ± 636.8**
- `win_rate`: **0.048883 ± 0.006199**
- `deal_in_rate`: **0.570317 ± 0.002372**

### 2.2 eval_before -> eval 差分（runs[].eval_diff 平均）

- `Δavg_rank`: **+0.0240**（悪化寄り）
- `Δavg_score`: **-297.2**（悪化寄り）
- `Δwin_rate`: **-0.008613**（悪化）
- `Δdeal_in_rate`: **+0.001803**（悪化寄り）

### 2.3 所要時間（5 seeds 平均）

- imitation: **44.67s**
- selfplay: **16.73s**
- eval_before: **258.61s**
- learner: **14.42s**
- eval: **258.95s**
- total: **593.38s**（約 9.9 分 / seed）

---

## 3. 比較対象を明示した解釈

比較対象を以下で固定する。

- **主比較**: Runbook5 条件C（同一設定、3 seeds, eval=20）  
  - batch: `runs/20260306_stage1_full_flat_mlp_imitation_then_ppo_batch_a4992bbb`
- **補助比較**: Runbook5 条件A baseline（3 seeds, eval=20）  
  - batch: `runs/20260306_stage1_full_flat_mlp_imitation_then_ppo_batch_37703f02`
- **参考**: Runbook4 E（5 seeds, eval=50）  
  - batch: `runs/20260306_stage1_full_flat_mlp_imitation_then_ppo_batch_3585de77`

### 3.1 Runbook5-C との比較

- Runbook5-C（3 seeds, eval=20）では delta が改善寄り
  - `Δavg_rank = -0.0208`
  - `Δavg_score = +38.75`
- Runbook6（5 seeds, eval=50）では delta が悪化寄り
  - `Δavg_rank = +0.0240`
  - `Δavg_score = -297.2`

解釈:
- 小規模条件で見えた「PPO 段を壊しにくい」傾向は、  
  seed/eval を増やすと維持されなかった。

### 3.2 Runbook5-A baseline との比較

- Runbook5-A の delta はより悪い（`Δavg_rank +0.0583`, `Δavg_score -1540`）ため、  
  **`lr=0.0001` は baseline よりは改善**。
- ただし「悪化しない」水準にはまだ届いていない。

### 3.3 Runbook4-E（参考）との比較

- Runbook4-E（5 seeds, eval=50）より after 指標は改善
  - `avg_rank`: 3.55 → **3.401**
  - `avg_score`: -15612 → **-13382**
  - `deal_in_rate`: 0.587 → **0.570**
- それでも delta は依然マイナス方向（PPO段で一部悪化）を残す。

---

## 4. self-play 統計について

Runbook6 の各 run で、以下は確認可能。

- `summary.json.phase_stats.selfplay`
  - `policy_wins`
  - `policy_deal_ins`
  - `policy_draws`
  - `tsumo_count`
  - `ron_count`
  - `ryukyoku_count`
- 詳細:
  - `selfplay/worker_*/round_results.jsonl`

注記:
- 現行 `batch_summary.json` では self-play 統計の自動集約はされないため、
  run 単位で確認する運用となる。

---

## 5. 結論

Runbook6 の結果より:

1. `training.lr=0.0001` は baseline（Runbook5-A）より明確に良い  
2. ただし「PPO 段が壊さない」水準には未達（delta は平均で悪化寄り）  
3. よって、`lr=0.0001` を暫定基準として次段探索（`epochs=2` など）に進むのが妥当

---

## 6. 次アクション候補

1. 同条件で `training.epochs=2` を追加比較（最優先）  
2. `training.entropy_coef` を低めに振って比較  
3. 必要なら `training.value_loss_coef` を調整  

