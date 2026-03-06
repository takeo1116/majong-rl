# Experiment Report: exp_009

作成日: 2026-03-07  
対象: `experiments/exp_009/runbook.md`  
目的: `training.lr=0.0001, training.epochs=4` 固定で `training.value_loss_coef` を比較

## 1. 実行条件

- 共通 config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44`
- 固定主要値:
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
  - `training.epochs=4`
  - `training.batch_size=256`
  - `training.gamma=0.99`
  - `training.gae_lambda=0.95`
  - `training.entropy_coef=0.01`
  - `training.clip_epsilon=0.2`
  - `training.device=cuda`
  - `selfplay.inference_device=cpu`
  - `evaluation.inference_device=cpu`

比較条件:
- A: `training.value_loss_coef=0.25`
- B: `training.value_loss_coef=0.5`
- C: `training.value_loss_coef=1.0`

## 2. batch 実行結果

| 条件 | value_loss_coef | batch_dir | success |
|---|---:|---|---:|
| A | 0.25 | `runs/20260307_stage1_full_flat_mlp_imitation_then_ppo_batch_213c6ffa` | 3/3 |
| B | 0.5  | `runs/20260307_stage1_full_flat_mlp_imitation_then_ppo_batch_9a96c16c` | 3/3 |
| C | 1.0  | `runs/20260307_stage1_full_flat_mlp_imitation_then_ppo_batch_c258f215` | 3/3 |

全条件で `aggregate.eval_mode=rotation` を確認。

## 3. 主評価（eval_before -> eval の delta）

mean ± std（seed=3）

| 条件 | Δavg_rank | Δavg_score | Δdeal_in_rate | Δwin_rate |
|---|---:|---:|---:|---:|
| A (0.25) | -0.0167 ± 0.0425 | +29.6 ± 519.0 | -0.0084 ± 0.0155 | -0.0007 ± 0.0039 |
| B (0.5)  | -0.0208 ± 0.0695 | +38.8 ± 1114.2 | -0.0087 ± 0.0064 | -0.0038 ± 0.0049 |
| C (1.0)  | +0.0167 ± 0.0524 | -583.8 ± 635.5 | -0.0024 ± 0.0097 | -0.0146 ± 0.0052 |

所見:
- C は 4 指標中 3 指標で明確に悪化方向。
- A と B は近いが、runbook の優先順（`Δavg_rank`→`Δavg_score`→`Δdeal_in_rate`）では B が僅差で優位。
- `Δwin_rate` だけは A が良い。

## 4. 最終到達点（after eval）

mean ± std（seed=3）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A (0.25) | 3.4042 ± 0.0520 | -13649.2 ± 614.1 | 0.0498 ± 0.0113 | 0.5661 ± 0.0061 |
| B (0.5)  | 3.4000 ± 0.0760 | -13640.0 ± 1260.1 | 0.0468 ± 0.0093 | 0.5657 ± 0.0147 |
| C (1.0)  | 3.4375 ± 0.1068 | -14262.5 ± 706.0 | 0.0359 ± 0.0081 | 0.5721 ± 0.0153 |

所見:
- after 指標も C が最下位。
- A/B はほぼ同等。avg_rank/avg_score/deal_in_rate は B が僅差で良い。

## 5. self-play 統計（phase_stats.selfplay）

3条件とも同一（mean ± std）:

- `policy_wins`: 29.667 ± 4.497
- `policy_deal_ins`: 22.333 ± 3.091
- `policy_draws`: 1761.000 ± 4.320
- `tsumo_count`: 7.333 ± 2.625
- `ron_count`: 22.333 ± 3.091
- `ryukyoku_count`: 1761.000 ± 4.320
- `num_rounds`: 1790.667 ± 1.700

解釈:
- これは期待通りで、`value_loss_coef` は learner フェーズのノブなので self-play 統計は不変。

## 6. フェーズ時間（1 run 平均, 秒）

| 条件 | imitation | selfplay | eval_before | learner | eval | total |
|---|---:|---:|---:|---:|---:|---:|
| A (0.25) | 45.75 | 18.54 | 114.38 | 15.09 | 112.28 | 306.04 |
| B (0.5)  | 44.99 | 19.48 | 117.44 | 15.22 | 114.34 | 311.46 |
| C (1.0)  | 43.90 | 17.51 | 113.72 | 15.68 | 111.59 | 302.41 |

## 7. 結論

1. **`value_loss_coef=1.0` は不採用**（delta/after とも悪化傾向が明確）。
2. **`0.25` と `0.5` は僅差**。今回の主評価優先順では `0.5` がわずかに優位。
3. ただし seed=3, eval=20 のため分散が大きく、**最終判断は未確定**。

推奨次アクション:
1. `0.25` と `0.5` のみを対象に、`5 seeds + evaluation.num_matches=50` で再確認。
2. 次回 runbook では phase 再利用を使い、`imitation/selfplay/eval_before` 固定で learner 比較を高速化。
