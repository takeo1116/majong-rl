# exp_046 runbook

最終更新: 2026-03-15  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: `policy_anchor`（KL）を導入し、20 seedで「PPO後の劣化幅が縮むか」を検証する

---

## 0. 実験の位置づけ

- 参照:
  - `exp_044 B`（turn_contextのみ有効、anchorなし）
  - `exp_045`（長期では初期悪化後に低性能帯へ収束）
- 仮説:
  - imitation固定参照への `policy_anchor` で、PPOの方策ドリフトを抑えられる
  - 結果として `eval_before -> eval` の悪化幅（特に avg_rank）が縮小する

## 1. 共通固定

- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42..61`（20 seeds）
- 共通 override:
  - `feature_encoder.shanten_hint.enabled=true`
  - `feature_encoder.discard_ukeire_hint.enabled=false`
  - `feature_encoder.current_shanten.enabled=true`
  - `feature_encoder.shape_hint.enabled=true`
  - `feature_encoder.turn_context.enabled=true`
  - `training.imitation_loss_mode=tie_aware_best_set`
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.3`
  - `training.exclude_post_riichi_discards.enabled=true`
  - `training.value_loss.type=mse`
  - `training.advantage_stabilization.clip=null`
  - `reward.point_delta_scale=0.0001`
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.01`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
  - `imitation.num_workers=10`
  - `selfplay.imitation_matches=200`
  - `training.imitation_epochs=8`
  - `selfplay.num_matches=200`
  - `selfplay.num_workers=10`
  - `selfplay.policy_ratio=1.0`
  - `selfplay.save_baseline_actions=false`
  - `evaluation.mode=rotation`
  - `evaluation.rotation_seats=[0,1,2,3]`
  - `evaluation.num_matches=30`
  - `evaluation.num_workers=10`
  - `model.hidden_dims=[512,256]`
  - `model.policy_tower.enabled=true`
  - `model.policy_tower.hidden_dim=128`
  - `model.value_tower.enabled=true`
  - `model.value_tower.hidden_dim=128`
  - `model.value_features.current_shanten.enabled=true`
  - `training.lr=5e-5`
  - `training.epochs=1`
  - `training.value_loss_coef=0.25`
  - `training.batch_size=512`
  - `training.gamma=0.99`
  - `training.gae_lambda=0.85`
  - `training.entropy_coef=0.01`
  - `training.clip_epsilon=0.15`
  - `training.device=cuda`
  - `selfplay.inference_device=cpu`
  - `evaluation.inference_device=cpu`

## 2. 実験条件（単条件）

- A: `policy_anchor_kl`
  - `training.policy_anchor.enabled=true`
  - `training.policy_anchor.type=kl`
  - `training.policy_anchor.coef=0.1`
  - `training.policy_anchor.reference=imitation_fixed`

## 3. 主評価

優先順位:
1. `eval.avg_rank`（after）
2. `eval_diff.avg_rank.delta`（悪化幅）
3. `eval.avg_score`, `win_rate`, `deal_in_rate`
4. `learner_diag.policy_anchor.anchor_loss_mean`
5. `learner_diag.clip_fraction`, `ratio_std`

比較基準:
- 主に `exp_044 B` と比較
- 必要に応じて `exp_044 E` も補助参照

## 4. 成功判定

- batch: `success_count == 20` かつ `failure_count == 0`
- run ごとに以下を満たす:
  - `summary.success == true`
  - `summary.phase_stats.learner.ppo_diag.policy_anchor.enabled == true`
  - `summary.phase_stats.learner.ppo_diag.policy_anchor.anchor_kl_mean` が記録される

## 5. 実行コマンド（予定）

```bash
python3 scripts/local/exp_046_driver.py
```

## 6. 判定メモ

- 採用寄り:
  - `exp_044 B` 比で `eval_diff.avg_rank.delta` が改善（より小さい/負側）
  - かつ `avg_rank` が同等以上
- 不採用寄り:
  - 悪化幅縮小が見られない、または `avg_rank`/`avg_score` が明確悪化
