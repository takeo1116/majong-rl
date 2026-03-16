# exp_044 runbook

最終更新: 2026-03-14  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: `exp_043 E` を基準に、`turn_context` と `value/advantage` 安定化オプションの寄与を切り分ける

---

## 0. 実験の位置づけ

- 基準条件:
  - `exp_043 E` (`lr=5e-5, epochs=1, gae_lambda=0.85, clip_epsilon=0.15, discard_ukeire_hint=false`)
- 今回の狙い:
  - `turn_context` の純効果確認
  - `huber value loss` の純効果確認
  - `advantage clip` の純効果確認
  - 同時適用時の相互作用確認

## 1. 共通固定

- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42..61`（20 seeds）
- 共通 override:
  - `feature_encoder.shanten_hint.enabled=true`
  - `feature_encoder.discard_ukeire_hint.enabled=false`
  - `feature_encoder.current_shanten.enabled=true`
  - `feature_encoder.shape_hint.enabled=true`
  - `training.imitation_loss_mode=tie_aware_best_set`
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.3`
  - `training.exclude_post_riichi_discards.enabled=true`
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

## 2. 実験条件（5 conditions）

- A: baseline
  - `feature_encoder.turn_context.enabled=false`
  - `training.value_loss.type=mse`
  - `training.advantage_stabilization.clip=null`
- B: turn_context_on
  - `feature_encoder.turn_context.enabled=true`
  - `training.value_loss.type=mse`
  - `training.advantage_stabilization.clip=null`
- C: huber_on
  - `feature_encoder.turn_context.enabled=false`
  - `training.value_loss.type=huber`
  - `training.value_loss.huber_delta=1.0`
  - `training.advantage_stabilization.clip=null`
- D: adv_clip_on
  - `feature_encoder.turn_context.enabled=false`
  - `training.value_loss.type=mse`
  - `training.advantage_stabilization.clip=2.0`
- E: all_on
  - `feature_encoder.turn_context.enabled=true`
  - `training.value_loss.type=huber`
  - `training.value_loss.huber_delta=1.0`
  - `training.advantage_stabilization.clip=2.0`

## 3. 主評価

優先順位:
1. `eval.avg_rank`（低いほど良い）
2. `eval_before -> eval` の悪化量
3. `deal_in_rate`
4. `win_rate`
5. learner 診断（`clip_fraction`, `ratio_std`, `value_error_mean`）

追加確認:
- `advantage_abs_mean_before_clip`
- `advantage_abs_mean_after_clip`
- `advantage_clip_fraction`

## 4. 成功判定

- 各 batch: `success_count > 0` かつ `failure_count == 0`
- 各 run:
  - `summary.json.success == true`
  - `summary.phase_stats.learner.ppo_diag` が存在
  - `train_metrics.json` が存在
  - 条件 B/E では `summary.encoder_features.turn_context == true`

## 5. 実行コマンド

```bash
python3 scripts/local/exp_044_driver.py
```

## 6. 運用メモ

- driver は条件単位 fail-fast せず継続（最終 `run_map.json` に失敗理由を残す）。
- `turn_context` と安定化オプションの比較なので、今回は `discard_ukeire_hint` を固定 off にする。
