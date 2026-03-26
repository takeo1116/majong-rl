# exp_043 runbook

最終更新: 2026-03-14  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: 高速化後の計算資源を使い、PPO安定化（5条件）と `discard_ukeire_hint` 効果（2条件）を **20seed 一括**で評価する

---

## 0. 実験の位置づけ

- 背景:
  - `exp_042` で高速化後の再現性（速度のみ改善、挙動同値）を確認済み
  - `discard_ukeire_hint=true` は速度問題が解消された一方、対戦成績は悪化傾向だった
- 今回の狙い:
  - seed を 20 に増やして統計力を上げる
  - PPO更新強度の適正域を同時探索
  - `discard_ukeire_hint` の採否を「偶然でない差」で判断する

## 1. 共通固定

- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42..61`（20 seeds）
- 共通 override:
  - `feature_encoder.shanten_hint.enabled=true`
  - `feature_encoder.current_shanten.enabled=true`
  - `feature_encoder.shape_hint.enabled=true`
  - `training.imitation_loss_mode=tie_aware_best_set`
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.3`
  - `reward.point_delta_scale=0.0001`
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
  - `training.value_loss_coef=0.25`
  - `training.batch_size=512`
  - `training.gamma=0.99`
  - `training.entropy_coef=0.01`
  - `training.device=cuda`
  - `selfplay.inference_device=cpu`
  - `evaluation.inference_device=cpu`
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.01`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
  - `model.hidden_dims=[512,256]`
  - `model.value_features.current_shanten.enabled=true`
  - `model.policy_tower.enabled=true`
  - `model.policy_tower.hidden_dim=128`
  - `model.value_tower.enabled=true`
  - `model.value_tower.hidden_dim=128`
  - `training.exclude_post_riichi_discards.enabled=true`

## 2. 実験条件（7 conditions）

- A: baseline_off
  - `discard_ukeire_hint=false`
  - `training.lr=1e-4`
  - `training.epochs=2`
  - `training.gae_lambda=0.90`
  - `training.clip_epsilon=0.20`
- B: lr7e5_off
  - `discard_ukeire_hint=false`
  - `training.lr=7e-5`
  - `training.epochs=2`
  - `training.gae_lambda=0.90`
  - `training.clip_epsilon=0.20`
- C: lr5e5_off
  - `discard_ukeire_hint=false`
  - `training.lr=5e-5`
  - `training.epochs=2`
  - `training.gae_lambda=0.90`
  - `training.clip_epsilon=0.20`
- D: weak_update_off
  - `discard_ukeire_hint=false`
  - `training.lr=7e-5`
  - `training.epochs=1`
  - `training.gae_lambda=0.90`
  - `training.clip_epsilon=0.15`
- E: weaker_update_off
  - `discard_ukeire_hint=false`
  - `training.lr=5e-5`
  - `training.epochs=1`
  - `training.gae_lambda=0.85`
  - `training.clip_epsilon=0.15`
- F: baseline_on
  - `discard_ukeire_hint=true`
  - `training.lr=1e-4`
  - `training.epochs=2`
  - `training.gae_lambda=0.90`
  - `training.clip_epsilon=0.20`
- G: weak_update_on
  - `discard_ukeire_hint=true`
  - `training.lr=7e-5`
  - `training.epochs=1`
  - `training.gae_lambda=0.90`
  - `training.clip_epsilon=0.15`

## 3. 主評価

優先順位:
1. `eval.avg_rank`（低いほど良い）
2. `eval_before -> eval` の悪化量（`Δavg_rank`）
3. `deal_in_rate`
4. `win_rate`
5. 更新安定性（`clip_fraction`, `ratio_std`）

## 4. 成功判定

- 各 batch: `success_count > 0` かつ `failure_count == 0`
- 各 run:
  - `summary.json.success == true`
  - `summary.phase_stats.learner.ppo_diag` が存在
  - `train_metrics.json` が存在
- `run_map.json` に全条件の batch_dir / status が記録されること

## 5. 実行コマンド

```bash
python3 scripts/local/exp_043_driver.py
```

## 6. 運用メモ

- 夜間実行向けに、driver は「条件単位 fail-fast せず継続」設計。
- 途中で条件失敗が出ても残り条件は継続し、最終 run_map に失敗理由を残す。

