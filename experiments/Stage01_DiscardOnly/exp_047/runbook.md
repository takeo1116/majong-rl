# exp_047 runbook

最終更新: 2026-03-15  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: `policy_anchor (KL)` の係数を強めたときの効果を、`PPO 3 cycle` 条件で比較する

---

## 0. 実験の位置づけ

- 参照:
  - `exp_046` (`policy_anchor.kl coef=0.1`, 1 cycle)
- 今回の問い:
  - `coef` を `0.3 / 0.5` に上げると、PPO後の壊れ幅はさらに縮むか
  - 一方でアンカーが強すぎて伸びが止まらないか
- 追加変更:
  - PPOを `3 cycle` に変更し、短い反復で安定性を確認する

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
  - `training.policy_anchor.enabled=true`
  - `training.policy_anchor.type=kl`
  - `training.policy_anchor.reference=imitation_fixed`
  - `reward.point_delta_scale=0.0001`
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.01`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
  - `imitation.num_workers=10`
  - `selfplay.imitation_matches=200`
  - `training.imitation_epochs=8`
  - `selfplay.num_matches=200`  # fallback
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
  - `training.multi_cycle.enabled=true`
  - `training.multi_cycle.num_cycles=3`
  - `training.multi_cycle.selfplay_matches_per_cycle=200`
  - `training.multi_cycle.eval_each_cycle=true`

## 2. 実験条件（2条件）

- A: `anchor_kl_coef_03_cycle3`
  - `training.policy_anchor.coef=0.3`
- B: `anchor_kl_coef_05_cycle3`
  - `training.policy_anchor.coef=0.5`

## 3. 主評価

優先順位:
1. after `eval.avg_rank`
2. `eval_diff.avg_rank.delta`（最終 cycle）
3. `eval.avg_score`, `win_rate`, `deal_in_rate`
4. `learner_diag.policy_anchor.anchor_kl_mean`
5. `aggregate.cycles[0..2]` の推移（悪化→回復の有無）

比較基準:
- `exp_046 (coef=0.1, 1 cycle)`
- 条件間比較（0.3 vs 0.5）

## 4. 成功判定

- 各条件で `success_count == 20`, `failure_count == 0`
- 各 run で:
  - `summary.phase_stats.cycles` 長さ `3`
  - `summary.phase_stats.learner.ppo_diag.policy_anchor.enabled == true`
  - `summary.phase_stats.learner.ppo_diag.policy_anchor.type == kl`
  - `summary.phase_stats.learner.ppo_diag.policy_anchor.coef` が設定値と一致

## 5. 実行コマンド

```bash
python3 scripts/local/exp_047_driver.py
```

## 6. 判定メモ

- 採用寄り:
  - `Δavg_rank` が 0.1設定よりさらに縮小
  - after 指標の悪化なし（少なくとも同等）
- 不採用寄り:
  - `coef` 増加で after が悪化、または `Δavg_rank` 改善が消える
