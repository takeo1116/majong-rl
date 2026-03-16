# exp_051 runbook

最終更新: 2026-03-15  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: `rule_mix(actor3+rule1) + 2段学習` で、PPO崩壊傾向が緩和されるかを `5 seeds × 10 cycles` で確認する

---

## 0. 実験の位置づけ

- 背景:
  - 直近の `policy_anchor + low entropy` 単独では、cycle を重ねると最終的な悪化傾向が残った。
  - 次は self-play のデータ分布を変えるため、`rule_mix`（actor3+rule1 相当）を導入する。
- 方針:
  - self-play は `policy_ratio=0.75`・`save_baseline_actions=true` で baseline 行動を shard に保存。
  - learner は `rule_mix_learner` で 2段学習:
    1. baseline サンプルのみ imitation
    2. policy サンプルのみ PPO

## 1. 条件

- 条件数: 1（A のみ）
- seeds: `42..46`（5 seeds）
- A: `rule_mix_policy075_two_stage_anchor05_entropy0000_cycle10_eval100`

## 2. 共通固定（override）

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
- `training.policy_anchor.coef=0.5`
- `training.policy_anchor.reference=imitation_fixed`
- `training.entropy_coef=0.0`
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
- `selfplay.policy_ratio=1.0`  # fallback（cycle中は rule_mix で上書き）
- `selfplay.save_baseline_actions=false`  # fallback（cycle中は rule_mix で上書き）
- `evaluation.mode=rotation`
- `evaluation.rotation_seats=[0,1,2,3]`
- `evaluation.num_matches=100`
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
- `training.clip_epsilon=0.15`
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`
- `training.multi_cycle.enabled=true`
- `training.multi_cycle.num_cycles=10`
- `training.multi_cycle.selfplay_matches_per_cycle=200`
- `training.multi_cycle.eval_each_cycle=true`
- `training.rule_mix.enabled=true`
- `training.rule_mix.policy_ratio=0.75`
- `training.rule_mix.save_baseline_actions=true`
- `training.rule_mix_learner.enabled=true`
- `training.rule_mix_learner.baseline_imitation_epochs=1`
- `training.rule_mix_learner.policy_ppo_epochs=1`
- `training.rule_mix_learner.order=baseline_then_policy`

## 3. 主評価

1. 最終 after 指標（aggregate）
   - `avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`
2. 最終 `eval_before -> eval` 差分（aggregate）
   - `Δavg_rank`, `Δavg_score`
3. cycle 推移（aggregate.cycles[0..9]）
   - `eval.avg_rank`, `eval.avg_score`, `eval_diff_avg_rank`
4. rule_mix 動作確認
   - `actor_type_counts`（policy/baseline）
   - `learner_stages.baseline_imitation`, `learner_stages.policy_ppo`

## 4. 成功判定

- `success_count == 5`, `failure_count == 0`
- 各 run で:
  - `summary.phase_stats.cycles` 長さ `10`
  - `summary.phase_stats.learner.ppo_diag.policy_anchor.enabled == true`
  - `cycles[*].learner_stages.policy_ppo.executed == true`

## 5. 判定基準（傾向確認）

- 採用寄り:
  - `Δavg_rank <= 0` かつ `Δavg_score >= 0` を維持する seed が増える
  - cycle 後半（7-9）で崩れ方が緩くなる
- 非採用寄り:
  - 後半 cycle で一貫して悪化（rank 悪化 + score 悪化）

## 6. 実行コマンド

```bash
python3 scripts/local/exp_051_driver.py
```
