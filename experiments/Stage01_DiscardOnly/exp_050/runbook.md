# exp_050 runbook

最終更新: 2026-03-15  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: `policy_anchor + low entropy` 条件の改善傾向を、`20 seeds × 20 cycles` で統計的に確定する

---

## 0. 実験の位置づけ

- 背景:
  - `exp_049`（5 seeds, 10 cycles, eval=100）で、平均的に `Δavg_rank<0` かつ `Δavg_score>0` を確認。
  - ただし seed 数が少なく、確証としてはまだ弱い。
- 方針:
  - 同一条件をそのまま拡大し、`20 seeds × 20 cycles` で確証を取る。
  - 評価は継続して `evaluation.num_matches=100` を維持。

## 1. 条件

- 条件数: 1（A のみ）
- seeds: `42..61`（20 seeds）
- A: `anchor_kl_coef_05_entropy_0000_cycle20_eval100`

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
- `selfplay.policy_ratio=1.0`
- `selfplay.save_baseline_actions=false`
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
- `training.multi_cycle.num_cycles=20`
- `training.multi_cycle.selfplay_matches_per_cycle=200`
- `training.multi_cycle.eval_each_cycle=true`

## 3. 主評価

1. 最終 after 指標（aggregate）
   - `avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`
2. 最終 `eval_before -> eval` 差分（aggregate）
   - `Δavg_rank`, `Δavg_score`, `Δwin_rate`, `Δdeal_in_rate`
3. cycle 推移（aggregate.cycles[0..19]）
   - `eval.avg_rank`, `eval.avg_score`
   - `eval_diff_avg_rank`
4. 更新安定性
   - `learner_diag.clip_fraction`, `ratio_std`

## 4. 成功判定

- `success_count == 20`, `failure_count == 0`
- 各 run で:
  - `summary.phase_stats.cycles` 長さ `20`
  - `summary.phase_stats.learner.ppo_diag.policy_anchor.enabled == true`

## 5. 判定基準

- 採用寄り:
  - aggregate で `Δavg_rank <= 0` かつ `Δavg_score >= 0`
  - cycle 後半（15-19）で `eval.avg_rank` が初期より悪化固定しない
- 非採用寄り:
  - aggregate で `Δavg_rank > 0` かつ `Δavg_score < 0`
  - cycle が進むほど一貫して崩れる

## 6. 実行コマンド

```bash
python3 scripts/local/exp_050_driver.py
```
