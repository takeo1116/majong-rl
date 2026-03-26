# exp_045 runbook

最終更新: 2026-03-14  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: 最良近傍条件で `100 cycle` の長期学習を 1 seed で実行し、  
PPO が「初期悪化後に回復するか / 長期で悪化し続けるか」を時系列で観測する

---

## 0. 実験の位置づけ

- 基準条件:
  - `exp_044 B`（`turn_context=true`, `mse`, `adv_clipなし`）
- 今回の狙い:
  - 20 seed の横比較ではなく、1 seed で縦方向（時間軸）の挙動を深掘り
  - `eval.avg_rank` / `eval.avg_score` の cycle 推移で回復局面の有無を確認

## 1. 共通固定

- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42`（単 seed）
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
  - `selfplay.num_matches=200`  # fallback（multi_cycleで上書き）
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

- A: long_cycle_100_seed42
  - `training.multi_cycle.enabled=true`
  - `training.multi_cycle.num_cycles=100`
  - `training.multi_cycle.selfplay_matches_per_cycle=200`
  - `training.multi_cycle.eval_each_cycle=true`

## 3. 主評価

優先順位:
1. cycle別 `eval.avg_rank` 推移
2. cycle別 `eval.avg_score` 推移
3. cycle別 `eval_diff.avg_rank.delta` と `eval_diff.avg_score.delta`
4. cycle別 `deal_in_rate`, `win_rate`
5. cycle別 learner 診断（`clip_fraction`, `ratio_std`, `value_error_mean`）

判定観点:
- 回復型: 前半悪化後に中後半で `avg_rank` 改善トレンドへ転換
- 停滞型: 悪化後に横ばい（改善なし）
- 継続悪化型: cycle 進行に従って悪化が継続

## 4. 成功判定

- batch: `success_count == 1` かつ `failure_count == 0`
- run:
  - `summary.json.success == true`
  - `summary.phase_stats.cycles` が存在し、長さ `100`
  - `batch_summary.json.aggregate.cycles` が存在し、長さ `100`
  - `checkpoints/checkpoint_cycle_00.pt` 〜 `checkpoint_cycle_99.pt` が存在

## 5. 実行コマンド

```bash
python3 scripts/local/exp_045_driver.py
```

## 6. 運用メモ

- 長時間実行になるため、途中停止時はまず `run_dir` の `summary.json` と `checkpoints/` の残存状況を確認する。
- report では、cycle 1-20 / 21-60 / 61-100 の3区間でトレンドを分けて記述する。
