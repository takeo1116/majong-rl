# exp_042 runbook

最終更新: 2026-03-14  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: 高速化修正後の実行時間を、`exp_041 B` と**完全同条件**で再計測する

---

## 0. この実験の位置づけ

- 背景:
  - `exp_041` は `discard_ukeire_hint=false` 条件で速度と性能を確認済み
  - その後、C++ 側/encoder 側の高速化修正が入った
- いま確認したいこと:
  - 学習条件を固定したまま、実行時間（特に self-play）だけがどれだけ改善したか

## 1. 実験の問い

1. `exp_041 B` と比べて `total_duration_sec` は短縮したか。
2. `phase_timing.selfplay.duration_sec` は短縮したか。
3. 速度改善の代わりに性能 (`eval_before` / `eval`) が悪化していないか。

## 2. 実験条件

### 2.1 比較軸

- A: reference（再実行なし）
  - `exp_041 B`: `runs/20260314_stage1_full_flat_mlp_imitation_then_ppo_batch_4aae4f28`
  - encoder:
    - `discard_ukeire_hint=false`
    - `current_shanten=true`
    - `shape_hint=true`
- B: new run（新規実行）
  - A と**完全同条件**で再実行（コードのみ最新）

### 2.2 共通固定（B）

- seeds: `42,43,44,45,46`
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- 固定 override:
  - `feature_encoder.shanten_hint.enabled=true`
  - `feature_encoder.discard_ukeire_hint.enabled=false`
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
  - `training.epochs=2`
  - `training.lr=0.0001`
  - `training.value_loss_coef=0.25`
  - `training.batch_size=512`
  - `training.gamma=0.99`
  - `training.gae_lambda=0.90`
  - `training.entropy_coef=0.01`
  - `training.clip_epsilon=0.2`
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

## 3. 実行コマンド

```bash
python3 scripts/local/exp_042_driver.py
```

## 4. 成功判定

- `batch_summary.json` で `failure_count == 0`
- 各 run の `summary.json.success == true`
- B の `summary.encoder_features` で:
  - `discard_ukeire_hint == false`
  - `current_shanten == true`
  - `shape_hint == true`
- `summary.phase_timing` に `imitation/selfplay/eval_before/learner/eval` が存在

## 5. 主評価（速度）

- `total_duration_sec`
- `phase_timing.selfplay.duration_sec`
- `phase_timing.imitation.duration_sec`
- seed ごとの差分（B - A）と平均相対差分（%）

## 6. 副評価（性能）

- `eval_before`: `avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`
- `eval`（after）: 同上
- imitation 指標:
  - `teacher_top1_match_rate`
  - `teacher_best_set_hit_rate`
- learner 指標:
  - `clip_fraction`, `ratio_std`, `value_error_mean`

## 7. 判断基準

- 速度改善が明確で、性能差がノイズ範囲なら高速化を採用
- 速度が改善しても性能悪化が明確なら、最適化箇所のロールバック候補を切り分ける

