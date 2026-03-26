# exp_040 runbook

最終更新: 2026-03-13  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: 新規追加した encoder 特徴量（`discard_ukeire_hint`, `current_shanten`, `shape_hint`）を同時に有効化し、`exp_039 B` 基準で性能と診断値の変化を確認する

---

## 0. この実験の位置づけ

- 背景:
  - CQ-0168/0169/0170 で追加した特徴量の実験導入準備が完了
  - CQ-0173 で encoder API 回帰修正も完了
- いま確認したいこと:
  - 新特徴量を同時ONしたとき、`eval_before` / after 指標 / PPO診断がどう変わるか
- なぜ単条件か:
  - まずは導入直後の挙動確認を優先し、比較基準は `exp_039 B` を再利用する

## 1. 実験の問い

1. 新特徴量3種同時ONで `eval_before` は改善するか。
2. PPO後の劣化幅（`eval_before -> eval`）は縮小するか。
3. `teacher_best_set_hit_rate` と `ppo_diag`（特に `shanten_diag` / `turn_diag`）に有意な変化は出るか。

## 2. 実験条件

### 2.1 比較軸
- A: reference（再実行なし）
  - `exp_039 B` (`runs/..._fd32f5ff`)
- B: new feature pack ON（新規実行）
  - A の設定を固定し、以下のみ追加:
    - `feature_encoder.discard_ukeire_hint.enabled=true`
    - `feature_encoder.current_shanten.enabled=true`
    - `feature_encoder.shape_hint.enabled=true`

### 2.2 共通固定（B）
- seeds: `42,43,44,45,46`
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- 固定 override（`exp_039 B` 準拠）:
  - `feature_encoder.shanten_hint.enabled=true`
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
python3 scripts/local/exp_040_driver.py
```

## 4. 成功判定

- `batch_summary.json` で `failure_count == 0`
- 各 run の `summary.json.success == true`
- `summary.encoder_features` に以下が `true` で記録される:
  - `discard_ukeire_hint`
  - `current_shanten`
  - `shape_hint`
- `summary.phase_stats.learner.ppo_diag` に
  - `shanten_diag`
  - `turn_diag`
  が存在すること

## 5. 主評価

- `eval_before`:
  - `avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`
- `eval`（after）:
  - `avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`
- 差分:
  - `eval_before -> eval` の落ち幅

## 6. 副評価

- imitation:
  - `teacher_top1_match_rate`
  - `teacher_best_set_hit_rate`
  - `value_loss`
- learner:
  - `clip_fraction`
  - `ratio_std`
  - `value_error_mean`
  - `shanten_diag`
  - `turn_diag`

## 7. 期待と判断

- 期待:
  - `teacher_best_set_hit_rate` の改善
  - `eval_before` 改善
- 判断:
  - 改善が見えれば、次は各特徴量のアブレーション（1個ずつON）へ進む
  - 改善が薄ければ、特徴量設計を再検討する
