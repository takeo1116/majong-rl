# exp_041 runbook

最終更新: 2026-03-14  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: `discard_ukeire_hint` が実行時間に与える影響を、`exp_040` と同条件で定量比較する

---

## 0. この実験の位置づけ

- 背景:
  - `exp_040` で新特徴量パックON後、1条件あたりの実行時間が大幅に増加
  - 事前のミクロ計測では `discard_ukeire_hint` が主要ボトルネック候補
- いま確認したいこと:
  - フル実験条件（5 seeds）で `discard_ukeire_hint` の ON/OFF が総時間・フェーズ時間へ与える寄与

## 1. 実験の問い

1. `discard_ukeire_hint=false` にすると、`phase_timing.selfplay.duration_sec` はどれだけ短縮するか。
2. 総実行時間（`total_duration_sec`）はどれだけ短縮するか。
3. 性能指標（`eval_before` / `eval`）に実用上無視できない劣化が出るか。

## 2. 実験条件

### 2.1 比較軸

- A: reference（再実行なし）
  - `exp_040 B`: `runs/20260313_stage1_full_flat_mlp_imitation_then_ppo_batch_1927d470`
  - encoder:
    - `discard_ukeire_hint=true`
    - `current_shanten=true`
    - `shape_hint=true`
- B: new run（新規実行）
  - A と同一設定で **`feature_encoder.discard_ukeire_hint.enabled=false` のみ変更**
  - それ以外の override は完全固定

### 2.2 共通固定（B）

- seeds: `42,43,44,45,46`
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- 固定 override:
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
python3 scripts/local/exp_041_driver.py
```

## 4. 成功判定

- `batch_summary.json` で `failure_count == 0`
- 各 run の `summary.json.success == true`
- B の `summary.encoder_features` で:
  - `discard_ukeire_hint == false`
  - `current_shanten == true`
  - `shape_hint == true`
- A/B ともに `summary.phase_timing` に以下が存在:
  - `imitation`, `selfplay`, `eval_before`, `learner`, `eval`

## 5. 主評価（時間）

- `total_duration_sec`（run 単位）
- `phase_timing.<phase>.duration_sec`（phase 単位）
  - 特に `selfplay` を重点評価
- 集計:
  - seed ごとの差分（B - A）
  - 平均差分と相対差分（%）

## 6. 副評価（性能）

- `eval_before`: `avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`
- `eval`（after）: 同上
- imitation 指標:
  - `teacher_top1_match_rate`
  - `teacher_best_set_hit_rate`
- learner 指標:
  - `clip_fraction`, `ratio_std`, `value_error_mean`

## 7. 期待と判断

- 期待:
  - `selfplay` 時間の有意短縮（主要因の切り分け）
- 判断:
  - 速度改善が大きく、性能劣化が許容範囲なら、以後の夜間実験は `discard_ukeire_hint=false` をデフォルト候補にする
  - 劣化が大きければ、`discard_ukeire_hint` 自体は維持しつつ実装最適化（キャッシュ/差分計算）を優先する

## 8. 備考

- 本実験は「性能改善」よりも「速度要因の確証」を主目的とする。
- A は再実行せず既存成果物を参照し、B のみ新規実行する。
