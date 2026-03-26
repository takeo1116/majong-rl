# exp_039 runbook

最終更新: 2026-03-13  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: `exp_038 B` を基準に imitation データをさらに増やし、`eval_before` がまだ伸びるか、またその改善が PPO 後の最終性能にも波及するかを確認する

---

## 0. この実験の位置づけ

- 直前までの結論:
  - `exp_038 B`（`imitation_matches=50`, `imitation_epochs=8`）は `exp_037 D` より `eval_before` と after 指標の両方で前向きだった
  - shaping 再評価（`exp_038 C/D/E`）は baseline を更新できず、一旦主系列から外れた
  - 現在の主な問いは「この特徴量 + モデルに imitation だけでまだ伸びしろがあるか」
- なぜ今この比較をするのか:
  - imitation をさらに増やしても `eval_before` が伸びるなら、現行特徴量/モデルでの学習余地はまだある
  - 逆に `eval_before` が頭打ちなら、特徴量改善へ進む判断が強くなる
- この実験で更新したい判断:
  - 現行特徴量/モデルで PPO 改善を追う価値がまだあるか、それとも特徴量改善へ進むべきか

## 1. この実験の問い

1. `selfplay.imitation_matches=200` にすると `eval_before` は `exp_038 B` よりさらに伸びるか。
2. imitation 増量の効果は after 指標にも波及するか。
3. `eval_before` は伸びるが PPO 後悪化が変わらない場合、特徴量改善へ進む判断材料になるか。

## 2. 実験方針

### 2.1 比較軸
- A: baseline reference（reuse / 参照のみ）
  - `exp_038 B`
  - `selfplay.imitation_matches=50`
  - `training.imitation_epochs=8`
  - `selfplay.num_matches=200`
- B: heavier imitation only
  - A + `selfplay.imitation_matches=200`
  - `training.imitation_epochs=8`（固定）
  - `selfplay.num_matches=200`（固定）

### 2.2 共通固定
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- 比較基準:
  - A は `exp_038 B`
- model / encoder:
  - `feature_encoder.shanten_hint.enabled=true`
  - `model.hidden_dims=[512,256]`
  - `model.value_features.current_shanten.enabled=true`
  - `model.policy_tower.enabled=true`
  - `model.policy_tower.hidden_dim=128`
  - `model.value_tower.enabled=true`
  - `model.value_tower.hidden_dim=128`
- 固定 override:
  - `training.imitation_loss_mode=tie_aware_best_set`
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.3`
  - `reward.point_delta_scale=0.0001`
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.01`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
  - `imitation.num_workers=10`
  - `selfplay.imitation_matches=50`（Bのみ 200）
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
  - `training.exclude_post_riichi_discards.enabled=true`

### 2.3 交絡回避
- 何を固定するか:
  - `exp_038 B` の構造・reward・PPO 条件・exclusion を固定する
- 何を変えるか:
  - `selfplay.imitation_matches` のみ
- 流用を使う理由:
  - A は既存の最有力 imitation 増量条件であり、新規実行不要

## 3. 実行方式

### 3.1 実行単位
- A: reuse / 参照のみ
- B: batch 実行（新規 1 条件）

### 3.2 既存実験からの流用
- 参照可能な既存 run:
  - `exp_038 B`
- 流用するもの:
  - baseline 参照値
- 新規実行するもの:
  - `selfplay.imitation_matches=200` の 1 条件
- 実データ確認:
  - `exp_038` の report に主要値は転記済み
  - `runs/20260313_stage1_full_flat_mlp_imitation_then_ppo_batch_4c6fdea1` が残っていることを前提にする
  - 実データが消えても `report.md` 転記値で baseline 比較は継続可能
- 再実行が必要な理由:
  - imitation 追加増量の直接比較が必要

### 3.3 run_map
- `experiments/exp_039/run_map.json` に B の batch_dir を記録する
- A は `reference_batch_dir` として併記する
- report には `exp_038 B` と今回 run の対応を転記する

## 4. 実行コマンド

```bash
# 条件B: imitation_matches=200
python3 -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46 \
  --override \
    feature_encoder.shanten_hint.enabled=true \
    training.imitation_loss_mode=tie_aware_best_set \
    training.imitation_value_warmstart.enabled=true \
    training.imitation_value_warmstart.coef=0.3 \
    reward.point_delta_scale=0.0001 \
    imitation.num_workers=10 \
    selfplay.imitation_matches=200 \
    training.imitation_epochs=8 \
    selfplay.num_matches=200 \
    selfplay.num_workers=10 \
    selfplay.policy_ratio=1.0 \
    selfplay.save_baseline_actions=false \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=30 \
    evaluation.num_workers=10 \
    training.epochs=2 \
    training.lr=0.0001 \
    training.value_loss_coef=0.25 \
    training.batch_size=512 \
    training.gamma=0.99 \
    training.gae_lambda=0.90 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu \
    reward.shaping.shanten_delta.enabled=true \
    reward.shaping.shanten_delta.scale=0.01 \
    reward.shaping.shanten_delta.mode=both \
    reward.shaping.shanten_delta.schedule.type=linear_decay \
    model.hidden_dims='[512,256]' \
    model.value_features.current_shanten.enabled=true \
    model.policy_tower.enabled=true \
    model.policy_tower.hidden_dim=128 \
    model.value_tower.enabled=true \
    model.value_tower.hidden_dim=128 \
    training.exclude_post_riichi_discards.enabled=true
```

## 5. 成功判定

### 5.1 共通
- `summary.json.success == true`
- 必須成果物:
  - `summary.json`
  - `config.yaml`
  - `metrics/train_metrics.json`
  - `eval/eval_diff.json`
  - `batch_summary.json`

### 5.2 診断キー
- `summary.model_features.policy_tower.enabled == true`
- `summary.model_features.value_tower.enabled == true`
- `summary.phase_stats.learner.post_riichi_exclusion` が存在すること
- `ppo_diag.shanten_diag` が存在すること
- `ppo_diag.turn_diag` が存在すること

## 6. 主評価と副評価

### 6.1 主評価
- `eval_before`
  - `avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`
- after 指標
  - `avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`
- `eval_before -> eval`

### 6.2 副評価
- imitation 指標
  - `teacher_top1_match_rate`
  - `teacher_best_set_hit_rate`
  - `value_loss`
- `clip_fraction`
- `ratio_std`
- `value_error_mean`
- `shanten_diag`
- `turn_diag`

### 6.3 比較優先順
- `eval_before`
- after 指標
- `eval_before -> eval`
- imitation 指標
- learner 診断

## 7. 集計方法

- 正本:
  - `runs/<batch>/batch_summary.json`
  - `runs/<batch>/<run>/metrics/train_metrics.json`
  - `runs/<batch>/<run>/summary.json`
- mean/std は seed=5 集約
- A baseline は `exp_038/report.md` から転記、必要なら run 実データも参照

## 8. 想定リスクと回避

- 実行失敗しやすい箇所:
  - imitation データ量増加で時間増
- 長時間実行時の注意:
  - `selfplay.imitation_matches=200` なので、`exp_038 B` よりかなり長い
- 交絡要因:
  - imitation 条件だけを変え、PPO 条件は全固定する
- 再開方針:
  - 中断時は batch 単位で再実行し、`run_map.json` を更新する
- 計算時間見積もり:
  - `exp_038 B` より大きく増える。夜間実行前提

## 9. レポートに必ず含める項目

- A baseline との比較表
- `eval_before` 比較表
- after 指標比較表
- `eval_before -> eval` 比較表
- imitation 指標比較表
- 結論
- 次アクション

## 10. 次アクション判定

- どの結果なら採用:
  - `eval_before` が明確に伸び、after 指標も悪化しない
- どの結果なら却下:
  - `eval_before` がほぼ伸びず、after も改善しない
- どの結果なら特徴量改善へ進む:
  - `eval_before` は伸びるが after / `eval_before -> eval` が改善しない
- 次に回すべき実験:
  - imitation がさらに効けば self-play 増量
  - 効かなければ特徴量改善へ移行

## 11. 作成前チェック

- [x] 既存実験との条件重複を確認し、流用可否を判断した
- [x] 参照する既存 run の実データが残っているか、または必要値が `report.md` に転記済みかを確認した
- [x] 再実行する条件について、流用しない理由を明記した
