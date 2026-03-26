# exp_032 runbook

最終更新: 2026-03-11  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: post-fix baseline (`exp_031`) に対して `policy_tower_only` を 1 条件だけ追加し、PPO 後悪化を縮められるかを確認する

---

## 0. この実験の位置づけ

- 直前までの結論:
  - `exp_031` で reward scale バグ修正後の baseline を再取得し、単位整合は回復した
  - ただし `eval_before -> eval` 悪化は依然残り、PPO 自体の課題がまだある
  - pre-fix 系列の `exp_029` では `policy_tower_only` が通常評価・更新安定性・診断のバランスで最良だった
- なぜ今この比較をするのか:
  - `exp_031` を新しい baseline に固定した上で、最も有望だった構造変更を最小差分で post-fix 再検証するため
- この実験で更新したい判断:
  - `policy_tower_only` が post-fix 環境でも baseline より良いか
  - `shanten_diag` / `turn_diag` の歪みがどこまで縮むか

## 1. この実験の問い

1. `policy_tower_only` は post-fix baseline より `eval_before -> eval` 悪化を縮められるか。
2. `policy_tower_only` は after 指標（`avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`）で baseline を更新できるか。
3. `shanten_diag` の `advantage` 群構造と `turn_diag` の `advantage/value_error` は baseline より自然になるか。

## 2. 実験方針

### 2.1 比較軸
- A: post-fix policy tower only
  - `model.hidden_dims=[256,128]`
  - `model.policy_tower.enabled=true`
  - `model.policy_tower.hidden_dim=128`
  - `model.value_tower.enabled=false`
  - `model.value_features.current_shanten.enabled=true`

### 2.2 共通固定
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- model / encoder:
  - `feature_encoder.shanten_hint.enabled=true`
  - `model.hidden_dims=[256,128]`
- 固定 override:
  - `training.imitation_loss_mode=tie_aware_best_set`
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.1`
  - `reward.point_delta_scale=0.0001`
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.01`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
  - `model.value_features.current_shanten.enabled=true`
  - `model.policy_tower.enabled=true`
  - `model.policy_tower.hidden_dim=128`
  - `model.value_tower.enabled=false`
  - `evaluation.mode=rotation`
  - `evaluation.rotation_seats=[0,1,2,3]`
  - `evaluation.num_matches=30`
  - `evaluation.num_workers=10`
  - `imitation.num_workers=10`
  - `selfplay.imitation_matches=25`
  - `training.imitation_epochs=4`
  - `selfplay.num_matches=200`
  - `selfplay.num_workers=10`
  - `selfplay.policy_ratio=1.0`
  - `selfplay.save_baseline_actions=false`
  - `training.epochs=4`
  - `training.lr=0.0001`
  - `training.value_loss_coef=0.25`
  - `training.batch_size=256`
  - `training.gamma=0.99`
  - `training.gae_lambda=0.95`
  - `training.entropy_coef=0.01`
  - `training.clip_epsilon=0.2`
  - `training.device=cuda`
  - `selfplay.inference_device=cpu`
  - `evaluation.inference_device=cpu`

### 2.3 交絡回避
- 何を固定するか:
  - `exp_031` baseline の全条件
- 何を変えるか:
  - `policy_tower.enabled`
  - `policy_tower.hidden_dim`
- reuse を使わない理由:
  - post-fix + `reward / delta_t` 診断込みの新規 run が必要

## 3. 実行方式

### 3.1 実行単位
- batch 実行（1 条件）

### 3.2 既存実験からの流用
- 参照可能な既存 run:
  - `exp_031` A post-fix baseline
  - `exp_029` C policy tower only（pre-fix 参照）
- 流用するもの:
  - 比較参照値のみ
- 新規実行するもの:
  - post-fix `policy_tower_only` 1 条件
- 実データ確認:
  - `exp_031` は post-fix baseline として有効
  - `exp_029 C` は pre-fix のため定量比較の正本には使わない
- 再実行が必要な理由:
  - `policy_tower_only` の効果を post-fix 条件で再検証するため

### 3.3 run_map
- `experiments/exp_032/run_map.json` に batch_dir を記録
- report には `exp_031` baseline と並べて対応表を転記

## 4. 実行コマンド

```bash
python3 -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46 \
  --override \
    feature_encoder.shanten_hint.enabled=true \
    training.imitation_loss_mode=tie_aware_best_set \
    training.imitation_value_warmstart.enabled=true \
    training.imitation_value_warmstart.coef=0.1 \
    reward.point_delta_scale=0.0001 \
    imitation.num_workers=10 \
    selfplay.imitation_matches=25 \
    training.imitation_epochs=4 \
    selfplay.num_matches=200 \
    selfplay.num_workers=10 \
    selfplay.policy_ratio=1.0 \
    selfplay.save_baseline_actions=false \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=30 \
    evaluation.num_workers=10 \
    training.epochs=4 \
    training.lr=0.0001 \
    training.value_loss_coef=0.25 \
    training.batch_size=256 \
    training.gamma=0.99 \
    training.gae_lambda=0.95 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu \
    reward.shaping.shanten_delta.enabled=true \
    reward.shaping.shanten_delta.scale=0.01 \
    reward.shaping.shanten_delta.mode=both \
    reward.shaping.shanten_delta.schedule.type=linear_decay \
    model.hidden_dims='[256,128]' \
    model.value_features.current_shanten.enabled=true \
    model.policy_tower.enabled=true \
    model.policy_tower.hidden_dim=128 \
    model.value_tower.enabled=false
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
- `ppo_diag.shanten_diag` に以下が存在すること
  - `reward`
  - `point_delta_reward`
  - `shanten_delta_reward`
  - `delta_t`
- `ppo_diag.turn_diag` が存在すること
- `summary.model_features.policy_tower.enabled == true`

## 6. 主評価と副評価

### 6.1 主評価
- `exp_031` baseline との比較で、以下を最優先とする
  - `Δavg_rank`
  - `Δavg_score`
  - `shanten_diag` の `advantage.mean`
  - `turn_diag` の `advantage.mean`

### 6.2 副評価
- after 指標（`avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`）
- 更新安定性（`clip_fraction`, `ratio_std`, `value_error_mean`）
- `shanten_diag` の `reward / delta_t / value_error`

### 6.3 比較優先順
- `eval_before -> eval`
- after 指標
- `shanten_diag.advantage`
- `turn_diag.advantage`
- `clip_fraction / ratio_std`

## 7. 集計方法

- 正本:
  - `runs/<batch>/batch_summary.json`
  - `runs/<batch>/<run>/metrics/train_metrics.json`
  - `runs/<batch>/<run>/summary.json`
- mean/std は seed=5 集約
- report では `exp_031` baseline と並列表で比較する

## 8. 想定リスクと回避

- 実行失敗しやすい箇所:
  - GPU メモリ不足、worker 起動失敗
- 長時間実行時の注意:
  - 途中停止時は batch_dir を控え、成功 seed 数を確認
- 交絡要因:
  - `policy_tower` 以外の差分を混入させない
- 再開方針:
  - 失敗時は同一条件で再実行
- 計算時間見積もり:
  - 約 1.5〜2.5 時間

## 9. レポートに必ず含める項目

- 実行対応表（exp_032 と exp_031 baseline）
- 通常評価（before/after, delta）
- `shanten_diag` の主要比較
- `turn_diag` の主要比較
- 更新安定性比較
- 結論と次アクション

## 10. 次アクション判定

- 採用判断:
  - `policy_tower_only` が `exp_031` baseline より `eval_before -> eval` と after 指標で改善し、診断値も悪化しなければ次の基準候補
- 却下判断:
  - 通常評価が baseline 以下で、診断も改善しない場合は不採用
- 追加診断判断:
  - 通常評価は良いが診断が悪化する場合は `dual_towers` を post-fix で比較

## 11. 作成前チェック

- [x] 既存実験との条件重複を確認し、流用可否を判断した
- [x] `exp_031` を post-fix baseline として参照する前提を明記した
- [x] 再実行理由（post-fix での `policy_tower_only` 再検証）を明記した
