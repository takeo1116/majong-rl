# exp_036 runbook

最終更新: 2026-03-12  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: 高表現力モデル + 立直後打牌除外条件で、`batch_size=1024` と `lr` の組み合わせ最適化を確認する

---

## 0. この実験の位置づけ

- 直前までの結論:
  - `exp_034` で高表現力モデルは PPO 後悪化を抑えたが、更新が強すぎた
  - `exp_035` で `batch_size=512, epochs=2` にすると更新安定性・turn 歪み・after 指標が改善した
- なぜ今この比較をするのか:
  - `exp_035` の改善は「大きい batch が効く」仮説を支持している
  - 次に `batch_size=1024` を共通化し、その上で `lr=1e-4` と `5e-5` を比較して、更新強度の最適域を探る
- この実験で更新したい判断:
  - 大きい batch をさらに増やす価値があるか
  - そのとき `lr` も下げる必要があるか

## 1. この実験の問い

1. `batch_size=1024` は `exp_035` (`batch_size=512`) より更新安定性を改善するか。
2. `batch_size=1024` 条件で `lr=1e-4` と `lr=5e-5` のどちらが良いか。
3. after 指標と `eval_before -> eval` を総合したとき、次の高表現力 baseline 候補はどちらか。

## 2. 実験方針

### 2.1 比較軸
- A: larger batch / base lr
  - `training.batch_size=1024`
  - `training.epochs=2`
  - `training.lr=0.0001`
- B: larger batch / lower lr
  - `training.batch_size=1024`
  - `training.epochs=2`
  - `training.lr=0.00005`

両条件の共通:
- `model.hidden_dims=[512,256]`
- `model.policy_tower.enabled=true`
- `model.policy_tower.hidden_dim=128`
- `model.value_tower.enabled=true`
- `model.value_tower.hidden_dim=128`
- `training.exclude_post_riichi_discards.enabled=true`

### 2.2 共通固定
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- 比較基準:
  - `exp_035` large-batch low-epoch dual towers
- model / encoder:
  - `feature_encoder.shanten_hint.enabled=true`
  - `model.value_features.current_shanten.enabled=true`
- 固定 override:
  - `training.imitation_loss_mode=tie_aware_best_set`
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.1`
  - `reward.point_delta_scale=0.0001`
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.01`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
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
  - `training.value_loss_coef=0.25`
  - `training.gamma=0.99`
  - `training.gae_lambda=0.95`
  - `training.entropy_coef=0.01`
  - `training.clip_epsilon=0.2`
  - `training.device=cuda`
  - `selfplay.inference_device=cpu`
  - `evaluation.inference_device=cpu`
  - `training.exclude_post_riichi_discards.enabled=true`

### 2.3 交絡回避
- 何を固定するか:
  - `exp_035` の構造・reward・exclusion・epochs を固定する
- 何を変えるか:
  - `training.batch_size`
  - `training.lr`
- 2 条件にした理由:
  - `batch_size=1024` の価値と、そこで `lr` も下げる必要があるかを同時に見たい
- reuse を使わない理由:
  - learner 更新条件が変わるため、新規 run が必要

## 3. 実行方式

### 3.1 実行単位
- batch 実行（2 条件）

### 3.2 既存実験からの流用
- 参照可能な既存 run:
  - `exp_035` large-batch low-epoch dual towers
  - `exp_034` high-cap dual towers with exclusion
- 流用するもの:
  - 比較参照値
- 新規実行するもの:
  - A/B の 2 条件
- 実データ確認:
  - `exp_035` の report に主要値は転記済み
- 再実行が必要な理由:
  - `batch_size=1024` と `lr` の直接比較をまだ行っていないため

### 3.3 run_map
- `experiments/exp_036/run_map.json` に batch_dir を記録する
- report には `exp_035` 参照 run と今回 2 条件の対応を転記する

## 4. 実行コマンド

```bash
# 条件A: batch_size=1024, lr=1e-4
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
    training.epochs=2 \
    training.lr=0.0001 \
    training.value_loss_coef=0.25 \
    training.batch_size=1024 \
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
    model.hidden_dims='[512,256]' \
    model.value_features.current_shanten.enabled=true \
    model.policy_tower.enabled=true \
    model.policy_tower.hidden_dim=128 \
    model.value_tower.enabled=true \
    model.value_tower.hidden_dim=128 \
    training.exclude_post_riichi_discards.enabled=true
```

```bash
# 条件B: batch_size=1024, lr=5e-5
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
    training.epochs=2 \
    training.lr=0.00005 \
    training.value_loss_coef=0.25 \
    training.batch_size=1024 \
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
- `exp_035` と比較した以下を最優先で見る
  - `clip_fraction`
  - `ratio_std`
  - after 指標（`avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`）
  - `turn_diag.early/mid/late.advantage.mean`

### 6.2 副評価
- `eval_before -> eval`
- `shanten_diag.same.advantage.mean`
- `shanten_diag.improve.advantage.mean`
- `value_error_mean`
- `post_riichi_exclusion.excluded_post_riichi_discards`

### 6.3 比較優先順
- after 指標
- `clip_fraction / ratio_std`
- `turn_diag`
- `eval_before -> eval`
- `shanten_diag`

## 7. 集計方法

- 正本:
  - `runs/<batch>/batch_summary.json`
  - `runs/<batch>/<run>/metrics/train_metrics.json`
  - `runs/<batch>/<run>/summary.json`
- mean/std は seed=5 集約
- `exp_035` の値は report に並列表で転記する

## 8. 想定リスクと回避

- 実行失敗しやすい箇所:
  - `batch_size=1024` による learner 側メモリ圧迫
  - A/B とも大モデルなので総時間増加
- 長時間実行時の注意:
  - `epochs=2` で learner 時間は抑えられるが、2 条件なので総時間は長い
- 交絡要因:
  - 両条件で `batch_size` を揃え、`lr` だけを変える
- 再開方針:
  - 失敗条件のみ個別再実行
- 計算時間見積もり:
  - 約 4〜6 時間

## 9. レポートに必ず含める項目

- 実行対応表（`exp_035` vs `exp_036 A/B`）
- 通常評価（before/after, delta）
- 更新安定性（`clip_fraction`, `ratio_std`, `value_error_mean`）
- `shanten_diag` の主要比較
- `turn_diag` の主要比較
- `post_riichi_exclusion` 件数
- 結論と次アクション

## 10. 次アクション判定

- 採用判断:
  - `exp_035` を超える条件があれば、その条件を次の高表現力 baseline 候補とする
- 却下判断:
  - A/B とも `exp_035` を超えないなら、`batch_size=512` 周辺が妥当とみなす
- 追加診断判断:
  - A/B が拮抗するなら、`lr` と `batch_size` の中間条件を検討する
- 次に回すべき実験:
  - 結果次第で `target/value` 側改善へ戻るか、高表現力 baseline を固定して次へ進む

## 11. 作成前チェック

- [x] 既存実験との条件重複を確認し、流用可否を判断した
- [x] 参照する既存 run の実データが残っているか、または必要値が `report.md` に転記済みかを確認した
- [x] 再実行する条件について、流用しない理由を明記した
