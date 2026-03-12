# exp_035 runbook

最終更新: 2026-03-12  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: 高表現力モデル + 立直後打牌除外条件で、`batch_size↑ / epochs↓` により PPO 更新強度を適正化できるかを確認する

---

## 0. この実験の位置づけ

- 直前までの結論:
  - `exp_033` で `exclude_post_riichi_discards=true` により PPO 後悪化はかなり縮んだ
  - `exp_034` で `hidden_dims=[512,256] + dual_towers` を入れると、PPO 後悪化はさらに縮んだ
  - ただし `exp_034` は `clip_fraction` / `ratio_std` と `turn_diag` が悪化し、after 指標の総合改善には届かなかった
- なぜ今この比較をするのか:
  - `exp_034` の結果は「構造改善は効いているが、更新が強すぎる」ことを示している
  - そこで、まず `lr` は固定し、`batch_size` を増やして `epochs` を減らし、更新の分散と強度を同時に抑える
- この実験で更新したい判断:
  - 高表現力モデルは、更新強度を適正化すれば実用的な候補になるか

## 1. この実験の問い

1. `exp_034` に対して `batch_size=512, epochs=2` にすると、`clip_fraction` と `ratio_std` は下がるか。
2. `eval_before -> eval` の悪化幅は維持または改善したまま、after 指標が良くなるか。
3. `turn_diag` の early/mid/late の歪みは緩和するか。

## 2. 実験方針

### 2.1 比較軸
- A: weak-update by larger batch
  - `model.hidden_dims=[512,256]`
  - `model.policy_tower.enabled=true`
  - `model.policy_tower.hidden_dim=128`
  - `model.value_tower.enabled=true`
  - `model.value_tower.hidden_dim=128`
  - `training.exclude_post_riichi_discards.enabled=true`
  - `training.batch_size=512`
  - `training.epochs=2`
  - `training.lr=0.0001`（固定）

### 2.2 共通固定
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- 比較基準:
  - `exp_034` high-capacity dual towers with exclusion
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
  - `training.lr=0.0001`
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
  - `exp_034` の構造・reward・exclusion 条件を全固定する
- 何を変えるか:
  - `training.batch_size`
  - `training.epochs`
- `lr` を固定する理由:
  - まずは `batch_size↑ / epochs↓` の効果だけを見たい
- reuse を使わない理由:
  - learner 更新条件が変わるため、新規 run が必要

## 3. 実行方式

### 3.1 実行単位
- batch 実行（1 条件）

### 3.2 既存実験からの流用
- 参照可能な既存 run:
  - `exp_034` high-cap dual towers with exclusion
  - `exp_033` exclude post-riichi
- 流用するもの:
  - 比較参照値
- 新規実行するもの:
  - `batch_size=512, epochs=2` の 1 条件
- 実データ確認:
  - `exp_034` の report に主要値は転記済み
- 再実行が必要な理由:
  - 更新強度条件を変えた直接比較が必要

### 3.3 run_map
- `experiments/exp_035/run_map.json` に batch_dir を記録する
- report には `exp_034` 参照 run と今回 run の対応を転記する

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
    training.epochs=2 \
    training.lr=0.0001 \
    training.value_loss_coef=0.25 \
    training.batch_size=512 \
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
- `exp_034` と比較した以下の差分を最優先で見る
  - `clip_fraction`
  - `ratio_std`
  - `eval_before -> eval` の `avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`
  - `turn_diag.early/mid/late.advantage.mean`

### 6.2 副評価
- after 指標（`avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`）
- `shanten_diag.same.advantage.mean`
- `shanten_diag.improve.advantage.mean`
- `value_error_mean`
- `post_riichi_exclusion.excluded_post_riichi_discards`

### 6.3 比較優先順
- `clip_fraction / ratio_std`
- `eval_before -> eval`
- `turn_diag`
- after 指標
- `shanten_diag`

## 7. 集計方法

- 正本:
  - `runs/<batch>/batch_summary.json`
  - `runs/<batch>/<run>/metrics/train_metrics.json`
  - `runs/<batch>/<run>/summary.json`
- mean/std は seed=5 集約
- `exp_034` の値は report に並列表で転記する

## 8. 想定リスクと回避

- 実行失敗しやすい箇所:
  - モデル大型化による GPU メモリ不足
  - `batch_size=512` による learner 側メモリ圧迫
- 長時間実行時の注意:
  - `epochs=2` なので learner 時間はむしろ短縮が見込める
- 交絡要因:
  - lr は固定し、batch size / epochs のみを変える
- 再開方針:
  - 同条件で再実行し、採用 run のみ report に記載する
- 計算時間見積もり:
  - 約 1.8〜3.0 時間

## 9. レポートに必ず含める項目

- 実行対応表（`exp_034` vs `exp_035`）
- 通常評価（before/after, delta）
- 更新安定性（`clip_fraction`, `ratio_std`, `value_error_mean`）
- `shanten_diag` の主要比較
- `turn_diag` の主要比較
- `post_riichi_exclusion` 件数
- 結論と次アクション

## 10. 次アクション判定

- 採用判断:
  - `clip_fraction / ratio_std` が改善し、`eval_before -> eval` と after 指標も維持以上なら追試対象とする
- 却下判断:
  - 更新安定性も after 指標も改善しないなら、batch size 調整単独では不十分とみなす
- 追加診断判断:
  - 更新安定性だけ改善して after が伸びない場合、次は `lr=5e-5` も組み合わせる
- 次に回すべき実験:
  - 必要なら `exp_034` 構造を維持したまま `lr=5e-5` を試す

## 11. 作成前チェック

- [x] 既存実験との条件重複を確認し、流用可否を判断した
- [x] 参照する既存 run の実データが残っているか、または必要値が `report.md` に転記済みかを確認した
- [x] 再実行する条件について、流用しない理由を明記した
