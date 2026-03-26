# exp_033 runbook

最終更新: 2026-03-12  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: 立直後打牌除外により `same` 群の advantage 正偏りと PPO 後悪化が緩和するかを 1 条件で確認する

---

## 0. この実験の位置づけ

- 直前までの結論:
  - `exp_031` で reward scale 修正後 baseline を再取得し、報酬単位は正常化した
  - ただし `same` 群の `advantage.mean > 0` と `improve` 群の `advantage.mean < 0` は残った
  - `exp_032` の `policy_tower_only` は post-fix baseline を更新できなかった
- なぜ今この比較をするのか:
  - 現在の最有力仮説は「立直後打牌が `same` 群へ混入し、終端近傍の信号で `same` を押し上げている」である
  - CQ-0163/0164/0165 により、立直後打牌の診断と learner 除外が利用可能になった
- この実験で更新したい判断:
  - 立直後打牌除外を baseline 改善案として追う価値があるか
  - `same` 群正偏りの主因が立直後混入かどうか

## 1. この実験の問い

1. `training.exclude_post_riichi_discards.enabled=true` により、`same` 群の `post_riichi_discard_ratio` は高いままでも、学習後の `same.advantage.mean` は低下するか。
2. `exp_031` baseline と比べて、`eval_before -> eval` の悪化幅は縮小するか。
3. `improve / same / worsen` の advantage 構造は、立直後打牌除外によって自然な方向に近づくか。

## 2. 実験方針

### 2.1 比較軸
- A: post-riichi 除外あり（1 条件のみ）
  - `training.exclude_post_riichi_discards.enabled=true`

### 2.2 共通固定
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- 比較基準:
  - `exp_031` baseline
- model / encoder:
  - `feature_encoder.shanten_hint.enabled=true`
  - `model.hidden_dims=[256,128]`
  - `model.value_features.current_shanten.enabled=true`
  - `model.policy_tower.enabled=false`
  - `model.value_tower.enabled=false`
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
  - `exp_031` baseline のハイパラ、モデル、reward 条件を全固定する
- 何を変えてよいか:
  - `training.exclude_post_riichi_discards.enabled` のみ
- reuse を使わない理由:
  - learner 入力サンプル集合が変わるため、既存 run は比較参照にのみ使う

## 3. 実行方式

### 3.1 実行単位
- batch 実行（1 条件）

### 3.2 既存実験からの流用
- 参照可能な既存 run:
  - `exp_031` baseline
- 流用するもの:
  - 比較参照値
- 新規実行するもの:
  - post-riichi 除外あり 1 条件
- 実データ確認:
  - `exp_031` の report に主要値は転記済み
  - `runs/` 側の実データが残っていれば summary/train_metrics を直接参照してよい
- 再実行が必要な理由:
  - learner の学習対象が変わるため、新規 run が必要

### 3.3 run_map
- `experiments/exp_033/run_map.json` に batch_dir を記録する
- report には `exp_031` 参照 run と今回 run の対応を転記する

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
    model.policy_tower.enabled=false \
    model.value_tower.enabled=false \
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
- `ppo_diag.shanten_diag` に以下が存在すること
  - `post_riichi_discard_count`
  - `post_riichi_discard_ratio`
  - `reward`
  - `delta_t`
  - `advantage`
- `phase_stats.learner.post_riichi_exclusion` が存在すること
- `batch_summary.json.runs[*].post_riichi_exclusion` が存在すること
- `ppo_diag.turn_diag` が存在すること

## 6. 主評価と副評価

### 6.1 主評価
- `exp_031` と比較した以下の差分を最優先で見る
  - `eval_before -> eval` の `avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`
  - `shanten_diag.same.advantage.mean`
  - `shanten_diag.improve.advantage.mean`
  - `shanten_diag.same.post_riichi_discard_ratio`
  - `post_riichi_exclusion.excluded_post_riichi_discards`

### 6.2 副評価
- after 指標（`avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`）
- `shanten_diag` の `reward.mean`, `delta_t.mean`
- `turn_diag` の `advantage.mean`, `value_error.mean`
- 更新安定性（`clip_fraction`, `ratio_std`, `value_error_mean`）

### 6.3 比較優先順
- `same.advantage.mean` の変化
- `eval_before -> eval` の悪化幅
- `improve/same/worsen` の advantage 並び
- after 指標

## 7. 集計方法

- 正本:
  - `runs/<batch>/batch_summary.json`
  - `runs/<batch>/<run>/metrics/train_metrics.json`
  - `runs/<batch>/<run>/summary.json`
- mean/std は seed=5 集約
- `post_riichi_exclusion` は `summary.json.phase_stats.learner` と `batch_summary.json.runs[*]` から読む
- 比較対象 `exp_031` の値は report に並列表で転記する

## 8. 想定リスクと回避

- 実行失敗しやすい箇所:
  - GPU メモリ不足、worker 起動失敗
- 長時間実行時の注意:
  - batch 途中停止時も `run_map.json` と成功 seed 数を記録する
- 交絡要因:
  - 立直後除外以外の override を変えない
- 再開方針:
  - 同条件で再実行し、採用 run のみ report に記載する
- 計算時間見積もり:
  - 約 1.5〜2.5 時間

## 9. レポートに必ず含める項目

- 実行対応表（`exp_031` baseline vs `exp_033`）
- 通常評価（before/after, delta）
- `post_riichi_exclusion` 件数
- `shanten_diag` の主要比較
- `turn_diag` の主要比較
- 結論と次アクション

## 10. 次アクション判定

- 採用判断:
  - `same` 群正偏りが緩み、`eval_before -> eval` の悪化幅も縮むなら追試対象とする
- 却下判断:
  - 除外件数が十分あるのに advantage 構造も通常評価も改善しなければ、立直後混入主因説は弱まる
- 追加診断判断:
  - `same.post_riichi_discard_ratio` が高いのに影響が薄い場合、次は `turn × shanten group` などの分解を検討する
- 次に回すべき実験:
  - 結果次第で `exclude_post_riichi_discards` を baseline 採用するか、target/value 側の別診断へ戻る

## 11. 作成前チェック

- [x] 既存実験との条件重複を確認し、流用可否を判断した
- [x] 参照する既存 run の実データが残っているか、または必要値が `report.md` に転記済みかを確認した
- [x] 再実行する条件について、流用しない理由を明記した
