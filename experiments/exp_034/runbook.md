# exp_034 runbook

最終更新: 2026-03-12  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: 立直後打牌除外を維持したまま表現力を上げたとき、PPO 後悪化と診断指標が改善するかを確認する

---

## 0. この実験の位置づけ

- 直前までの結論:
  - `exp_031` で reward scale 修正後 baseline を再取得した
  - `exp_032` の post-fix `policy_tower_only` は baseline を更新できなかった
  - `exp_033` で `exclude_post_riichi_discards=true` により PPO 後悪化はかなり縮んだが、`same > 0 / improve < 0` は残った
- なぜ今この比較をするのか:
  - learner ノイズの一部を減らした条件でも本丸が残ったため、次にモデル表現力不足の可能性を切り分けたい
- この実験で更新したい判断:
  - `exclude_post_riichi_discards=true` を前提に、より大きい trunk + dual towers が改善に効くか

## 1. この実験の問い

1. `exp_033` に対して `hidden_dims=[512,256] + dual_towers` を入れると、`eval_before -> eval` の悪化幅はさらに縮むか。
2. `same > 0 / improve < 0` の advantage 構造は、表現力強化で緩和するか。
3. `turn_diag` の early/mid/late の歪みは、より表現力の高いモデルで改善するか。

## 2. 実験方針

### 2.1 比較軸
- A: high-capacity dual towers（1 条件のみ）
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
  - `exp_033` exclude post-riichi baseline
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
  - `training.exclude_post_riichi_discards.enabled=true`

### 2.3 交絡回避
- 何を固定するか:
  - `exp_033` の reward, learner, exclusion 条件を全固定する
- 何を変えるか:
  - モデル構造のみ
- reuse を使わない理由:
  - trunk / head 構造変更により imitation, selfplay, learner の全出力が変わるため

## 3. 実行方式

### 3.1 実行単位
- batch 実行（1 条件）

### 3.2 既存実験からの流用
- 参照可能な既存 run:
  - `exp_033` exclude post-riichi
  - 参考として `exp_029` dual towers, `exp_026`/`exp_027` の大きいモデル系列
- 流用するもの:
  - 比較参照値
- 新規実行するもの:
  - post-riichi exclusion + high-capacity dual towers 1 条件
- 実データ確認:
  - `exp_033` の report に主要値は転記済み
- 再実行が必要な理由:
  - post-fix かつ exclusion あり条件で dual towers を直接確認していないため

### 3.3 run_map
- `experiments/exp_034/run_map.json` に batch_dir を記録する
- report には `exp_033` 参照 run と今回 run の対応を転記する

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
- `summary.model_features.value_features.current_shanten.enabled == true`
- `phase_stats.learner.post_riichi_exclusion` が存在すること
- `ppo_diag.shanten_diag` が存在すること
- `ppo_diag.turn_diag` が存在すること

## 6. 主評価と副評価

### 6.1 主評価
- `exp_033` と比較した以下の差分を最優先で見る
  - `eval_before -> eval` の `avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`
  - `shanten_diag.same.advantage.mean`
  - `shanten_diag.improve.advantage.mean`
  - `turn_diag.early/mid/late.advantage.mean`

### 6.2 副評価
- after 指標（`avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`）
- 更新安定性（`clip_fraction`, `ratio_std`, `value_error_mean`）
- `post_riichi_exclusion.excluded_post_riichi_discards`
- imitation 指標（teacher 再現率, imitation value_loss）

### 6.3 比較優先順
- `eval_before -> eval` の悪化幅
- `same / improve` の advantage 構造
- `turn_diag` の歪み
- after 指標

## 7. 集計方法

- 正本:
  - `runs/<batch>/batch_summary.json`
  - `runs/<batch>/<run>/metrics/train_metrics.json`
  - `runs/<batch>/<run>/summary.json`
- mean/std は seed=5 集約
- `exp_033` の値は report に並列表で転記する

## 8. 想定リスクと回避

- 実行失敗しやすい箇所:
  - モデル大型化による GPU メモリ不足
  - self-play / evaluation の長時間化
- 長時間実行時の注意:
  - 途中停止時も `run_map.json` に採用 batch_dir を記録する
- 交絡要因:
  - exclusion や reward 条件は変えない
- 再開方針:
  - 同条件で再実行し、採用 run のみ report に記載する
- 計算時間見積もり:
  - 約 2.0〜3.5 時間

## 9. レポートに必ず含める項目

- 実行対応表（`exp_033` vs `exp_034`）
- 通常評価（before/after, delta）
- `post_riichi_exclusion` 件数
- `shanten_diag` の主要比較
- `turn_diag` の主要比較
- 結論と次アクション

## 10. 次アクション判定

- 採用判断:
  - `eval_before -> eval` と診断値の両方が改善し、after 指標も baseline 同等以上なら追試対象とする
- 却下判断:
  - 表現力を上げても `same > 0 / improve < 0` と PPO 後悪化がほぼ変わらないなら、主因は別とみなす
- 追加診断判断:
  - 更新安定性は良化するが after が伸びない場合、target/value 側へ戻る
- 次に回すべき実験:
  - 結果次第で `exclude_post_riichi_discards=true` を固定したまま別構造や target 側改善を試す

## 11. 作成前チェック

- [x] 既存実験との条件重複を確認し、流用可否を判断した
- [x] 参照する既存 run の実データが残っているか、または必要値が `report.md` に転記済みかを確認した
- [x] 再実行する条件について、流用しない理由を明記した
