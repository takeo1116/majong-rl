# exp_031 runbook

最終更新: 2026-03-11  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: CQ-0162（reward scale 適用バグ修正）後の baseline を 1 条件で再取得し、`reward / point_delta_reward / shanten_delta_reward / delta_t` の単位整合と逆転挙動を再確認する

---

## 0. この実験の位置づけ

- 直前までの結論:
  - `exp_030` で advantage 逆転の主因分解を実施したが、同時に `point_delta_scale` 未適用バグを検出した
  - CQ-0162 で self-play / imitation / eval すべての `Stage1Env` に reward config を注入する修正が入った
- なぜ今この比較をするのか:
  - 旧結果は報酬単位が想定と異なる可能性があるため、まず修正後 baseline を 1 本取り直す必要がある
- この実験で更新したい判断:
  - 修正後の単位で見ても `improve/worsen` の逆転が継続するか
  - `point_delta_reward.mean` の桁が期待スケール（`point_delta_scale=0.0001`）に整合するか

## 1. この実験の問い

1. 修正後 baseline で `point_delta_reward` のスケールは期待通りに縮小するか。
2. `improve/worsen` の `reward.mean` / `delta_t.mean` の逆転は、単位修正後も残るか。
3. `eval_before -> eval` と after 指標は、修正前 baseline（exp_030）からどう変化するか。

## 2. 実験方針

### 2.1 比較軸
- A: post-fix baseline（1 条件のみ）
  - `model.hidden_dims=[256,128]`
  - `model.policy_tower.enabled=false`
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
  - `model.policy_tower.enabled=false`
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
  - exp_030 baseline の条件をそのまま固定
- 何を変えるか:
  - 実質的にはコード修正（CQ-0162）だけ
- reuse を使わない理由:
  - バグ修正後の新データが必要であり、既存 run を流用すると修正効果を検証できない

## 3. 実行方式

### 3.1 実行単位
- batch 実行（1 条件）

### 3.2 既存実験からの流用
- 参照可能な既存 run:
  - `exp_030` A baseline only
- 流用するもの:
  - 比較参照値（報告値）
- 新規実行するもの:
  - post-fix baseline 1 条件
- 実データ確認:
  - `runs/` 配下は残っていても、exp_030 は pre-fix データ
- 再実行が必要な理由:
  - reward scale 不整合バグ修正後の基準値を取得するため

### 3.3 run_map
- `experiments/exp_031/run_map.json` に batch_dir を記録
- report には `exp_030` 対比表とあわせて転記

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

## 6. 主評価と副評価

### 6.1 主評価
- `shanten_diag.improve/worsen` の以下を最優先で比較
  - `point_delta_reward.mean`
  - `reward.mean`
  - `delta_t.mean`
  - `advantage.mean`

### 6.2 副評価
- `eval_before -> eval` の delta
- after 指標（`avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`）
- 更新安定性（`clip_fraction`, `ratio_std`, `value_error_mean`）

### 6.3 比較優先順
- `point_delta_reward.mean`（桁確認）
- `reward/delta_t/advantage` の improve-worsen 関係
- `eval_before -> eval`
- after 指標

## 7. 集計方法

- 正本:
  - `runs/<batch>/batch_summary.json`
  - `runs/<batch>/<run>/metrics/train_metrics.json`
  - `runs/<batch>/<run>/summary.json`
- mean/std は seed=5 集約
- `exp_030` との差分は report で並列表にする

## 8. 想定リスクと回避

- 実行失敗しやすい箇所:
  - GPU メモリ不足、worker 起動失敗
- 長時間実行時の注意:
  - 途中停止時は batch_dir を記録し、成功 seed 数を確認
- 交絡要因:
  - 条件差分が reward scale 以外に混入しないよう override を固定
- 再開方針:
  - 失敗時は同一条件で再実行、report には採用 run のみ記載
- 計算時間見積もり:
  - 約 1.5〜2.5 時間

## 9. レポートに必ず含める項目

- 実行対応表（exp_031 run と exp_030 参照 run）
- 通常評価（before/after, delta）
- `shanten_diag` の主要比較（improve/same/worsen）
- `turn_diag` の主要比較
- 単位整合の確認結果
- 結論と次アクション

## 10. 次アクション判定

- 採用判断:
  - `point_delta_reward` の桁が期待通りに縮小し、診断値の解釈が可能になったら次段へ進む
- 追加診断判断:
  - 逆転が残る場合は、修正後 baseline を新しい参照点として `policy_tower_only` を追試
- 再検証判断:
  - もし桁が未整合なら、reward config 伝播経路を再点検

## 11. 作成前チェック

- [x] 既存実験との条件重複を確認し、流用可否を判断した
- [x] 参照 run の実データ有無と、pre-fix / post-fix の違いを確認した
- [x] 再実行する理由（バグ修正後 baseline 取得）を明記した
