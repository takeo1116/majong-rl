# exp_038 runbook

最終更新: 2026-03-13  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: `exp_037 D` を高表現力 baseline として、(1) imitation データ増量で `eval_before` が伸びるか、(2) post-fix 条件で shaping 設定を再評価すると after 指標が改善するかを同時に確認する

---

## 0. この実験の位置づけ

- 直前までの結論:
  - `exp_035` は高表現力モデル側の有力 baseline だった
  - `exp_037` で `gae_lambda=0.90` と `imitation_value_warmstart.coef=0.3` の組み合わせ（D）が総合では最良候補になった
  - ただし `exp_037 D` も決定打ではなく、表現の伸びしろと shaping の再評価が未確認
- なぜ今この比較をするのか:
  - 現行特徴量/モデルにそもそも imitation だけでまだ伸びしろがあるかを見たい
  - shaping の過去結論は reward scale バグの影響を強く受けていたため、post-fix 条件で再評価したい
- この実験で更新したい判断:
  - `exp_037 D` を基準に、次に優先すべき方向が「データ量増加」か「shaping 再設定」かを決める

## 1. この実験の問い

1. imitation データ量を増やすと `eval_before` は伸びるか。
2. `reward.shaping.shanten_delta.mode=improve_only` は `mode=both` より良いか。
3. `reward.shaping.shanten_delta.scale=0.02` は `0.01` より良いか。
4. shaping 設定の変更と比べて、imitation 増量のほうが改善余地として有望か。

## 2. 実験方針

### 2.1 比較軸
- A: baseline reference（reuse / 参照のみ）
  - `exp_037 D`
  - `training.gae_lambda=0.90`
  - `training.imitation_value_warmstart.coef=0.3`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.scale=0.01`
- B: imitation 増量
  - A + `selfplay.imitation_matches=50`
  - A + `training.imitation_epochs=8`
- C: shaping improve_only
  - A + `reward.shaping.shanten_delta.mode=improve_only`
- D: shaping scale 0.02
  - A + `reward.shaping.shanten_delta.scale=0.02`
- E: shaping improve_only + scale 0.02
  - A + `reward.shaping.shanten_delta.mode=improve_only`
  - A + `reward.shaping.shanten_delta.scale=0.02`

### 2.2 共通固定
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- 比較基準:
  - A は `exp_037 D`
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
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
  - `imitation.num_workers=10`
  - `selfplay.imitation_matches=25`（Bのみ 50）
  - `training.imitation_epochs=4`（Bのみ 8）
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
  - `exp_037 D` の構造・update・exclusion 条件を固定する
- 何を変えるか:
  - B は imitation データ量だけ
  - C/D/E は shaping mode / scale だけ
- 流用を使う理由:
  - A は既存最有力条件であり、新規実行不要

## 3. 実行方式

### 3.1 実行単位
- A: reuse / 参照のみ
- B/C/D/E: batch 実行（新規 4 条件）

### 3.2 既存実験からの流用
- 参照可能な既存 run:
  - `exp_037 D`
- 流用するもの:
  - A baseline の参照値
- 新規実行するもの:
  - B/C/D/E の 4 条件
- 実データ確認:
  - `exp_037` の report に主要値は転記済み
  - `runs/20260312_stage1_full_flat_mlp_imitation_then_ppo_batch_1d5383df` が残っていることを前提にする
  - 万一 `runs/` 実データが消えても、report 転記値で baseline 比較は継続可能
- 再実行が必要な理由:
  - imitation 増量と shaping 再設定は、`exp_037 D` からの直接比較が必要

### 3.3 run_map
- `experiments/exp_038/run_map.json` に B/C/D/E の batch_dir を記録する
- A は `reference_batch_dir` として併記する
- report には `exp_037 D` と B/C/D/E の対応を転記する

## 4. 実行コマンド

```bash
# 条件B: imitation 増量
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
    selfplay.imitation_matches=50 \
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

```bash
# 条件C: shaping improve_only
python3 -m mahjong_rl.cli ... \
  --override ... \
    reward.shaping.shanten_delta.scale=0.01 \
    reward.shaping.shanten_delta.mode=improve_only
```

```bash
# 条件D: shaping scale 0.02
python3 -m mahjong_rl.cli ... \
  --override ... \
    reward.shaping.shanten_delta.scale=0.02 \
    reward.shaping.shanten_delta.mode=both
```

```bash
# 条件E: shaping improve_only + scale 0.02
python3 -m mahjong_rl.cli ... \
  --override ... \
    reward.shaping.shanten_delta.scale=0.02 \
    reward.shaping.shanten_delta.mode=improve_only
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
- B は `eval_before` を最重要とする
  - imitation 増量で `eval_before` が伸びるか
- C/D/E は after 指標を最重要とする
  - `avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`
- 全条件共通で `exp_037 D` との差分も併記する

### 6.2 副評価
- `eval_before -> eval`
- `clip_fraction`
- `ratio_std`
- `value_error_mean`
- `turn_diag.early/mid/late.advantage.mean`
- `shanten_diag.improve/same/worsen.advantage.mean`

### 6.3 比較優先順
- B: `eval_before -> after 指標 -> eval_before→eval -> learner_diag`
- C/D/E: `after 指標 -> eval_before→eval -> learner_diag -> shanten_diag`

## 7. 集計方法

- 正本:
  - `runs/<batch>/batch_summary.json`
  - `runs/<batch>/<run>/metrics/train_metrics.json`
  - `runs/<batch>/<run>/summary.json`
- mean/std は seed=5 集約
- A baseline は `exp_037/report.md` から転記、必要なら run 実データも参照

## 8. 想定リスクと回避

- 実行失敗しやすい箇所:
  - B は imitation データ量増加で時間増
- 長時間実行時の注意:
  - 夜間実行前提。B が最も長い
- 交絡要因:
  - B は imitation 条件だけ、C/D/E は shaping 条件だけを変える
- 再開方針:
  - 条件ごとに batch_dir を `run_map.json` に記録し、中断時は未完条件だけ再実行する
- 計算時間見積もり:
  - A は流用
  - B/C/D/E の 4 条件で長時間。B が基準より重く、C/D/E は `exp_037 D` と同程度

## 9. レポートに必ず含める項目

- 条件一覧
- A baseline との比較表
- B の `eval_before` 比較
- C/D/E の shaping 比較表
- 主評価表
- 副評価表
- 結論
- 次アクション

## 10. 次アクション判定

- どの結果なら採用:
  - B: `eval_before` が明確に改善し、after 指標も悪化しない
  - C/D/E: after 指標で A を上回る
- どの結果なら却下:
  - A に対して主要指標が全面悪化
- どの結果なら追加診断:
  - `eval_before` は伸びるが after が伸びない
  - shaping で `shanten_diag` は良いが after が悪い
- 次に回すべき実験:
  - B が効けば imitation さらなる増量 or self-play 増量
  - shaping が効けば mode/scale の微調整
  - 両方ダメならモデル/特徴量限界か target/value 設計へ戻る

## 11. 作成前チェック

- [x] 既存実験との条件重複を確認し、流用可否を判断した
- [x] 参照する既存 run の実データが残っているか、または必要値が `report.md` に転記済みかを確認した
- [x] 再実行する条件について、流用しない理由を明記した
