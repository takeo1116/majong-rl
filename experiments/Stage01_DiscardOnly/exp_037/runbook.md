# exp_037 runbook

最終更新: 2026-03-12  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: `exp_035` を高表現力 baseline として、`gae_lambda` 短縮と imitation value warmstart 強化が PPO 安定性と after 指標を改善するかをコード変更なしで確認する

---

## 0. この実験の位置づけ

- 直前までの結論:
  - `exp_035` は `hidden_dims=[512,256] + dual_towers + exclude_post_riichi_discards=true` 条件で、現時点の高表現力 baseline 候補
  - `exp_036` では `batch_size=1024` 系が過度に保守的となり、更新指標は綺麗でも after 指標が大きく悪化した
  - よって、次は batch をさらに触るより、target/value 側に近いハイパラをコード変更なしで確認する段階
- なぜ今この比較をするのか:
  - `exp_035` でもなお PPO 後悪化と turn 依存歪みが一部残っている
  - その原因候補として、GAE の長さと PPO 開始時 critic 初期値の弱さを切り分けたい
- この実験で更新したい判断:
  - `exp_035` を維持したまま、`gae_lambda` と imitation value warmstart を調整すると、after 指標や PPO 安定性がさらに改善するか

## 1. この実験の問い

1. `training.gae_lambda=0.90` は `exp_035` に対して `turn_diag` と `eval_before -> eval` を改善するか。
2. `training.imitation_value_warmstart.coef=0.3` は critic の初期値を改善し、after 指標や `value_error_mean` を改善するか。
3. 両方を同時に入れると、単独条件より良い相乗効果があるか。
4. どの条件が次の高表現力 baseline 候補になるか。

## 2. 実験方針

### 2.1 比較軸
- A: baseline replay
  - `exp_035` と同一
  - `training.gae_lambda=0.95`
  - `training.imitation_value_warmstart.coef=0.1`
- B: shorter GAE
  - `training.gae_lambda=0.90`
  - warmstart coef は `0.1`
- C: stronger imitation value warmstart
  - `training.gae_lambda=0.95`
  - `training.imitation_value_warmstart.coef=0.3`
- D: both
  - `training.gae_lambda=0.90`
  - `training.imitation_value_warmstart.coef=0.3`

### 2.2 共通固定
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- 比較基準:
  - `exp_035`
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
  - `reward.point_delta_scale=0.0001`
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.01`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
  - `imitation.num_workers=10`
  - `selfplay.imitation_matches=25`
  - `training.imitation_epochs=4`
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
  - `training.entropy_coef=0.01`
  - `training.clip_epsilon=0.2`
  - `training.device=cuda`
  - `selfplay.inference_device=cpu`
  - `evaluation.inference_device=cpu`
  - `training.exclude_post_riichi_discards.enabled=true`

### 2.3 交絡回避
- 何を固定するか:
  - `exp_035` の構造・reward・exclusion・batch/lr を固定する
- 何を変えるか:
  - `training.gae_lambda`
  - `training.imitation_value_warmstart.coef`
- reuse を使わない理由:
  - PPO 学習条件が変わるため、新規 run が必要

## 3. 実行方式

### 3.1 実行単位
- batch 実行（4 条件）

### 3.2 既存実験からの流用
- 参照可能な既存 run:
  - `exp_035`
  - `exp_036`
- 流用するもの:
  - 比較参照値
- 新規実行するもの:
  - A/B/C/D の 4 条件
- 実データ確認:
  - `exp_035`, `exp_036` の report に主要値は転記済み
- 再実行が必要な理由:
  - `gae_lambda` と warmstart coef の直接比較が必要

### 3.3 run_map
- `experiments/exp_037/run_map.json` に batch_dir を記録する
- report には `exp_035` 参照 run と今回 run の対応を転記する

## 4. 実行コマンド

```bash
# 条件A: baseline replay
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

```bash
# 条件B: shorter GAE
python3 -m mahjong_rl.cli ... \
  --override ... \
    training.gae_lambda=0.90 \
    training.imitation_value_warmstart.coef=0.1
```

```bash
# 条件C: stronger imitation value warmstart
python3 -m mahjong_rl.cli ... \
  --override ... \
    training.gae_lambda=0.95 \
    training.imitation_value_warmstart.coef=0.3
```

```bash
# 条件D: both
python3 -m mahjong_rl.cli ... \
  --override ... \
    training.gae_lambda=0.90 \
    training.imitation_value_warmstart.coef=0.3
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
  - after 指標: `avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`
  - `clip_fraction`
  - `ratio_std`
  - `eval_before -> eval` の `avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`

### 6.2 副評価
- `value_error_mean`
- `turn_diag.early/mid/late.advantage.mean`
- `shanten_diag.improve/same/worsen.advantage.mean`
- `post_riichi_exclusion.excluded_post_riichi_discards`

### 6.3 比較優先順
- after 指標
- `clip_fraction / ratio_std`
- `eval_before -> eval`
- `value_error_mean`
- `turn_diag`
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
  - 特になし（コード変更なし、`exp_035` の派生条件）
- 長時間実行時の注意:
  - 4 条件 x 5 seeds なので所要時間は長い
- 交絡要因:
  - 構造・batch・lr を固定し、`gae_lambda` と warmstart coef のみを変える
- 再開方針:
  - 条件ごとに batch_dir を `run_map.json` に記録し、中断時は未完条件だけ再実行する
- 計算時間見積もり:
  - `exp_035` と同等の 1 条件あたり 5〜6 時間未満ではなく、今回は 4 条件で合計長時間。夜間実行前提

## 9. レポートに必ず含める項目

- 条件一覧
- 実行対応表
- 主評価表
- 副評価表
- `exp_035` 比較表
- 結論
- 次アクション

## 10. 次アクション判定

- どの結果なら採用:
  - after 指標で `exp_035` を上回り、`clip_fraction / ratio_std` も同等以下
- どの結果なら却下:
  - after 指標が `exp_035` を明確に下回る
- どの結果なら追加診断:
  - after 指標は良いが `eval_before -> eval` または `turn_diag` が悪化する
- 次に回すべき実験:
  - GAE が効けば `0.90 -> 0.92/0.88` の微調整
  - warmstart が効けば coef の再調整
  - 両方ダメなら target/value 設計や turn feature へ戻る

## 11. 作成前チェック

- [x] 既存実験との条件重複を確認し、流用可否を判断した
- [x] 参照する既存 run の実データが残っているか、または必要値が `report.md` に転記済みかを確認した
- [x] 再実行する条件について、流用しない理由を明記した
