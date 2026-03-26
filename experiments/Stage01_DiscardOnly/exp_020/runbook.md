# experiments/exp_020/runbook.md

最終更新: 2026-03-09  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: **reward の sparse さが PPO 悪化の主因かを、`point_delta` 単独と `shanten_delta_reward` 追加条件の比較で直接検証する**

---

## 0. この実験の位置づけ

- 直前までの結論:
  - `shanten_hint=true` と `tie_aware_best_set` は imitation 側では有効
  - ただし PPO を入れると平均では依然として悪化方向
  - `exp_018` / `exp_019` で、`epochs` や `lr` を弱めても主評価は改善せず、更新強度単独は主因でなさそう
  - 現行 reward は `point_delta * 0.0001` で、分布はかなり sparse
- なぜ今この比較をするのか:
  - いま最も強い仮説は「PPO が imitation 方策を壊す主因は reward の sparse さ」
  - その仮説を、最小 shaping reward を追加して前向きに検証する
- この実験で更新したい判断:
  - `shanten_delta_reward` を足すと PPO 悪化は軽減するか
  - shaping は constant のまま残すべきか、後半で decay した方がよいか
  - sparse reward 主因説を前に進めるか、いったん弱めるか

## 1. この実験の問い

1. `point_delta` 単独に対して、`shanten_delta_reward` を追加すると `eval_before -> eval` の悪化幅は改善するか。
2. `constant` shaping と `linear_decay` shaping のどちらが、主評価と after 指標の両方で自然か。
3. reward 内訳統計（`point_delta/shanten_delta/total`）は、shaping 導入によって sparse / tail 構造がどう変わるか。

## 2. 実験方針

### 2.1 比較軸
- A: baseline reward
  - `point_delta` のみ
  - shaping off
- B: shaping constant
  - `point_delta + shanten_delta_reward`
  - `schedule.type=constant`
- C: shaping decay
  - `point_delta + shanten_delta_reward`
  - `schedule.type=linear_decay`

### 2.2 共通固定
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- model / encoder:
  - `feature_encoder.shanten_hint='{"enabled":true}'`
  - `training.imitation_loss_mode=tie_aware_best_set`
  - `hidden_dims=[256,128]`
  - `value_heads=["round_delta"]`
- 固定 override:
  - `imitation.num_workers=10`
  - `selfplay.imitation_matches=25`
  - `training.imitation_epochs=4`
  - `selfplay.num_matches=200`
  - `selfplay.num_workers=10`
  - `selfplay.policy_ratio=1.0`
  - `selfplay.save_baseline_actions=false`
  - `evaluation.mode=rotation`
  - `evaluation.rotation_seats='[0,1,2,3]'`
  - `evaluation.num_matches=30`
  - `evaluation.num_workers=10`
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
  - imitation 方針
  - self-play 規模
  - eval 条件
  - learner ハイパラ
  - model / encoder
- 何を変えてよいか:
  - reward shaping の有無
  - shaping の schedule
- reuse を使う場合の理由:
  - reward 自体は self-play 中に変わるため、A/B/C で self-play は共有できない
  - そのため今回は reuse を使わず、**reward 条件ごとに full batch** を回す
  - 比較ノイズは seeds で吸収する

## 3. 実行方式

### 3.1 実行単位
- batch + driver
- 条件ごとに 5 seeds の full run

### 3.2 reuse を使わない理由
- shaping reward は self-play で生成される sample reward 自体を変える
- 既存 self-play shard を再利用すると reward 条件差が消える
- 今回の目的は reward 条件そのものの比較なので、各条件で full run が必要

### 3.3 run_map
- `experiments/exp_020/run_map.json` をローカル管理
- report には最終的に
  - `condition`
  - `seed`
  - `run_dir`
  を転記する

## 4. 実行コマンド

```bash
# 条件A: baseline reward (shaping off)
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46 \
  --override \
    feature_encoder.shanten_hint='{"enabled":true}' \
    training.imitation_loss_mode=tie_aware_best_set \
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
    evaluation.inference_device=cpu
```

```bash
# 条件B: shanten shaping constant
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46 \
  --override \
    feature_encoder.shanten_hint='{"enabled":true}' \
    training.imitation_loss_mode=tie_aware_best_set \
    imitation.num_workers=10 \
    selfplay.imitation_matches=25 \
    training.imitation_epochs=4 \
    selfplay.num_matches=200 \
    selfplay.num_workers=10 \
    selfplay.policy_ratio=1.0 \
    selfplay.save_baseline_actions=false \
    reward.shaping='{"shanten_delta":{"enabled":true,"scale":0.01,"mode":"both","schedule":{"type":"constant"}}}' \
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
    evaluation.inference_device=cpu
```

```bash
# 条件C: shanten shaping linear decay
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46 \
  --override \
    feature_encoder.shanten_hint='{"enabled":true}' \
    training.imitation_loss_mode=tie_aware_best_set \
    imitation.num_workers=10 \
    selfplay.imitation_matches=25 \
    training.imitation_epochs=4 \
    selfplay.num_matches=200 \
    selfplay.num_workers=10 \
    selfplay.policy_ratio=1.0 \
    selfplay.save_baseline_actions=false \
    reward.shaping='{"shanten_delta":{"enabled":true,"scale":0.01,"mode":"both","schedule":{"type":"linear_decay"}}}' \
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
    evaluation.inference_device=cpu
```

備考:
- 実運用では `scripts/local/exp_020_driver.py` を使う
- `run_map.json` と `driver_logs/` はローカル管理とする

## 5. 成功判定

### 5.1 共通
- `summary.json.success == true`
- 必須成果物:
  - `summary.json`
  - `notes.md`
  - `config.yaml`

### 5.2 評価成果物
- `eval_before/eval_rotation.json`
- `eval/eval_rotation.json`
- `eval/eval_diff.json`
- batch 実行では `batch_summary.json`

### 5.3 追跡キー
- `summary.phase_stats.selfplay.reward_composition`
- `summary.phase_stats.selfplay.reward_shaping`
- `summary.phase_stats.learner.ppo_diag`
- `metrics/train_metrics.json`
- `batch_summary.json.runs[*].reward_composition`
- `batch_summary.json.runs[*].reward_shaping`

## 6. 主評価と副評価

### 6.1 主評価
- `eval_before -> eval` の delta
  - `Δavg_rank`
  - `Δavg_score`
  - `Δdeal_in_rate`
  - `Δwin_rate`

### 6.2 副評価
- after 指標:
  - `avg_rank`
  - `avg_score`
  - `win_rate`
  - `deal_in_rate`
- 補助指標:
  - reward 内訳統計
    - `point_delta`: `mean/std/p50/p90/p99`
    - `shanten_delta`: `mean/std/p50/p90/p99`
    - `total`: `mean/std/p50/p90/p99`
  - self-play 統計
  - learner 診断
    - `clip_fraction`
    - `ratio_mean/std/p90/p99`
    - `advantage_mean/std/p90/p99`
    - `return_mean/std/p90/p99`
    - `value_error_mean/std/p90/p99`

### 6.3 比較優先順
- `Δavg_rank -> Δavg_score -> Δdeal_in_rate -> Δwin_rate`

## 7. 集計方法

- どのファイルを正とするか:
  - 評価: `summary.json` および `eval/*`
  - reward 内訳: `summary.json.phase_stats.selfplay.reward_composition`
  - shaping 設定: `summary.json.phase_stats.selfplay.reward_shaping`
  - learner 診断: `summary.json.phase_stats.learner.ppo_diag`
- mean/std の単位:
  - seed 単位の run 集計
- seed 対応の取り方:
  - `experiments/exp_020/run_map.json` をローカルで保持
  - report には condition × seed × run_dir 対応表を転記
- offline 集計が必要なもの:
  - 原則なし
  - 必要なら `batch_summary.json` と `summary.json` から補助表を作る

## 8. 想定リスクと回避

- 実行失敗しやすい箇所:
  - reward shaping の config キー typo
  - `linear_decay` の挙動誤認
- 長時間実行時の注意:
  - 3 条件 × 5 seeds の full run なので長時間
  - driver で進捗と `run_map.json` を保存する
- 交絡要因:
  - reward 条件差が self-play 分布そのものを変える
  - これは今回の実験目的に含まれるため許容する
- 再開方針:
  - driver ログと `run_map.json` を見て、未完条件だけ手動再実行
- 計算時間見積もり:
  - `exp_018` 相当の 3 条件 full run と同程度
  - 目安 `6〜8時間`

## 9. レポートに必ず含める項目

- 条件一覧（A/B/C の reward 設定）
- 実行対応表
- 主評価表（delta）
- after 指標表
- reward 内訳表
- learner 診断表
- 結論
- 次アクション

## 10. 次アクション判定

- どの結果なら採用:
  - B または C が A より主評価で明確に改善し、after 指標も悪化しない
- どの結果なら却下:
  - B/C とも A を更新できず、reward 内訳だけが変わって評価改善がない
- どの結果なら追加診断:
  - reward 内訳は大きく変わるのに評価が変わらない
  - shaping は効くが constant/decay の優劣が不明
- 次に回すべき実験:
  - shaping 有効なら scale / mode の小規模比較
  - shaping 無効なら reward/target ではなく value target / critic 側の改善へ戻る
