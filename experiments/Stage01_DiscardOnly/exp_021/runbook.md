# experiments/exp_021/runbook.md

最終更新: 2026-03-09  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: **`shanten_delta_reward + linear_decay` の scale 感度を確認し、reward shaping の実用域と過剰域を切り分ける**

---

## 0. この実験の位置づけ

- 直前までの結論:
  - `exp_020` で `shanten_delta_reward` は有効だった
  - `linear_decay` は `constant` より主評価・after 指標とも自然だった
  - sparse reward 主因説はかなり強まったが、悪化は完全には消えていない
- なぜ今この比較をするのか:
  - 次に知りたいのは「shaping が効くか」ではなく「どの強さまでが自然か」
  - scale が弱すぎると効果が出ず、強すぎると `point_delta` を食って shortcut 化する可能性がある
- この実験で更新したい判断:
  - `linear_decay` shaping の実用 scale を 1 つ決める
  - `0.1` の極端条件で、過剰 shaping の壊れ方を観測する

## 1. この実験の問い

1. `linear_decay` shaping の scale は `0.005 / 0.01 / 0.02` のどこが最も自然か。
2. `scale=0.1` の極端条件では、reward 分布・learner 診断・主評価がどう崩れるか。
3. shaping scale を増やしたとき、主評価改善は単調か、それとも途中で頭打ち/悪化に転じるか。

## 2. 実験方針

### 2.1 比較軸
- A: baseline
  - shaping off
- B: `scale=0.005`
  - `shanten_delta_reward`
  - `schedule.type=linear_decay`
- C: `scale=0.01`
  - `shanten_delta_reward`
  - `schedule.type=linear_decay`
- D: `scale=0.02`
  - `shanten_delta_reward`
  - `schedule.type=linear_decay`
- E: `scale=0.1`
  - `shanten_delta_reward`
  - `schedule.type=linear_decay`
  - 診断用の極端条件

### 2.2 共通固定
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- model / encoder:
  - `feature_encoder.shanten_hint='{"enabled":true}'`
  - `training.imitation_loss_mode=tie_aware_best_set`
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
  - shaping on 条件では
    - `reward.shaping.shanten_delta.enabled=true`
    - `reward.shaping.shanten_delta.mode=both`
    - `reward.shaping.shanten_delta.schedule.type=linear_decay`

### 2.3 交絡回避
- 何を固定するか:
  - imitation 方針
  - learner ハイパラ
  - eval 条件
  - shaping の schedule と mode
- 何を変えてよいか:
  - shaping scale のみ
- reuse を使わない理由:
  - reward 条件差が self-play reward 自体を変えるため
  - 各条件で full run が必要

## 3. 実行方式

### 3.1 実行単位
- batch + driver
- 条件ごとに 5 seeds の full run

### 3.2 run_map
- `experiments/exp_021/run_map.json` をローカル管理
- report には最終的に
  - `condition`
  - `scale`
  - `batch_dir`
  を転記する

## 4. 実行コマンド

```bash
# 条件A: baseline
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
# 条件B: scale=0.005
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
    'reward.shaping={"shanten_delta":{"enabled":true,"scale":0.005,"mode":"both","schedule":{"type":"linear_decay"}}}' \
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
# 条件C: scale=0.01
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
    'reward.shaping={"shanten_delta":{"enabled":true,"scale":0.01,"mode":"both","schedule":{"type":"linear_decay"}}}' \
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
# 条件D: scale=0.02
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
    'reward.shaping={"shanten_delta":{"enabled":true,"scale":0.02,"mode":"both","schedule":{"type":"linear_decay"}}}' \
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
# 条件E: scale=0.1 (極端条件)
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
    'reward.shaping={"shanten_delta":{"enabled":true,"scale":0.1,"mode":"both","schedule":{"type":"linear_decay"}}}' \
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
- 実運用では `scripts/local/exp_021_driver.py` を使う
- `run_map.json` と `driver_logs/` はローカル管理

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
- `batch_summary.json`

### 5.3 追跡キー
- `summary.phase_stats.selfplay.reward_composition`
- `summary.phase_stats.selfplay.reward_shaping`
- `summary.phase_stats.learner.ppo_diag`
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
  - reward 内訳
    - `point_delta/shanten_delta/total` の `mean/std/p50/p90/p99`
  - learner 診断
    - `clip_fraction`
    - `ratio_std`
    - `value_error_mean/std`
  - `reward_shaping` 設定の確認

### 6.3 比較優先順
- `Δavg_rank -> Δavg_score -> Δdeal_in_rate -> Δwin_rate`

## 7. 集計方法

- どのファイルを正とするか:
  - 評価: `summary.json` と `eval/*`
  - reward 内訳: `summary.phase_stats.selfplay.reward_composition`
  - shaping 設定: `summary.phase_stats.selfplay.reward_shaping`
  - learner 診断: `summary.phase_stats.learner.ppo_diag`
- mean/std の単位:
  - seed 単位
- `0.1` 条件の扱い:
  - 実用候補ではなく、過剰 shaping の境界確認として解釈する

## 8. 想定リスクと回避

- 実行失敗しやすい箇所:
  - `reward.shaping` の JSON override 記法
- 長時間実行時の注意:
  - 5 条件 × 5 seeds の full run で長時間
  - driver を前提とする
- 交絡要因:
  - scale が self-play 分布そのものを変える
  - これは今回の実験目的に含まれるため許容
- 再開方針:
  - driver ログと `run_map.json` を見て未完条件だけ再実行
- 計算時間見積もり:
  - `exp_020` より長い
  - 目安 `9〜12時間`

## 9. レポートに必ず含める項目

- 条件一覧（scale 一覧）
- 実行対応表
- 主評価表
- after 指標表
- reward 内訳表
- learner 診断表
- `0.1` 条件の診断メモ
- 結論

## 10. 次アクション判定

- どの結果なら採用:
  - baseline より主評価が一貫して良く、after 指標も悪化しない scale
- どの結果なら却下:
  - baseline を更新できない scale
- どの結果なら追加診断:
  - scale を上げるほど reward 分布は変わるが、主評価が単調に動かない
  - `0.1` だけ極端に壊れ、境界確認としてのみ価値がある
- 次に回すべき実験:
  - 最良 scale を固定して `mode=both` vs `mode=improve_only`
