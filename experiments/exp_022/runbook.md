# experiments/exp_022/runbook.md

最終更新: 2026-03-09  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: **`shanten_delta_reward + linear_decay + scale=0.01` を固定し、`mode=both` と `mode=improve_only` を比較して、悪化側 shaping が本当に必要かを切り分ける**

---

## 0. この実験の位置づけ

- 直前までの結論:
  - `exp_020` で `shanten_delta_reward` は有効だった
  - `exp_021` で `linear_decay + scale=0.01` が暫定標準として最も自然だった
  - `scale=0.1` は過剰 shaping で明確に悪化した
- なぜ今この比較をするのか:
  - 次に知りたいのは「改善に正報酬を与えるだけで十分か」、それとも「悪化に負報酬を与える必要があるか」
  - `mode=both` は sparse reward を密にするが、負側 shaping が強すぎると PPO を再び壊す可能性がある
- この実験で更新したい判断:
  - shaping の標準 mode を 1 つ決める
  - その後の value/target 診断の基準 reward 条件を固定する
  - baseline は `exp_021` の A 条件を参照点として流用してよいかを確認する

## 1. この実験の問い

1. `mode=both` と `mode=improve_only` のどちらが主評価で自然か。
2. 負側 shaping を消しても、`point_delta` 単独 baseline より改善が保てるか。
3. learner 診断と reward 内訳を見ると、負側 shaping は補助かノイズか。

## 2. 実験方針

### 2.1 比較軸
- A: 参照 baseline（再実行しない）
  - `exp_021` の A 条件を流用
  - shaping off
- B: `mode=both`
  - `shanten_delta_reward`
  - `scale=0.01`
  - `schedule.type=linear_decay`
- C: `mode=improve_only`
  - `shanten_delta_reward`
  - `scale=0.01`
  - `schedule.type=linear_decay`

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

### 2.3 交絡回避
- 何を固定するか:
  - imitation 方針
  - learner ハイパラ
  - eval 条件
  - shaping の scale と schedule
- 何を変えてよいか:
  - shaping mode のみ
- reuse を使わない理由:
  - reward 条件差が self-play reward 自体を変えるため
  - 各条件で full run が必要
- baseline を流用できる理由:
  - `exp_021` の A 条件は、今回の B/C と比較するための非-shaping 参照点として条件一致している
  - よって baseline を再実行しても新情報がほぼ増えない

## 3. 実行方式

### 3.1 実行単位
- batch + driver
- 今回新規に実行するのは B/C の 2 条件のみ
- baseline A は `exp_021` の A を流用する

### 3.2 run_map
- `experiments/exp_022/run_map.json` をローカル管理
- report には最終的に
  - `condition`
  - `mode`
  - `batch_dir`
  を転記する
- baseline の参照元:
  - `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_07349dda`

## 4. 実行コマンド

参照 baseline:
- `exp_021` 条件 A
- batch_dir: `runs/20260309_stage1_full_flat_mlp_imitation_then_ppo_batch_07349dda`

```bash
# 条件B: mode=both
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
# 条件C: mode=improve_only
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
    'reward.shaping={"shanten_delta":{"enabled":true,"scale":0.01,"mode":"improve_only","schedule":{"type":"linear_decay"}}}' \
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
- 実運用では `scripts/local/exp_022_driver.py` を使う
- `run_map.json` と `driver_logs/` はローカル管理

## 5. 成功判定

### 5.1 共通
- `summary.json.success == true`
- 必須成果物:
  - `summary.json`
  - `notes.md`
  - `config.yaml`
- baseline 参照元 (`exp_021` A) が存在し、比較に必要な成果物が読める

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
    - `point_delta`
    - `shanten_delta`
    - `total`
  - learner 診断
    - `clip_fraction`
    - `ratio_std`
    - `value_error_mean`
    - `value_error_std`

## 7. レポートに必ず含める項目

1. 条件一覧
   - `condition`
   - `mode`
   - `batch_dir`
   - baseline 参照元
2. 主評価表
3. after 指標表
4. reward 内訳表
   - 特に `shanten_delta p90/p99`
   - `total p90/p99`
5. learner 診断表
6. 総合結論
   - `both` を維持するか
   - `improve_only` を標準にするか
   - 次に value/target 診断へ進む基準 reward 条件は何か

## 8. 想定所要時間

- 約 4〜6 時間
- `evaluation.mode=rotation`, `num_matches=30`, 5 seeds, 新規実行 2 条件のため

## 9. 実行後メモ欄

- 実行日時:
- 実行者:
- 備考:
  - override 記法は `reward.shaping=JSON` を使う
  - 深いドット記法は現状の CLI では使わない
