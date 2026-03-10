# Experiment Runbook: exp_027

作成日: 2026-03-10  
対象: `exp_026` の延長として、value 表現を強めた状態で hidden_dims をさらに拡大したときの傾向を確認する

## 1. この実験の位置づけ

目的は、`exp_026` で見えた

- 通常評価は悪化した
- しかし `shanten_diag` / `turn_diag` / `value_error` は改善した

という結果を受けて、**value 表現をさらに強くしたときに同じ改善傾向が続くか** を確認すること。

今回は「純粋な hidden_dims 効果の分離」よりも、  
**`current_shanten` を含む強化版 value 表現のスケーリング傾向** を見ることを優先する。

## 2. 既存実験からの流用

- 参照可能な既存 run:
  - `exp_025`: 小モデル, `current_shanten=false`
  - `exp_026`: 中モデル `[512,256]`, `current_shanten=true`
- 流用するもの:
  - 比較参照点として `exp_025` と `exp_026` の report / run 結果を使う
- 新規実行するもの:
  - A: `[768,384]` + `current_shanten=true`
  - B: `[1024,512]` + `current_shanten=true`
- 再実行が必要な理由:
  - 新しい hidden_dims 条件の結果は未取得

## 3. 比較条件

共通固定:

- `feature_encoder.shanten_hint={"enabled":true}`
- `training.imitation_loss_mode=tie_aware_best_set`
- reward:
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.01`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
- imitation:
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.1`
- model:
  - `model.value_features.current_shanten.enabled=true`

新規条件:

- A: hidden_dims `[768,384]`
- B: hidden_dims `[1024,512]`

参照条件:

- `exp_025`: `[256,128]`, `current_shanten=false`
- `exp_026`: `[512,256]`, `current_shanten=true`

## 4. 実行条件

- seeds: `42,43,44,45,46`
- phases: `imitation,selfplay,learner,eval`
- evaluation: `rotation`
- `evaluation.rotation_seats=[0,1,2,3]`
- `evaluation.num_matches=30`
- `selfplay.num_matches=200`

## 5. コマンド例

### A: `[768,384]`

```bash
python3 -m mahjong_rl.cli \
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
    evaluation.inference_device=cpu \
    reward.shaping.shanten_delta.enabled=true \
    reward.shaping.shanten_delta.scale=0.01 \
    reward.shaping.shanten_delta.mode=both \
    reward.shaping.shanten_delta.schedule.type=linear_decay \
    training.imitation_value_warmstart.enabled=true \
    training.imitation_value_warmstart.coef=0.1 \
    model.hidden_dims='[768,384]' \
    model.value_features.current_shanten.enabled=true
```

### B: `[1024,512]`

`model.hidden_dims='[1024,512]'` 以外は A と同じ。

## 6. 主確認項目

### 6.1 通常評価

`exp_025` / `exp_026` と比較して以下を確認する。

- `eval_before -> eval` の `avg_rank/avg_score/win_rate/deal_in_rate`
- after 指標

### 6.2 shanten_diag

以下を `improve/same/worsen` ごとに比較する。

- `advantage.mean`
- `return.mean`
- `old_value.mean`
- `new_value.mean`
- `value_update_delta.mean`
- `value_error.mean`

特に見たい点:

- `improve.advantage.mean` がさらに 0 に近づくか
- `worsen.advantage.mean` がさらに 0 に近づくか
- `improve.value_error.mean` が `exp_026` より下がるか

### 6.3 turn_diag

以下を `early/mid/late` ごとに比較する。

- `advantage.mean`
- `return.mean`
- `old_value.mean`
- `new_value.mean`
- `value_update_delta.mean`
- `value_error.mean`

特に見たい点:

- `late.value_error.mean` が `exp_026` より下がるか
- `late.advantage.mean` が `exp_026` より改善するか

### 6.4 learner 補助指標

- `clip_fraction`
- `ratio_std`
- `value_error_mean`

ここで、診断改善と更新不安定化のトレードオフが強まるかを確認する。

## 7. 成功判定

- 各 batch run が `5/5 success`
- `summary.json.success=true`
- `summary.json.phase_stats.learner.ppo_diag.shanten_diag` が存在する
- `summary.json.phase_stats.learner.ppo_diag.turn_diag` が存在する
- `summary.json.model_features.value_features.current_shanten.enabled=true` を確認できる
- `NaN` / 欠落で集計不能になっていない

## 8. レポートで答える問い

1. value 表現をさらに大きくすると、`shanten_diag` / `turn_diag` の改善傾向は継続するか。  
2. その改善は通常評価にも追随するか。  
3. `exp_026` で見えた「診断改善と通常評価悪化の乖離」は縮まるか、それとも広がるか。  
4. 次に進むべきは、さらに大きくする方向か、target/value loss 設計を触る方向か。

## 9. 作成前チェック

- [x] 既存実験との条件重複を確認し、流用可否を判断した
- [x] `exp_025` / `exp_026` を参照点として流用する
- [x] 新規実行が必要なのは新 hidden_dims 条件のみ

## 10. 運用メモ

- 今回は新規 2 条件のみを実行する。  
- 比較参照は `exp_025` / `exp_026` とし、新規 baseline は回さない。  
- この run で診断改善が継続しても通常評価が改善しない場合、表現力不足は一因だが本丸ではないと判断しやすくなる。
