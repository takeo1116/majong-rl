# exp_030 runbook

最終更新: 2026-03-11  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: 新しく追加した `reward / point_delta_reward / shanten_delta_reward / delta_t` 診断を使って、baseline 条件 1 本で advantage 逆転の原因を分解して読む

---

## 0. この実験の位置づけ

- 直前までの結論:
  - `exp_029` では small model + `current_shanten=true` 条件で `policy tower only` が実用上は最良、`dual towers` は構造的には有望だが通常評価では更新できなかった
  - ただし当時は `shanten_diag` に `reward / point_delta_reward / shanten_delta_reward / delta_t` がなく、advantage 逆転の原因分解はできなかった
- なぜ今この実験をするのか:
  - `CQ-0160/0161` により、advantage 逆転を reward 成分と 1-step TD 誤差まで分解して見られるようになったため
- この実験で更新したい判断:
  - `improve/worsen` の逆転が
    - `point_delta_reward` 側で起きているのか
    - `shanten_delta_reward` では改善しているのか
    - `delta_t`（`reward + gamma * next_value - old_value`）で増幅されているのか
  を baseline 条件 1 本で確認する

## 1. この実験の問い

1. `improve/worsen` 群での `reward` 逆転は、`point_delta_reward` と `shanten_delta_reward` のどちらで生じているか。
2. `delta_t` の符号は `reward` の時点から逆転しているのか、それとも `gamma * next_value - old_value` によって悪化しているのか。
3. `turn_diag` を併せて見たとき、逆転の主因は全巡目共通なのか、特に終盤で強まるのか。

## 2. 実験方針

### 2.1 条件
- A: baseline
  - small model `[256,128]`
  - `current_shanten=true`
  - tower なし

### 2.2 共通固定
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- model / encoder:
  - `FlatFeatureEncoder`
  - `feature_encoder.shanten_hint.enabled=true`
  - `model.hidden_dims=[256,128]`
- 固定 override:
  - `training.imitation_loss_mode=tie_aware_best_set`
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.1`
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
  - reward 条件
  - imitation warm start 条件
  - `current_shanten=true`
  - tower 構造
  - PPO ハイパラ
- 何を変えないか:
  - 今回は比較条件を増やさない
- reuse を使わない理由:
  - `CQ-0160/0161` で追加した `reward / point_delta_reward / shanten_delta_reward / delta_t` を含む `shanten_diag` が必要なため

## 3. 実行方式

### 3.1 実行単位
- batch 実行 + driver

### 3.2 既存実験からの流用
- 参照可能な既存 run:
  - `exp_029` A/B/C/D
- 流用するもの:
  - 方針と旧診断の参照のみ
- 新規実行するもの:
  - A baseline 1 条件のみ
- 実データ確認:
  - `runs/` 配下の既存 run は残っているが、新しい `reward/delta_t` 診断がない
- 再実行が必要な理由:
  - 新診断を含む baseline を取得して、逆転原因の分解を行うため

### 3.3 run_map
- `experiments/exp_030/run_map.json` に batch_dir を記録する
- report に最終対応表を転記する

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

### 5.2 今回の主診断
- `ppo_diag.shanten_diag` に以下があること
  - `reward`
  - `point_delta_reward`
  - `shanten_delta_reward`
  - `delta_t`
- `ppo_diag.turn_diag` があること
- `improve / same / worsen` 群の count が 0 でないこと

## 6. レポートで必ず確認すること

### 6.1 最優先
- `improve/worsen` 群での
  - `reward.mean`
  - `point_delta_reward.mean`
  - `shanten_delta_reward.mean`
  - `delta_t.mean`
  - `return.mean`
  - `old_value.mean`
- 逆転が
  - reward 時点で起きているのか
  - delta 時点で増幅されるのか
  を明示する

### 6.2 副評価
- `turn_diag` の early / mid / late で
  - `reward.mean`
  - `delta_t.mean`
  - `value_error.mean`
  - `advantage.mean`
- `eval_before -> eval`
- after 指標

## 7. 失敗時の扱い
- batch 成功数が 5 未満なら失敗として中止
- `shanten_diag` の新規キーが欠落していれば driver で停止
- ただし driver 検証は brittle にしすぎず、存在確認中心に留める

## 8. 想定所要時間
- 約 1.5〜2.5 時間
