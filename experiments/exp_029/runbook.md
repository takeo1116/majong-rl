# exp_029 runbook

## この実験の位置づけ

- 目的: small model + `current_shanten=true` 条件で、shared trunk 後の task-specific tower が `policy-value` 干渉を弱めるかを確認する。
- 背景:
  - `exp_024`〜`exp_028` で、value 診断改善と通常評価改善が単純には両立しないことが分かった。
  - 特に large model 条件では `value_error` は改善する一方、`clip_fraction` / `ratio_std` が悪化し、通常評価が崩れた。
  - 共有 trunk に policy/value の両タスクを背負わせる構造自体が干渉要因である可能性が高い。
- 今回の狙い:
  - model サイズは small `[256,128]` に戻す
  - `current_shanten=true` は全条件で固定する
  - `policy_tower` / `value_tower` の on/off だけを比較する

## 既存実験からの流用

- 参照可能な既存 run:
  - `exp_024 D`: small model + `current_shanten=true` に近いが、current_shanten 経路と診断成果物の世代が異なるため正式 baseline には使わない
  - `exp_025`: small model 実用基準だが `current_shanten=false`
  - `exp_028`: 更新強度比較であり、tower 構造比較の baseline には使わない
- 流用するもの:
  - 主評価/副評価の見方
  - reward / imitation / PPO の共通設定
- 新規実行するもの:
  - baseline
  - value tower only
  - policy tower only
  - dual towers
- 再実行が必要な理由:
  - `current_shanten=true` を全条件で固定し、tower 構造だけを差分にした baseline を新しく取る必要があるため

## 作成前チェック

- [x] 既存実験との条件重複を確認し、流用可否を判断した
- [x] 今回は baseline を含めて新規取得が必要であることを確認した
- [x] `current_shanten=true` を全条件で固定する方針を確認した
- [x] tower 構造以外の差分を入れないことを確認した

## 仮説

1. `value_tower` を追加すると、`shanten_diag` / `turn_diag` / `value_error` が改善し、通常評価悪化を抑えられる可能性がある。
2. `policy_tower` だけでも共有 trunk の負担を減らし、更新安定性が改善する可能性がある。
3. `dual towers` が最も自然な shared trunk + task-specific towers 構造であり、value 診断改善と通常評価改善の両立候補である。

## 比較条件

### 共通固定条件

- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- encoder:
  - `feature_encoder.shanten_hint.enabled=true`
- reward:
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.01`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
- imitation:
  - `training.imitation_loss_mode=tie_aware_best_set`
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.1`
- model:
  - `model.hidden_dims=[256,128]`
  - `model.value_features.current_shanten.enabled=true`
- rollout / eval:
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
- PPO:
  - `training.epochs=4`
  - `training.lr=0.0001`
  - `training.value_loss_coef=0.25`
  - `training.batch_size=256`
  - `training.gamma=0.99`
  - `training.gae_lambda=0.95`
  - `training.entropy_coef=0.01`
  - `training.clip_epsilon=0.2`
- devices:
  - `training.device=cuda`
  - `selfplay.inference_device=cpu`
  - `evaluation.inference_device=cpu`
- seeds:
  - `42,43,44,45,46`

### A: baseline

- `model.policy_tower.enabled=false`
- `model.value_tower.enabled=false`

### B: value tower only

- `model.policy_tower.enabled=false`
- `model.value_tower.enabled=true`
- `model.value_tower.hidden_dim=128`

### C: policy tower only

- `model.policy_tower.enabled=true`
- `model.policy_tower.hidden_dim=128`
- `model.value_tower.enabled=false`

### D: dual towers

- `model.policy_tower.enabled=true`
- `model.policy_tower.hidden_dim=128`
- `model.value_tower.enabled=true`
- `model.value_tower.hidden_dim=128`

## 実行コマンド

### A: baseline

```bash
python3 -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46 \
  --override \
    feature_encoder.shanten_hint.enabled=true \
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
    model.hidden_dims='[256,128]' \
    model.value_features.current_shanten.enabled=true \
    model.policy_tower.enabled=false \
    model.value_tower.enabled=false
```

### B: value tower only

```bash
python3 -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46 \
  --override \
    feature_encoder.shanten_hint.enabled=true \
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
    model.hidden_dims='[256,128]' \
    model.value_features.current_shanten.enabled=true \
    model.policy_tower.enabled=false \
    model.value_tower.enabled=true \
    model.value_tower.hidden_dim=128
```

### C: policy tower only

```bash
python3 -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46 \
  --override \
    feature_encoder.shanten_hint.enabled=true \
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
    model.hidden_dims='[256,128]' \
    model.value_features.current_shanten.enabled=true \
    model.policy_tower.enabled=true \
    model.policy_tower.hidden_dim=128 \
    model.value_tower.enabled=false
```

### D: dual towers

```bash
python3 -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46 \
  --override \
    feature_encoder.shanten_hint.enabled=true \
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
    model.hidden_dims='[256,128]' \
    model.value_features.current_shanten.enabled=true \
    model.policy_tower.enabled=true \
    model.policy_tower.hidden_dim=128 \
    model.value_tower.enabled=true \
    model.value_tower.hidden_dim=128
```

## 主評価

### 最重要

- `eval_before -> eval` の差分
  - `avg_rank`
  - `avg_score`
  - `win_rate`
  - `deal_in_rate`
- after 指標
  - `avg_rank`
  - `avg_score`

### 構造診断

- `shanten_diag`
  - `improve/same/worsen` ごとの
    - `advantage`
    - `return`
    - `old_value`
    - `new_value`
    - `value_update_delta`
    - `value_error`
- `turn_diag`
  - `early/mid/late` ごとの
    - `advantage`
    - `return`
    - `old_value`
    - `new_value`
    - `value_update_delta`
    - `value_error`
- PPO 更新安定性
  - `clip_fraction`
  - `ratio_std`

## 副評価

- imitation phase
  - `teacher_top1_match_rate`
  - `teacher_best_set_hit_rate`
  - `value_loss`
- reward 内訳
  - `point_delta`
  - `shanten_delta`
  - `total`

## レポートに必ず含める項目

1. 4 条件の model 条件表
   - `policy_tower.enabled`
   - `value_tower.enabled`
   - `current_shanten.enabled`
2. after 指標と `eval_diff`
3. `clip_fraction` / `ratio_std`
4. `shanten_diag` の `improve/worsen` 群比較
5. `turn_diag` の `late` 比較
6. 採用/不採用判断

## 成功判定

- 4 条件すべて `success_count = 5/5`
- `summary.json` に tower 条件が記録されている
- `batch_summary.json.runs[*].model_features` から tower 条件が読める
- `shanten_diag` と `turn_diag` が各条件で取得できる

## 期待する読み方

### value tower only が良い場合

- `value_error` / `shanten_diag` / `turn_diag` が改善
- 通常評価も baseline より良い
- `clip_fraction` / `ratio_std` が大きく悪化しない

→ value 専用変換が干渉緩和の本命候補

### policy tower only が良い場合

- policy 側の task-specific 変換不足が主因候補

### dual towers が良い場合

- shared trunk + task-specific towers が今後の標準構造候補

### どれも baseline を超えない場合

- tower 化だけでは不十分
- 次は target/loss/更新設計側へ戻る

## 実行メモ

- 今回は full run 4 条件
- 予想所要時間は約 5〜6 時間
- 実行時は `scripts/local/exp_029_driver.py` を使う前提

