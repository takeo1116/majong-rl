# Experiment Runbook: exp_028

作成日: 2026-03-10  
対象: `exp_027 A` の value 診断改善を維持したまま、PPO 更新強度を弱めて通常評価悪化を抑えられるかを確認する

## 1. この実験の位置づけ

`exp_027` では、`[768,384] + current_shanten=true` で

- `shanten_diag` / `turn_diag` / global `value_error` は改善した
- しかし通常評価は `exp_025` / `exp_026` より悪化した
- `clip_fraction` と `ratio_std` は悪化した

という結果になった。

したがって次は、**value 診断改善と policy 更新安定性のトレードオフ** を切るため、
`exp_027 A` を固定しつつ PPO 更新強度だけを弱める。

## 2. 既存実験からの流用

- 参照可能な既存 run:
  - `exp_025`: 小モデル基準点
  - `exp_027` A: `[768,384] + current_shanten=true`
- 流用するもの:
  - 比較参照点として `exp_025` と `exp_027` A を使う
- 新規実行するもの:
  - A: weak-lr (`training.lr=5e-5`)
  - B: weak-epochs (`training.epochs=2`)
- 再実行が必要な理由:
  - `exp_027 A` と同条件で更新強度だけを落とした結果は未取得

## 3. 比較条件

固定条件:

- `feature_encoder.shanten_hint.enabled=true`
- `training.imitation_loss_mode=tie_aware_best_set`
- reward shaping 標準
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.01`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
- `training.imitation_value_warmstart.enabled=true`
- `training.imitation_value_warmstart.coef=0.1`
- `model.hidden_dims=[768,384]`
- `model.value_features.current_shanten.enabled=true`

新規条件:

- A: weak-lr
  - `training.lr=5e-5`
  - `training.epochs=4`
- B: weak-epochs
  - `training.lr=1e-4`
  - `training.epochs=2`

参照条件:

- `exp_027` A
  - `training.lr=1e-4`
  - `training.epochs=4`

## 4. 実行条件

- seeds: `42,43,44,45,46`
- phases: `imitation,selfplay,learner,eval`
- evaluation: `rotation`
- `evaluation.rotation_seats=[0,1,2,3]`
- `evaluation.num_matches=30`
- `selfplay.num_matches=200`

## 5. コマンド例

### A: weak-lr

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
    training.lr=0.00005 \
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

### B: weak-epochs

`training.epochs=2`, `training.lr=0.0001` 以外は A と同じ。

## 6. 主確認項目

### 6.1 通常評価

- `eval_before -> eval` の `avg_rank/avg_score/win_rate/deal_in_rate`
- after 指標

最重要:
- `exp_027 A` より通常評価悪化が小さくなるか
- `exp_025` に近づくか

### 6.2 policy 更新安定性

- `clip_fraction`
- `ratio_std`
- `ratio_p90/p99`

最重要:
- `clip_fraction` が `exp_027 A` より下がるか
- `ratio_std` が `exp_027 A` より下がるか

### 6.3 value 診断の維持

以下が `exp_027 A` から極端に悪化しないことを確認する。

- `shanten_diag.improve.value_error.mean`
- `shanten_diag.worsen.value_error.mean`
- `turn_diag.late.value_error.mean`
- `turn_diag.late.advantage.mean`

意図:
- value 診断改善を維持したまま、policy 更新安定性だけ改善できるかを見る

## 7. 成功判定

- 各 batch run が `5/5 success`
- `summary.json.success=true`
- `summary.json.phase_stats.learner.ppo_diag.shanten_diag` が存在する
- `summary.json.phase_stats.learner.ppo_diag.turn_diag` が存在する
- `summary.json.model_features.value_features.current_shanten.enabled=true` を確認できる
- `NaN` / 欠落で集計不能になっていない

## 8. レポートで答える問い

1. `exp_027 A` の value 診断改善は、PPO 更新強度を弱めることで通常評価改善に繋げられるか。  
2. `weak-lr` と `weak-epochs` のどちらが、`clip_fraction` / `ratio_std` をより自然に下げるか。  
3. 更新強度を弱めても通常評価が改善しないなら、次に疑うべきは PPO update 以外（例: target 定義や policy-value 干渉）か。  

## 9. 作成前チェック

- [x] 既存実験との条件重複を確認し、流用可否を判断した
- [x] `exp_027` A を参照点として流用する
- [x] 新規実行が必要なのは更新強度を変えた 2 条件のみ

## 10. 運用メモ

- 参照 baseline は `exp_027 A` とし、新規 baseline は回さない。  
- 今回は value 表現を固定し、PPO 更新強度だけを触る。  
- ここで改善が見えない場合、次は更新強度より deeper な target / policy-value 干渉を疑う。
