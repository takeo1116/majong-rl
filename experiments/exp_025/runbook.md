# Experiment Runbook: exp_025

作成日: 2026-03-10  
対象: `exp_024` の採用候補 B 条件の単条件診断

## 1. この実験の位置づけ

目的は、`exp_024` で暫定採用候補になった B 条件を固定し、最近追加した learner 診断を早期確認することです。  
今回は比較実験ではなく **単条件の診断 run** とします。

診断したい主仮説:
- `shanten_improve` / `shanten_worsen` で advantage が逆向きなのは、`return` 自体ではなく `old_value` / `new_value` 側の系統誤差ではないか
- そのズレは巡目帯（`early/mid/late`）で強さが違うのではないか

## 2. 参照条件

参照元:
- `experiments/exp_024/report.md`
- 採用候補条件: B

固定する条件:
- `feature_encoder.shanten_hint={"enabled":true}`
- `training.imitation_loss_mode=tie_aware_best_set`
- `reward.shaping.shanten_delta.enabled=true`
- `reward.shaping.shanten_delta.scale=0.01`
- `reward.shaping.shanten_delta.mode=both`
- `reward.shaping.shanten_delta.schedule.type=linear_decay`
- `training.imitation_value_warmstart.enabled=true`
- `training.imitation_value_warmstart.coef=0.1`
- `model.value_features.current_shanten.enabled=false`

## 3. 実行条件

- seeds: `42,43,44,45,46`
- phases: `imitation,selfplay,learner,eval`
- evaluation: `rotation`
- `evaluation.rotation_seats=[0,1,2,3]`
- `evaluation.num_matches=30`
- `selfplay.num_matches=200`

## 4. コマンド

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
    model.value_features.current_shanten.enabled=false
```

## 5. 主確認項目

### 5.1 shanten_diag

以下を `improve/same/worsen` ごとに確認する。

- `advantage.mean/std/p50/p90/p99`
- `advantage.positive_ratio`
- `advantage.negative_ratio`
- `return.mean/std/p50/p90/p99`
- `old_value.mean/std/p50/p90/p99`
- `new_value.mean/std/p50/p90/p99`
- `value_update_delta.mean/std/p50/p90/p99`
- `value_error.mean/std/p50/p90/p99`
- `count`
- `available_samples`
- `unavailable_samples`

### 5.2 turn_diag

以下を `early/mid/late` ごとに確認する。

- `advantage`
- `return`
- `old_value`
- `new_value`
- `value_update_delta`
- `value_error`
- `count`

### 5.3 通常評価

補助確認として以下も残す。
- `eval_before -> eval` の `avg_rank/avg_score/win_rate/deal_in_rate`
- after 指標
- imitation phase の `teacher_top1_match_rate`, `teacher_best_set_hit_rate`, `value_loss`

## 6. 成功判定

- batch run が `5/5 success`
- `summary.json.success=true`
- `summary.json.phase_stats.learner.ppo_diag.shanten_diag` が存在する
- `summary.json.phase_stats.learner.ppo_diag.turn_diag` が存在する
- `improve/same/worsen` の 3 群が空でない
- `early/mid/late` の 3 バケットが参照できる
- `NaN` / 欠落で集計不能になっていない

## 7. レポートで答える問い

1. `improve` 群の `return` は本当に高いか、それとも `old_value` が高すぎるのか。  
2. `worsen` 群で `new_value` はどちら向きに更新されているか。  
3. `turn_diag` で終盤ほど `value_error` や `advantage` の歪みが強くなっていないか。  
4. 次に触るべきものは `value target/value loss` なのか、`turn/shanten` 相互作用特徴なのか。

## 8. 運用メモ

- 今回は単条件のため `run_map.json` は 1 件のみになる。  
- この run は比較用というより診断用なので、主レポートは `exp_024` の採否を上書きしない。  
- 次の実験条件は、本 run の `shanten_diag` / `turn_diag` を見てから決める。
