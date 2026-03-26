# Experiment Runbook: exp_026

作成日: 2026-03-10  
対象: `exp_025` の採用条件を基準に、モデル表現力を大きくした単条件診断

## 1. この実験の位置づけ

目的は、`exp_025` で観測された value/target の残差が、モデル表現力不足でどこまで改善するかを確認すること。  
今回は **単条件の診断 run** とし、比較参照は `exp_025` とする。

診断したい主仮説:
- 小さい MLP では value が十分に表現できておらず、`shanten_diag` / `turn_diag` の逆向き傾向が生じている
- hidden size を拡大し、value head 専用 `current_shanten` を有効にすると、value misfit が改善する可能性がある

## 2. 参照条件

参照元:
- `experiments/exp_025/report.md`
- 基準条件: `exp_024` 採用候補 B

固定する条件:
- `feature_encoder.shanten_hint={"enabled":true}`
- `training.imitation_loss_mode=tie_aware_best_set`
- `reward.shaping.shanten_delta.enabled=true`
- `reward.shaping.shanten_delta.scale=0.01`
- `reward.shaping.shanten_delta.mode=both`
- `reward.shaping.shanten_delta.schedule.type=linear_decay`
- `training.imitation_value_warmstart.enabled=true`
- `training.imitation_value_warmstart.coef=0.1`

今回だけ変更する条件:
- `model.hidden_dims=[512,256]`
- `model.value_features.current_shanten.enabled=true`

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
    model.hidden_dims='[512,256]' \
    model.value_features.current_shanten.enabled=true
```

## 5. 主確認項目

### 5.1 通常評価

`exp_025` と比較して以下を確認する。

- `eval_before -> eval` の `avg_rank/avg_score/win_rate/deal_in_rate`
- after 指標

### 5.2 shanten_diag

以下を `improve/same/worsen` ごとに `exp_025` と比較する。

- `advantage.mean`
- `return.mean`
- `old_value.mean`
- `new_value.mean`
- `value_update_delta.mean`
- `value_error.mean`
- `count`

特に見たい点:
- `improve.advantage.mean` が 0 に近づく、または正側へ寄るか
- `worsen.advantage.mean` が 0 に近づく、または負側へ寄るか
- `improve.value_error.mean` が下がるか

### 5.3 turn_diag

以下を `early/mid/late` ごとに `exp_025` と比較する。

- `advantage.mean`
- `return.mean`
- `old_value.mean`
- `new_value.mean`
- `value_update_delta.mean`
- `value_error.mean`

特に見たい点:
- `late.value_error.mean` が下がるか
- `late.advantage.mean` が負側から改善するか

### 5.4 imitation 指標

- `teacher_top1_match_rate`
- `teacher_best_set_hit_rate`
- `value_loss`

## 6. 成功判定

- batch run が `5/5 success`
- `summary.json.success=true`
- `summary.json.phase_stats.learner.ppo_diag.shanten_diag` が存在する
- `summary.json.phase_stats.learner.ppo_diag.turn_diag` が存在する
- `summary.json.model_features.value_features.current_shanten.enabled=true` を確認できる
- `NaN` / 欠落で集計不能になっていない

## 7. レポートで答える問い

1. モデル拡大 + value current_shanten により、`shanten_diag` の逆向き傾向は改善するか。  
2. `late` bucket の `value_error` は下がるか。  
3. 通常評価の改善が見えるか。  
4. 次に進むべきは、さらに表現力を上げる方向か、target/value loss 設計を触る方向か。

## 8. 運用メモ

- 今回は単条件のため `run_map.json` は 1 件のみになる。  
- 比較対象は `exp_025` とし、新規 baseline は回さない。  
- この run が改善しなければ、単純な表現力増加だけでは足りない可能性が高い。
