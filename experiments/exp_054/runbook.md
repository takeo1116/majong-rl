# exp_054 runbook

最終更新: 2026-03-16  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: `imitation=0 + policy_ratio=0.0 + mixed_ppo` の簡約問題で、`gae_lambda` をさらに下げたときに改善ピークと崩れ方がどう変わるかを確認する

---

## 0. 実験の位置づけ

- 背景:
  - `exp_053` では `lr / clip / batch` より `gae_lambda=0.70` の改善効果が大きく、update magnitude より advantage target の質が重要と示唆された。
  - ただし `0.70` より低い領域は未探索で、最適帯がまだ不明。
- 仮説:
  - 現行の rule-only PPO では、長い horizon の credit assignment が noisy すぎる。
  - `gae_lambda` をさらに下げると、最初の改善ピークが高くなるか、あるいは崩れが小さくなる可能性がある。
- 方針:
  - `exp_053` の簡約条件をそのまま固定し、新規に `gae_lambda=0.0 / 0.3 / 0.6` の3条件を比較する。
  - 既存の `0.70` と `0.85` は参照系列として後で並べて比較する。

## 1. 条件

- 条件数: 3
- seeds: `42,43,44`（3 seeds）
- cycles: `10`
- eval: `rotation, num_matches=100`

条件一覧:

- A: `gae_000`
  - `gae_lambda=0.00`
- B: `gae_030`
  - `gae_lambda=0.30`
- C: `gae_060`
  - `gae_lambda=0.60`

## 2. 共通固定（override）

- `experiment.phases=["selfplay","learner","eval"]`
- `feature_encoder.shanten_hint.enabled=true`
- `feature_encoder.discard_ukeire_hint.enabled=false`
- `feature_encoder.current_shanten.enabled=true`
- `feature_encoder.shape_hint.enabled=true`
- `feature_encoder.turn_context.enabled=true`
- `training.imitation_loss_mode=tie_aware_best_set`
- `training.imitation_value_warmstart.enabled=true`
- `training.imitation_value_warmstart.coef=0.3`
- `training.exclude_post_riichi_discards.enabled=true`
- `training.value_loss.type=mse`
- `training.advantage_stabilization.clip=null`
- `training.policy_anchor.enabled=false`
- `training.entropy_coef=0.0`
- `reward.point_delta_scale=0.0001`
- `reward.shaping.shanten_delta.enabled=true`
- `reward.shaping.shanten_delta.scale=0.01`
- `reward.shaping.shanten_delta.mode=both`
- `reward.shaping.shanten_delta.schedule.type=linear_decay`
- `selfplay.num_matches=200`
- `selfplay.num_workers=10`
- `selfplay.policy_ratio=1.0`  # fallback
- `selfplay.save_baseline_actions=false`  # fallback
- `evaluation.mode=rotation`
- `evaluation.rotation_seats=[0,1,2,3]`
- `evaluation.num_matches=100`
- `evaluation.num_workers=10`
- `model.hidden_dims=[512,256]`
- `model.policy_tower.enabled=true`
- `model.policy_tower.hidden_dim=128`
- `model.value_tower.enabled=true`
- `model.value_tower.hidden_dim=128`
- `model.value_features.current_shanten.enabled=true`
- `training.lr=5e-5`
- `training.epochs=1`
- `training.value_loss_coef=0.25`
- `training.batch_size=512`
- `training.gamma=0.99`
- `training.clip_epsilon=0.15`
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`
- `training.multi_cycle.enabled=true`
- `training.multi_cycle.num_cycles=10`
- `training.multi_cycle.selfplay_matches_per_cycle=200`
- `training.multi_cycle.eval_each_cycle=true`
- `training.rule_mix.enabled=true`
- `training.rule_mix.policy_ratio=0.0`
- `training.rule_mix.save_baseline_actions=true`
- `training.rule_mix_learner.enabled=true`
- `training.rule_mix_learner.ppo_mode=mixed`
- `training.rule_mix_learner.baseline_sample_weight=1.0`

条件ごとの追加 override:

- A:
  - `training.gae_lambda=0.0`
- B:
  - `training.gae_lambda=0.3`
- C:
  - `training.gae_lambda=0.6`

## 3. 主評価

1. 初期基準（`cycle0.eval_before`）に対する after の改善量
   - `Δrank_vs_init`
   - `Δscore_vs_init`
2. peak の位置と大きさ
   - `best avg_rank cycle`
   - `best avg_score cycle`
   - `best rank gain`
   - `best score gain`
3. 崩れ方
   - `best -> final` の落ち幅
4. 各 cycle 内 `eval_before -> eval` 差分
   - `Δavg_rank`
   - `Δavg_score`
5. learner 診断
   - `value_error_mean`
   - `advantage_abs_mean_before_clip`
   - `clip_fraction`
   - `ratio_std`
   - `turn_diag.late.value_error`
   - `shanten_diag.improve/same/worsen advantage`

## 4. 成功判定

- 各条件の `batch_summary.json` で:
  - `success_count == 3`, `failure_count == 0`
- driver 全体で:
  - `completed == 3`, `failed == 0`
- 各 run で:
  - `summary.phase_stats.cycles` 長さ `10`
  - `cycles[*].learner_stages` が `mixed_ppo` のみ
  - `cycles[*].learner_diag.mixed_ppo.mixed_ppo_enabled == true`
  - `cycles[*].learner_diag.mixed_ppo.num_policy_samples == 0`
  - `cycles[*].learner_diag.mixed_ppo.num_baseline_samples > 0`

## 5. 判定基準（診断実験）

- `gae` 低下が有効:
  - `0.70` より `best gain` または `final_vs_init` が改善
  - `best -> final` の落ち幅が小さくなる
  - `value_error` / `advantage_abs_mean` / `late.value_error` の少なくとも一部が改善
- `gae` を下げすぎ:
  - `0.0` で peak も final も悪化
  - `best cycle` が極端に前倒しになり、すぐ飽和する
- 長期 credit がほぼノイズ:
  - `0.0` や `0.3` が `0.6 / 0.7` を上回る

## 6. 見たい結論

この runbook で決めたいのは次の一点。

> 現行の rule-only mixed PPO で有効な `gae_lambda` は、`0.7` よりさらに低い帯にあるのか。

## 7. 想定所要時間

- 1条件 x 3 seeds: おおむね `30〜40分`
- 3条件合計: `90〜120分`
- 余裕込み: `2時間前後`

## 8. 実行方針

- まずはこの 3 条件を新規に回し、`exp_053` の `gae=0.70` と `gae=0.85` に後から合流して比較する
- ここで `0.3` か `0.6` が良ければ、その近傍を次段でさらに詰める
- ここで `0.0` が最良になるなら、現行 setting では長期 credit がほぼ毒になっているとみなす
