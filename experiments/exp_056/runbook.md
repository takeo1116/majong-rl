# exp_056 runbook

最終更新: 2026-03-16  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: `imitation=0 + policy_ratio=0.0 + mixed_ppo + gae=0.0 + shanten_delta.scale=0.003` の簡約問題で、`gamma` を大きく下げたときに plateau / 戻りがどう変わるかを確認する

---

## 0. 実験の位置づけ

- 背景:
  - `exp_054` で `gae=0.0` が最良だったため、現行の簡約問題では long-horizon credit assignment が強いノイズ源と分かった。
  - `exp_055` では `gamma=0.95, shanten_delta.scale=0.003` が最良となり、`shanten shaping` を少し弱めると plateau / 戻りがさらに改善することが分かった。
  - ただし `gamma` は `0.95` 未満を一度も見ておらず、future bootstrap がまだ主因として残っているかは未確定。
- 仮説:
  - 現状では `gae=0.0` にしても `gamma * V(s_{t+1})` が残っており、これが plateau / 戻りを作る主因かもしれない。
  - もしそうなら、`gamma` を `0.75 / 0.50 / 0.25` まで大胆に下げると、少なくともどこかで挙動が大きく変わる。
- 方針:
  - `exp_055` のベスト条件 `gamma=0.95, scale=0.003` を参照点としつつ、新規に `gamma=0.75 / 0.50 / 0.25` を追加する。
  - `scale` は今回は固定し、`gamma` 単軸の因果を明確にする。

## 1. 条件

- 新規条件数: 3
- 参照条件: `exp_055` D (`gamma=0.95, scale=0.003`)
- seeds: `42,43,44`（3 seeds）
- cycles: `10`
- eval: `rotation, num_matches=100`

新規条件一覧:

- A: `gamma075_shanten0003`
  - `gamma=0.75`
  - `shanten_delta.scale=0.003`
- B: `gamma050_shanten0003`
  - `gamma=0.50`
  - `shanten_delta.scale=0.003`
- C: `gamma025_shanten0003`
  - `gamma=0.25`
  - `shanten_delta.scale=0.003`

参照条件:

- `exp_055` D: `gamma095_shanten0003`
  - `gamma=0.95`
  - `shanten_delta.scale=0.003`

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
- `reward.shaping.shanten_delta.scale=0.003`
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
- `training.gae_lambda=0.0`
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
  - `training.gamma=0.75`
- B:
  - `training.gamma=0.50`
- C:
  - `training.gamma=0.25`

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
4. learner 診断
   - `value_error_mean`
   - `advantage_abs_mean_before_clip`
   - `clip_fraction`
   - `ratio_std`
   - `turn_diag.late.value_error`
5. teacher agreement 診断
   - `teacher_agreement.action_match_rate_before/after`
   - `teacher_agreement.best_set_hit_rate_before/after`
   - `before -> after` の改善量

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
  - `cycles[*].learner_diag.teacher_agreement.enabled == true`

## 5. 判定基準（診断実験）

- `gamma` をさらに下げる価値がある:
  - `0.75` または `0.50` が `0.95` より `final_vs_init` と `best->final` の両方で改善
  - `teacher_agreement` も維持または改善
- `future bootstrap` がまだ主因:
  - `0.75 -> 0.50 -> 0.25` と下げるにつれて plateau / 戻りが一貫して縮小
  - `late.value_error` も一緒に下がる
- 下げすぎ:
  - `0.25` で peak も final も明確に悪化
  - `teacher_agreement` は上がるのに score/rank が下がる
  - これは短期化しすぎて point reward / shaping の局所最適に寄っているサイン
- `gamma` は主因ではない:
  - `0.75 / 0.50 / 0.25` のいずれも `0.95` を超えない
  - この場合は次に `gamma` 以外の target / value 設計を見る

## 6. 見たい結論

この runbook で決めたいのは次の一点。

> `gae=0.0 + scale=0.003` の残差は、`gamma=0.95` でもまだ長すぎるのか。

## 7. 想定所要時間

- 1条件 x 3 seeds: おおむね `30〜40分`
- 3条件合計: `90〜120分`
- 余裕込み: `2時間前後`

## 8. 実行方針

- まず新規 3 条件を回し、`exp_055` D を参照点として並べて比較する
- `0.75` か `0.50` が良ければ、その近傍で次段の微調整を考える
- `0.25` まで良いなら、現行 target はまだかなり長すぎたとみなす
- どれも悪ければ、`gamma` はほぼ打ち止めと判断し、次は別の target / value 側を疑う
