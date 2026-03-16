# exp_053 runbook

最終更新: 2026-03-16  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: `imitation=0 + policy_ratio=0.0 + mixed_ppo` の簡約問題で、「最初だけ上がってすぐ下がる」挙動が update overshoot 起因かどうかを切り分ける

---

## 0. 実験の位置づけ

- 背景:
  - `policy_ratio=0.0` かつ `mixed_ppo` では、`imitation` ありでもなしでも「初期に少し改善し、その後悪化する」挙動が確認された。
  - 一方で `imitation=0` 条件では初期改善幅が大きく、rule-only PPO に学習信号自体はあることも確認できた。
- 仮説:
  - 長期悪化の主因が `rule` データそのものの無価値ではなく、更新量の強さ (`lr`, `clip`, `batch`, `gae`) による overshoot である可能性がある。
- 方針:
  - まずは最も簡約な設定 `imitation=0 + policy_ratio=0.0 + mixed_ppo` に固定し、update 強度に関わるノブだけを振る。
  - 本実験は「最強条件探索」ではなく「崩れ方の型を観察する診断実験」とする。

## 1. 条件

- 条件数: 6
- seeds: `42,43,44`（3 seeds）
- cycles: `10`
- eval: `rotation, num_matches=100`

条件一覧:

- A: `baseline_mixed_rule_only`
  - `lr=5e-5`, `clip=0.15`, `batch=512`, `gae=0.85`
- B: `low_lr`
  - `lr=2.5e-5`
- C: `low_clip`
  - `clip=0.075`
- D: `low_lr_low_clip`
  - `lr=2.5e-5`, `clip=0.075`
- E: `large_batch`
  - `batch=1024`
- F: `low_gae`
  - `gae=0.70`

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
- `training.epochs=1`
- `training.value_loss_coef=0.25`
- `training.gamma=0.99`
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
  - `training.lr=5e-5`
  - `training.clip_epsilon=0.15`
  - `training.batch_size=512`
  - `training.gae_lambda=0.85`
- B:
  - `training.lr=2.5e-5`
  - `training.clip_epsilon=0.15`
  - `training.batch_size=512`
  - `training.gae_lambda=0.85`
- C:
  - `training.lr=5e-5`
  - `training.clip_epsilon=0.075`
  - `training.batch_size=512`
  - `training.gae_lambda=0.85`
- D:
  - `training.lr=2.5e-5`
  - `training.clip_epsilon=0.075`
  - `training.batch_size=512`
  - `training.gae_lambda=0.85`
- E:
  - `training.lr=5e-5`
  - `training.clip_epsilon=0.15`
  - `training.batch_size=1024`
  - `training.gae_lambda=0.85`
- F:
  - `training.lr=5e-5`
  - `training.clip_epsilon=0.15`
  - `training.batch_size=512`
  - `training.gae_lambda=0.70`

## 3. 主評価

1. 初期基準（`cycle0.eval_before`）に対する after の改善量
   - `Δrank_vs_init`
   - `Δscore_vs_init`
2. 各 cycle 内 `eval_before -> eval` 差分
   - `Δavg_rank`
   - `Δavg_score`
3. peak の位置
   - `best avg_rank cycle`
   - `best avg_score cycle`
4. 崩れ方
   - best cycle から final cycle までの落ち幅
5. learner 診断
   - `clip_fraction`
   - `ratio_std`
   - `advantage_abs_mean_before_clip`
   - `mixed_ppo.num_baseline_samples`

## 4. 成功判定

- 各条件の `batch_summary.json` で:
  - `success_count == 3`, `failure_count == 0`
- driver 全体で:
  - `completed == 6`, `failed == 0`
- 各 run で:
  - `summary.phase_stats.cycles` 長さ `10`
  - `cycles[*].learner_stages` が `mixed_ppo` のみ
  - `cycles[*].learner_diag.mixed_ppo.mixed_ppo_enabled == true`
  - `cycles[*].learner_diag.mixed_ppo.num_policy_samples == 0`
  - `cycles[*].learner_diag.mixed_ppo.num_baseline_samples > 0`

## 5. 判定基準（診断実験）

- overshoot 仮説を支持:
  - `best cycle` が baseline 条件より後ろへずれる
  - final の悪化幅が baseline 条件より小さい
  - `clip_fraction` / `ratio_std` が明確に下がる
- overshoot 仮説を弱める:
  - どの条件でも「cycle 0-3で改善 → その後悪化」がほぼ同じ形で再現
  - update を弱めても peak 値も final 値も改善しない

## 6. 見たい結論

この runbook で決めたいのは次の一点。

> `rule-only mixed PPO` の長期悪化は、まず update 強度を弱めれば改善する問題なのか、それとも target / off-policy mismatch による構造問題なのか。

## 7. 想定所要時間

- 1条件あたり: おおむね `10〜12分`
- 1条件 x 3 seeds: おおむね `30〜40分`
- 6条件合計: `180〜240分`
- 余裕込み: `3〜4時間`

## 8. 実行方針

- まずはこの 6 条件で「崩れ方の型」を見る
- ここで改善条件が見えたら、次段で seed を増やす
- ここで差が出なければ、次はハイパラより `target / rule データの入れ方` を変える
