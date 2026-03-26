# Experiment Runbook: exp_067

作成日: 2026-03-22  
目的: CQ-0210 / CQ-0211 修正後の baseline を基準に、複数の仮説を **単発・1 seed・単一変更** で広く当たり、次に 3 seeds で詰める候補を探す。

## 1. 背景

- CQ-0210 / CQ-0211 により、GAE / return は flat sample 列ではなく **same-player decision trajectory** 上で計算されるようになった。
- 修正後の初回検証では、従来の best 条件 `g075_gae030` でも
  - `improve / same / worsen` の向きはかなり自然化
  - 一方で final `avg_score` は imitation 直後を下回る
  という結果になった。
- `exp_066` の途中結果では、`gamma=0.50` まで戻すのは短すぎる寄りに見えた。
- したがって今は、1 つの sweep を最後まで回すより、**単発で複数ノブの地形を広く見る** 方が情報量が高い。

## 2. 実験の問い

1. 修正後 semantics では、最適 horizon はさらに長いのか
2. 現在の保持不足は
   - update の強さ (`clip`)
   - anchor の強さ (`policy_anchor.coef`)
   - distribution mismatch (`policy_ratio`)
   - reward shaping (`shanten_scale`)
   のどれに最も強く反応するか
3. `improve / same / worsen` の自然化を保ったまま final を押し上げる方向はあるか

## 3. 基準条件

今回の exploratory は、以下の **修正後 baseline** を基準にして各条件で 1 個だけ変更する。

基準:
- 新モデル (`policy_direct_hints + context_gate`)
- `rule-only PPO`
- `training.gamma=0.75`
- `training.gae_lambda=0.3`
- `training.clip_epsilon=0.15`
- `training.policy_anchor.coef=0.5`
- `training.value_loss_coef=0.25`
- `training.rule_mix.policy_ratio=0.0`
- `reward.shaping.shanten_delta.scale=0.003`
- imitation `1000 matches x 3 chunks`
- PPO `200 matches x 30 cycles`
- seed `42`

参照条件:
- `REF g075_gae030`
  - 既存の修正後 run を再利用してよい
  - 新規実行は不要

## 4. 条件一覧

- 条件数: 10
- seeds: `42`
- 方針: **すべて baseline から 1 つだけ変更**

| 条件 | 変更内容 | 変更値 | 見たい仮説 |
|---|---|---:|---|
| A `gamma090` | `training.gamma` | `0.90` | same-player semantics ではさらに長い horizon が良いか |
| B `gae000` | `training.gae_lambda` | `0.0` | GAE 自体がまだ長すぎるか |
| C `clip010` | `training.clip_epsilon` | `0.10` | update がまだ強すぎるか |
| D `anchor075` | `training.policy_anchor.coef` | `0.75` | drift をまだ止め切れていないか |
| E `ratio005` | `training.rule_mix.policy_ratio` | `0.05` | 少量 actor mix が効くか |
| F `shape001` | `reward.shaping.shanten_delta.scale` | `0.001` | shaping が強すぎるか |
| G `clip020` | `training.clip_epsilon` | `0.20` | update が弱すぎるか |
| H `anchor025` | `training.policy_anchor.coef` | `0.25` | anchor が少し強すぎるか |
| I `ratio010` | `training.rule_mix.policy_ratio` | `0.10` | actor mix をもう少し増やすと良いか |
| J `shape000` | `reward.shaping.shanten_delta.scale` | `0.0` | shaping を消した方が良いか |

補足:
- `B gae000` は `gamma=0.75` を維持したまま `gae_lambda=0.0` にする単発であり、`exp_066` の `g050_gae000` とは問いが異なる
- `E/I ratio005/010` は corrected semantics 上で mixed path を実戦確認する意味もある
- `J shape000` は `reward.shaping.shanten_delta.enabled=false` でもよいが、まずは `scale=0.0` で差分を最小にする

## 5. 共通固定条件

- config:
  - `configs/stage1_full_flat_mlp_rule_only_anchor_ppo_baseline.yaml`
- 新モデル:
  - `model.policy_direct_hints.enabled=true`
  - `model.policy_direct_hints.sources=["shanten_hint","discard_ukeire_hint"]`
  - `model.policy_direct_hints.local_hidden_dim=16`
  - `model.policy_direct_hints.tile_embedding_dim=4`
  - `model.policy_direct_hints.context_gate.enabled=true`
- feature:
  - `feature_encoder.shanten_hint.enabled=true`
  - `feature_encoder.discard_ukeire_hint.enabled=true`
  - `feature_encoder.current_shanten.enabled=true`
  - `feature_encoder.shape_hint.enabled=true`
  - `feature_encoder.turn_context.enabled=true`
- imitation:
  - `training.imitation_loss_mode=tie_aware_best_set`
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.3`
  - `training.multi_chunk_imitation.enabled=true`
  - `training.multi_chunk_imitation.num_chunks=3`
  - `training.multi_chunk_imitation.imitation_matches_per_chunk=1000`
  - total imitation matches = `3000`
  - `training.imitation_epochs=8`
  - `imitation.num_workers=10`
- PPO:
  - `training.rule_mix.enabled=true`
  - `training.rule_mix.save_baseline_actions=true`
  - `training.rule_mix_learner.enabled=true`
  - `training.rule_mix_learner.ppo_mode=mixed`
  - `training.rule_mix_learner.baseline_sample_weight=1.0`
  - `training.policy_anchor.enabled=true`
  - `training.policy_anchor.type=kl`
  - `training.policy_anchor.reference=imitation_fixed`
- optimization:
  - `training.lr=5e-5`
  - `training.epochs=1`
  - `training.batch_size=512`
  - `training.value_loss.type=mse`
  - `training.value_loss_coef=0.25`
  - `training.advantage_stabilization.clip=null`
  - `training.entropy_coef=0.0`
- reward:
  - `reward.point_delta_scale=0.0001`
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
- selfplay / cycle:
  - `selfplay.imitation_matches=1000`
  - `selfplay.num_matches=200`
  - `selfplay.num_workers=10`
  - `selfplay.policy_ratio=1.0`
  - `selfplay.save_baseline_actions=false`
  - `training.multi_cycle.enabled=true`
  - `training.multi_cycle.num_cycles=30`
  - `training.multi_cycle.selfplay_matches_per_cycle=200`
  - `training.multi_cycle.eval_each_cycle=true`
- eval:
  - `evaluation.mode=rotation`
  - `evaluation.rotation_seats=[0,1,2,3]`
  - `evaluation.num_matches=100`
  - `evaluation.num_workers=10`
- device:
  - `training.device=cuda`
  - `selfplay.inference_device=cpu`
  - `evaluation.inference_device=cpu`

## 6. 主評価指標

1. performance
   - `cycle0.eval_before.avg_score`
   - `best avg_score`
   - `final avg_score`
   - `avg_rank`
   - `win_rate`
   - `deal_in_rate`
2. peak 保持
   - best cycle
   - `best -> final` drawdown
   - `final - imitation_initial`
3. teacher 診断
   - `teacher_agreement.best_set_hit_rate_after`
4. critic / target 診断
   - `value_error_mean`
   - `shanten_diag.improve/same/worsen.reward.mean`
   - `shanten_diag.improve/same/worsen.delta_t.mean`
   - `shanten_diag.improve/same/worsen.advantage.mean`
5. PPO update 診断
   - `clip_fraction`
   - `ratio_std`

## 7. 読み方

### `gamma090` が良い
- 修正後 semantics では、以前より長い horizon が必要
- 次は `0.75 / 0.90` 近辺を本命にする

### `gae000` が良い
- grouped semantics でも GAE は長すぎる
- 次は `gamma` を主に詰める

### `clip010` が良い
- corrected signal に対して step がまだ強い
- 次は `clip=0.10` を基準にしてよい

### `anchor075` が良い
- signal は良くなったが drift はまだ強い
- anchor を強める方向が有望

### `ratio005` / `ratio010` が良い
- distribution mismatch はまだ本物
- 次は rule-only 固定を崩して actor mix を見に行く

### `shape001` / `shape000` が良い
- 修正後は shanten shaping が強すぎる可能性
- reward 設計の見直しを優先する

## 8. 実行上の注意

- baseline config 自体は `gamma=0.50`, `gae=0.0` を内包しているため、今回の条件はすべて override 指定で実行する
- `REF g075_gae030` は既存の修正後 run を参照点として再利用できる
- report には `runs/` 配下のローカル成果物パスを書かず、必要な数値を転記して残す

## 9. 成功条件

- 10 条件の単発実行が完了する
- 各 run で:
  - imitation / selfplay / learner / eval が `success`
  - `summary.phase_stats.cycles` 長さ `30`
  - 変更対象の `config.yaml` が条件どおり
  - `phase_stats.learner.ppo_diag.policy_anchor.coef` が `anchor` 条件どおり
  - `mixed_ppo.num_policy_samples` / `num_baseline_samples` が `policy_ratio` 条件と整合する
- 翌朝、次の 3 seeds 候補を 2〜3 条件に絞れる

## 10. 想定所要時間

- `REF` は再利用
- 新規実行は `10` 条件
- 1 条件あたり `70〜90分` 程度
- 合計 `12〜15時間` 程度

## 11. 実行後にやること

1. exploratory の `report.md` を作る
2. 当たりの強いノブを 2〜3 本に絞る
3. その条件だけ 3 seeds で再確認する
