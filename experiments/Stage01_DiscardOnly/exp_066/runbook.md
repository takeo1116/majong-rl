# Experiment Runbook: exp_066

作成日: 2026-03-22  
目的: CQ-0210 / CQ-0211 修正後の new semantics 前提で、`gamma` / `gae_lambda` を**短い側**に振ったときの PPO 挙動を 1 seed pilot で再確認する。

## 1. 背景

- CQ-0210 / CQ-0211 により、GAE / return は flat sample 列ではなく **same-player decision trajectory** 上で計算されるようになった。
- 修正後の初回確認では、従来の最良条件
  - `gamma=0.75`
  - `gae_lambda=0.3`
  - `policy_anchor.coef=0.5`
  - `clip_epsilon=0.15`
  - `value_loss_coef=0.25`
  をそのまま使うと、
  - `shanten_diag.improve/same/worsen` の向きはかなり自然化
  - 一方で final `avg_score` は imitation 直後を下回る
  という結果になった。
- これは、**旧最良ハイパラが壊れた return semantics に対して最適化されていた**可能性を示している。
- 今の priority は、修正後 baseline に対してまず horizon を re-center することである。

したがって今回は、

**修正後 semantics 前提で `gamma` / `gae_lambda` を短い側に寄せた pilot を 1 seed で比較し、次の本番 sweep の当たりを探る。**

## 2. 実験の問い

1. 修正後 semantics では、`gamma=0.75, gae=0.3` は長すぎるのか
2. `gamma` / `gae` を短い側に戻すと、
   - final `avg_score`
   - best -> final drawdown
   - imitation 直後 vs final
   - `teacher_best_set_hit_rate_after`
   が改善するか
3. `shanten_diag.improve/same/worsen` の自然化を保ったまま、性能を戻せるか

## 3. 条件

- 条件数: 5
- seeds: `42`
- learner 形態: **すべて新 semantics + rule-only PPO + anchor(0.5)**

条件一覧:

| 条件 | gamma | gae_lambda | 位置づけ |
|---|---:|---:|---|
| A `g050_gae000` | `0.50` | `0.0` | 最短側 |
| B `g050_gae030` | `0.50` | `0.3` | 旧 baseline gamma に moderate gae |
| C `g065_gae000` | `0.65` | `0.0` | 中間 gamma, 短い gae |
| D `g065_gae030` | `0.65` | `0.3` | 中間 gamma, moderate gae |
| E `g075_gae030` | `0.75` | `0.3` | 修正後の現参照点 |

補足:
- `E g075_gae030` は 2026-03-22 の修正後 1 seed run を共通参照点として扱う。
- 今回は **pilot** なので 1 seed のみ。
- `gae=0.6` や `gamma=0.90` は、修正後の新しい地形が見えてから次段で判断する。

## 4. 共通固定条件

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
  - `training.rule_mix.policy_ratio=0.0`
  - `training.rule_mix.save_baseline_actions=true`
  - `training.rule_mix_learner.enabled=true`
  - `training.rule_mix_learner.ppo_mode=mixed`
  - `training.rule_mix_learner.baseline_sample_weight=1.0`
  - `training.policy_anchor.enabled=true`
  - `training.policy_anchor.type=kl`
  - `training.policy_anchor.reference=imitation_fixed`
  - `training.policy_anchor.coef=0.5`
- optimization:
  - `training.lr=5e-5`
  - `training.epochs=1`
  - `training.batch_size=512`
  - `training.value_loss.type=mse`
  - `training.value_loss_coef=0.25`
  - `training.advantage_stabilization.clip=null`
  - `training.entropy_coef=0.0`
  - `training.clip_epsilon=0.15`
- reward:
  - `reward.point_delta_scale=0.0001`
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.003`
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

## 5. 主評価指標

1. performance
   - `cycle0.eval_before.avg_score` (imitation 直後)
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
   - `teacher_agreement.action_match_rate_after`
   - `teacher_agreement.best_set_hit_rate_after`
4. critic / target 診断
   - `value_error_mean`
   - `turn_diag.late.value_error.mean`
   - `shanten_diag.improve/same/worsen.reward.mean`
   - `shanten_diag.improve/same/worsen.delta_t.mean`
   - `shanten_diag.improve/same/worsen.advantage.mean`
5. PPO update 診断
   - `clip_fraction`
   - `ratio_std`

## 6. 見たい読み方

### ケース 1: `g050_gae000` か `g050_gae030` が改善

解釈:
- 修正後 semantics では `0.75 / 0.3` は長すぎる
- まずは短い horizon に戻すのが正しい

### ケース 2: `g065_*` が最良

解釈:
- `0.75` は長すぎるが、`0.50` まで戻す必要はない
- 修正後 baseline の中心は `0.65` 付近にある

### ケース 3: `gae=0.0` が一貫して良い

解釈:
- grouped semantics でも GAE は短い方がよい
- 次は `gamma` を主に詰める

### ケース 4: `gae=0.3` が一貫して良い

解釈:
- GAE 自体は有効だが、`gamma=0.75` が長すぎた
- `gamma` を詰める方向が本命

### ケース 5: `g075_gae030` がまだ最良

解釈:
- horizon そのものより、anchor / clip / policy_ratio の方が本命
- 次は horizon 以外を見に行く

## 7. 実行上の注意

- 現在の baseline config は `gamma=0.50`, `gae=0.0` を内包しているので、今回の条件はすべて override で指定する。
- `E g075_gae030` は修正後 1 seed の既存 run を参照点として再利用できる。
- report には `runs/` 配下のローカル成果物パスを書かず、必要な数値を転記して残す。

## 8. 成功条件

- 新規条件 `A-D` が完走する
- 各 run で:
  - imitation / selfplay / learner / eval が `success`
  - `summary.phase_stats.cycles` 長さ `30`
  - `config.yaml -> training.gamma` が条件どおり
  - `config.yaml -> training.gae_lambda` が条件どおり
  - `phase_stats.learner.ppo_diag.policy_anchor.coef == 0.5`
  - `phase_stats.learner.ppo_diag.mixed_ppo.num_policy_samples == 0`
  - `phase_stats.learner.ppo_diag.mixed_ppo.num_baseline_samples > 0`
- `E g075_gae030` を含めた 5 条件比較で、次に 3 seeds へ進める候補を 1～2 条件に絞れる

## 9. 想定所要時間

- `E g075_gae030` は既存参照点
- 新規実行は `4` 条件
- 1 条件あたり `70〜90分` 程度
- 合計 `5〜6.5時間` 程度

## 10. 実行後にやること

1. pilot の `report.md` を作る
2. 修正後 semantics 前提での暫定 horizon を決める
3. その条件で
   - 3 seeds 再確認
   - もしくは `anchor / clip` の再調整
   に進む
