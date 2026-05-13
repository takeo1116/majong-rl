# Experiment Runbook: exp_037

作成日: 2026-05-11  
Stage: `Stage02_CallUnlock` / `Stage02b_OptionalActionUnlock`

## 1. 目的

`exp_037` は、optional action unlock による性能低下の原因 family を切り分ける 1seed probe である。

`exp_036` では optional all ON の 60 cycle probe が完走したが、`exp_034` の stable baseline より明確に弱かった。続く CQ-0294 では以下を修正した。

- `optional_riichi.enabled=true` 時の teacher / baseline discard mask を旧 auto-riichi 相当に戻した
- `riichi_opportunity_*` diagnostics を追加した
- `feature_encoder.riichi_discard_mask` を追加した
- value 側 optional summary の action-type presence を新 action type 数に追従させた

CQ-0294 後の 10-cycle smoke (`runs/20260511_20260511_stage2b_cq0294_optional_all_smoke_seed42_65806243`) では、shape mismatch は解消し、Riichi bypass は 0 だった。一方、10 cycle 時点の avg_rank はまだ弱めだった。

したがって、次は optional family を分解し、どの family が性能低下に効いているかを見る。

## 2. 背景データ

### 2.1 Stable baseline

`exp_034` は Stage02a stable baseline であり、ルール拡張前の基準として扱う。

| experiment | condition | seeds | result |
|---|---|---:|---|
| `exp_034` | optional all off / P1 / VS100x / target_kl on | 42,43,44 | best mean 1.998 / final mean 2.078 / tail20 mean 2.122 |

seed42 の参照値:

| metric | value |
|---|---:|
| final avg_rank | 1.960 |
| best avg_rank | 1.960 |
| best10 | 2.032 |
| tail10 | 2.098 |
| tail20 | 2.098 |

### 2.2 Optional all ON

| run | cycles | result |
|---|---:|---|
| `exp_036` pre-CQ0294 | 60 | final 2.360 / best 2.125 / tail20 2.319 |
| CQ-0294 smoke | 10 | final 2.425 / best 2.425 / mean 2.533 |

CQ-0294 smoke の重要な観察:

- `riichi_bypass_rate = 0.0` in all cycles
- `riichi_opportunity_discard_count == riichi_optional_opened_count`
- したがって「立直可能なのに discard branch で別牌を切って optional に入らない」問題は見えていない
- 性能低下は Riichi 入口 bypass ではなく、他 optional family または optional action 全体の学習負荷増加が候補

## 3. 実験条件

3 条件を `seed42` で 1 本ずつ回す。

| label | optional_riichi | optional_tsumo | optional_ron | optional_ankan | optional_kakan | optional_kyuushu | hypothesis |
|---|---:|---:|---:|---:|---:|---:|---|
| `RII_ONLY` | on | off | off | off | off | off | Riichi optional 単体の影響を見る |
| `WIN_ONLY` | off | on | on | off | off | off | Tsumo/Ron optional が悪さをしているか見る |
| `KAN_ONLY` | off | off | off | on | on | on | Kan/Kyuushu optional が悪さをしているか見る |

共通設定:

| parameter | value |
|---|---:|
| seed | 42 |
| cycles | 30 |
| selfplay_matches_per_cycle | 200 |
| policy_ratio | 1.0 |
| ppo_mode | separated |
| reward.point_delta_scale | 0.0001 |
| policy lr | 0.0001 |
| value_semantic lr | 0.01 |
| target_kl | 0.03 |
| target_kl stop_multiplier | 1.5 |
| target_kl skip | true |
| gradient_norms | enabled |
| `feature_encoder.tile_presence_flags` | true |
| `feature_encoder.riichi_discard_mask` | true |

`riichi_discard_mask` は全条件で true にする。理由は、CQ-0294 後の optional-all 本命構成と同じ encoder/model input dim に揃えるためである。`optional_riichi=false` の条件でも、旧 auto-riichi discard mask は維持されるため、この feature は追加情報に留まる。

## 4. 実行コマンド

手動で 3 本回す場合は以下を使う。時間節約のため、まず `WIN_ONLY` → `KAN_ONLY` → `RII_ONLY` の順を推奨する。

### 4.1 WIN_ONLY

```bash
mkdir -p experiments/Stage02_CallUnlock/exp_037/driver_logs

./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_core_minimal_mixed_s1_baseline.yaml \
  --base-dir runs \
  --override \
    experiment.name='"20260511_stage2b_exp037_WIN_ONLY_seed42"' \
    experiment.global_seed=42 \
    feature_encoder.tile_presence_flags=true \
    feature_encoder.riichi_discard_mask=true \
    model.value_hidden_dims='[256,128]' \
    model.semantic_aux.enabled=true \
    model.semantic_aux.tile_presence_flags_semantic_only=false \
    selfplay.policy_ratio=1.0 \
    selfplay.save_baseline_actions=false \
    selfplay.temperature=1.0 \
    reward.type='"point_delta"' \
    reward.point_delta_scale=0.0001 \
    training.lr=0.0001 \
    training.value_loss_coef=0.125 \
    training.clip_epsilon=0.15 \
    training.entropy_coef=0.0 \
    training.semantic_aux.enabled=true \
    training.semantic_aux.terminal_loss_coef=0.1 \
    training.semantic_aux.yaku_loss_coef=0.05 \
    training.rule_mix.enabled=true \
    training.rule_mix.policy_ratio=1.0 \
    training.rule_mix.save_baseline_actions=false \
    training.rule_mix_learner.enabled=true \
    training.rule_mix_learner.ppo_mode='"separated"' \
    training.rule_mix_learner.baseline_imitation_epochs=0 \
    training.rule_mix_learner.policy_ppo_epochs=1 \
    training.rule_mix_learner.allow_mixed_offpolicy_baseline=false \
    training.policy_anchor.enabled=false \
    training.policy_anchor.coef=0.0 \
    training.lr_groups.enabled=true \
    training.lr_groups.apply_to='["ppo"]' \
    training.lr_groups.policy=0.0001 \
    training.lr_groups.value_semantic=0.01 \
    training.lr_groups.default=0.0001 \
    training.ppo_target_kl.enabled=true \
    training.ppo_target_kl.target=0.03 \
    training.ppo_target_kl.stop_multiplier=1.5 \
    training.ppo_target_kl.skip_minibatch_on_exceed=true \
    training.diagnostics.gradient_norms.enabled=true \
    training.diagnostics.gradient_norms.max_batches_per_epoch=4 \
    training.diagnostics.gradient_norms.every_n_epochs=1 \
    training.optional_riichi.enabled=false \
    training.optional_tsumo.enabled=true \
    training.optional_ron.enabled=true \
    training.optional_ankan.enabled=false \
    training.optional_kakan.enabled=false \
    training.optional_kyuushu.enabled=false \
    training.multi_cycle.enabled=true \
    training.multi_cycle.num_cycles=30 \
    training.multi_cycle.selfplay_matches_per_cycle=200 \
    training.multi_cycle.eval_each_cycle=true \
  2>&1 | tee experiments/Stage02_CallUnlock/exp_037/driver_logs/$(date +%Y%m%d_%H%M%S)_WIN_ONLY_seed42.log
```

### 4.2 KAN_ONLY

```bash
mkdir -p experiments/Stage02_CallUnlock/exp_037/driver_logs

./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_core_minimal_mixed_s1_baseline.yaml \
  --base-dir runs \
  --override \
    experiment.name='"20260511_stage2b_exp037_KAN_ONLY_seed42"' \
    experiment.global_seed=42 \
    feature_encoder.tile_presence_flags=true \
    feature_encoder.riichi_discard_mask=true \
    model.value_hidden_dims='[256,128]' \
    model.semantic_aux.enabled=true \
    model.semantic_aux.tile_presence_flags_semantic_only=false \
    selfplay.policy_ratio=1.0 \
    selfplay.save_baseline_actions=false \
    selfplay.temperature=1.0 \
    reward.type='"point_delta"' \
    reward.point_delta_scale=0.0001 \
    training.lr=0.0001 \
    training.value_loss_coef=0.125 \
    training.clip_epsilon=0.15 \
    training.entropy_coef=0.0 \
    training.semantic_aux.enabled=true \
    training.semantic_aux.terminal_loss_coef=0.1 \
    training.semantic_aux.yaku_loss_coef=0.05 \
    training.rule_mix.enabled=true \
    training.rule_mix.policy_ratio=1.0 \
    training.rule_mix.save_baseline_actions=false \
    training.rule_mix_learner.enabled=true \
    training.rule_mix_learner.ppo_mode='"separated"' \
    training.rule_mix_learner.baseline_imitation_epochs=0 \
    training.rule_mix_learner.policy_ppo_epochs=1 \
    training.rule_mix_learner.allow_mixed_offpolicy_baseline=false \
    training.policy_anchor.enabled=false \
    training.policy_anchor.coef=0.0 \
    training.lr_groups.enabled=true \
    training.lr_groups.apply_to='["ppo"]' \
    training.lr_groups.policy=0.0001 \
    training.lr_groups.value_semantic=0.01 \
    training.lr_groups.default=0.0001 \
    training.ppo_target_kl.enabled=true \
    training.ppo_target_kl.target=0.03 \
    training.ppo_target_kl.stop_multiplier=1.5 \
    training.ppo_target_kl.skip_minibatch_on_exceed=true \
    training.diagnostics.gradient_norms.enabled=true \
    training.diagnostics.gradient_norms.max_batches_per_epoch=4 \
    training.diagnostics.gradient_norms.every_n_epochs=1 \
    training.optional_riichi.enabled=false \
    training.optional_tsumo.enabled=false \
    training.optional_ron.enabled=false \
    training.optional_ankan.enabled=true \
    training.optional_kakan.enabled=true \
    training.optional_kyuushu.enabled=true \
    training.multi_cycle.enabled=true \
    training.multi_cycle.num_cycles=30 \
    training.multi_cycle.selfplay_matches_per_cycle=200 \
    training.multi_cycle.eval_each_cycle=true \
  2>&1 | tee experiments/Stage02_CallUnlock/exp_037/driver_logs/$(date +%Y%m%d_%H%M%S)_KAN_ONLY_seed42.log
```

### 4.3 RII_ONLY

```bash
mkdir -p experiments/Stage02_CallUnlock/exp_037/driver_logs

./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_core_minimal_mixed_s1_baseline.yaml \
  --base-dir runs \
  --override \
    experiment.name='"20260511_stage2b_exp037_RII_ONLY_seed42"' \
    experiment.global_seed=42 \
    feature_encoder.tile_presence_flags=true \
    feature_encoder.riichi_discard_mask=true \
    model.value_hidden_dims='[256,128]' \
    model.semantic_aux.enabled=true \
    model.semantic_aux.tile_presence_flags_semantic_only=false \
    selfplay.policy_ratio=1.0 \
    selfplay.save_baseline_actions=false \
    selfplay.temperature=1.0 \
    reward.type='"point_delta"' \
    reward.point_delta_scale=0.0001 \
    training.lr=0.0001 \
    training.value_loss_coef=0.125 \
    training.clip_epsilon=0.15 \
    training.entropy_coef=0.0 \
    training.semantic_aux.enabled=true \
    training.semantic_aux.terminal_loss_coef=0.1 \
    training.semantic_aux.yaku_loss_coef=0.05 \
    training.rule_mix.enabled=true \
    training.rule_mix.policy_ratio=1.0 \
    training.rule_mix.save_baseline_actions=false \
    training.rule_mix_learner.enabled=true \
    training.rule_mix_learner.ppo_mode='"separated"' \
    training.rule_mix_learner.baseline_imitation_epochs=0 \
    training.rule_mix_learner.policy_ppo_epochs=1 \
    training.rule_mix_learner.allow_mixed_offpolicy_baseline=false \
    training.policy_anchor.enabled=false \
    training.policy_anchor.coef=0.0 \
    training.lr_groups.enabled=true \
    training.lr_groups.apply_to='["ppo"]' \
    training.lr_groups.policy=0.0001 \
    training.lr_groups.value_semantic=0.01 \
    training.lr_groups.default=0.0001 \
    training.ppo_target_kl.enabled=true \
    training.ppo_target_kl.target=0.03 \
    training.ppo_target_kl.stop_multiplier=1.5 \
    training.ppo_target_kl.skip_minibatch_on_exceed=true \
    training.diagnostics.gradient_norms.enabled=true \
    training.diagnostics.gradient_norms.max_batches_per_epoch=4 \
    training.diagnostics.gradient_norms.every_n_epochs=1 \
    training.optional_riichi.enabled=true \
    training.optional_tsumo.enabled=false \
    training.optional_ron.enabled=false \
    training.optional_ankan.enabled=false \
    training.optional_kakan.enabled=false \
    training.optional_kyuushu.enabled=false \
    training.multi_cycle.enabled=true \
    training.multi_cycle.num_cycles=30 \
    training.multi_cycle.selfplay_matches_per_cycle=200 \
    training.multi_cycle.eval_each_cycle=true \
  2>&1 | tee experiments/Stage02_CallUnlock/exp_037/driver_logs/$(date +%Y%m%d_%H%M%S)_RII_ONLY_seed42.log
```

## 5. 判定基準

30 cycle 1seed probe なので、最終値だけではなく `best10` / `tail10` / family counts を見る。

| result | interpretation | next action |
|---|---|---|
| ある family only で `tail10 > 2.30` | その family が性能低下の主因候補 | family 内をさらに分解 |
| 全 family only が `tail10 <= 2.20` | all-on の相互作用 or 学習負荷が原因候補 | 2-family combination を見る |
| `RII_ONLY` だけ弱い | Riichi optional / riichi feature 周辺を再確認 | riichi decision policy / imitation を重点確認 |
| `WIN_ONLY` だけ弱い | Ron/Tsumo optional が主因候補 | Ron-only / Tsumo-only を見る |
| `KAN_ONLY` だけ弱い | Kan/Kyuushu optional が主因候補 | Kakan-only / Ankan-only / Kyuushu-only を見る |

補助的に確認する diagnostics:

- `decision_family_counts`
- `optional_decision_count`
- `riichi_bypass_rate`
- `target_kl_skipped_minibatches`
- `clip_fraction`
- `max_prob_mean / max_prob_p95`
- final `win_rate / deal_in_rate`

## 6. 注意点

- この実験は原因切り分け用であり、3seed 結論用ではない。
- 30 cycle で明確に弱い family が見えた場合は、60 cycle まで待たずに深掘りしてよい。
- どれも悪くない場合、optional all-on の組み合わせで初めて難しくなる可能性がある。
- `feature_encoder.riichi_discard_mask=true` は全条件で有効にしているため、`exp_034` との比較では encoder dim が異なる。その点は report で明記する。
