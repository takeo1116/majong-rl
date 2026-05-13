# Experiment Runbook: exp_036

作成日: 2026-05-10  
Stage: `Stage02_CallUnlock` / `Stage02b_OptionalActionUnlock`

## 1. 目的

`exp_036` は、optional action unlock 全 ON 条件の 60 cycle 1seed probe である。

`exp_035` の short smoke では、初回 bug を修正した後に 10 cycle が完走した。次に、実験として意味のある長さで `seed42` を 1 本回し、Stage02b optional unlock が学習性能を大きく壊さないか確認する。

有効化する optional action:

- `Riichi / NoRiichi`
- `TsumoWin / Skip`
- `Ron / Skip`
- `Ankan / Skip`
- `Kakan / Skip`
- `Kyuushu / Skip`

Physical discard unlock はまだ行わない。34種打牌は維持し、赤牌/通常牌が同時にある場合は通常牌を優先する。

## 2. 前提

実装済み CQ:

- CQ-0290: 34種打牌の concrete tile 解決で通常牌を赤牌より優先
- CQ-0291: optional action unlock
- CQ-0292: optional flag propagation / evaluator / diagnostics follow-up
- CQ-0293: target_kl applied diagnostics consistency

直近 baseline:

| experiment | condition | result |
|---|---|---|
| `exp_034` | Stage02a stable baseline, 3seed | best avg_rank mean 1.998 / tail20 mean 2.122 |
| `exp_035` | optional all ON short smoke, seed42, 10 cycles | completed, final avg_rank 2.390 |

## 3. 実験条件

`exp_034` の安定設定を維持し、optional flags を全 ON にする。

| parameter | value |
|---|---:|
| seed | 42 |
| cycles | 60 |
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

Optional flags:

```yaml
training:
  optional_riichi:
    enabled: true
  optional_tsumo:
    enabled: true
  optional_ron:
    enabled: true
  optional_ankan:
    enabled: true
  optional_kakan:
    enabled: true
  optional_kyuushu:
    enabled: true
```

## 4. 実行コマンド

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_core_minimal_mixed_s1_baseline.yaml \
  --base-dir runs \
  --override \
    experiment.name='"20260510_stage2b_optional_all_probe_seed42"' \
    experiment.global_seed=42 \
    feature_encoder.tile_presence_flags=true \
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
    training.optional_tsumo.enabled=true \
    training.optional_ron.enabled=true \
    training.optional_ankan.enabled=true \
    training.optional_kakan.enabled=true \
    training.optional_kyuushu.enabled=true \
    training.multi_cycle.enabled=true \
    training.multi_cycle.num_cycles=60 \
    training.multi_cycle.selfplay_matches_per_cycle=200 \
    training.multi_cycle.eval_each_cycle=true
```

## 5. 確認項目

### 5.1 完走条件

- 60 cycle が crash せず完走する
- imitation / selfplay / learner / eval がすべて成功する
- checkpoint save/load が成功する
- `decision_family_counts` が summary / selfplay stats に出る
- `optional_decision_count > 0`

### 5.2 family diagnostics

確認する family:

| family | expected |
|---|---|
| `riichi` | 出るはず |
| `tsumo` | 出るはず |
| `ron` | 出るはず |
| `ankan` | 少数でも出ればよい |
| `kakan` | 少数でも出ればよい |
| `kyuushu` | rare。ゼロでも即失敗扱いにはしない |

### 5.3 性能判定

比較対象は `exp_034 seed42`。

| metric | exp_034 seed42 |
|---|---:|
| final avg_rank | 1.960 |
| best avg_rank | 1.960 |
| best10 | 2.032 |
| tail10 | 2.098 |
| tail20 | 2.098 |

`exp_036` の暫定判定:

| result | action |
|---|---|
| tail20 <= 2.20 | 3seed 化を強く検討 |
| 2.20 < tail20 <= 2.35 | family diagnostics を見て判断 |
| tail20 > 2.35 | optional family 分解、または一部 flag off 実験を検討 |
| crash | CQ を切って修正 |

## 6. 注意点

`exp_035` smoke は CQ-0292 batch 2 前だったため optional family counts が summary に出ていない。`exp_036` は CQ-0292 後なので、`decision_family_counts` を必ず確認する。

`CQ-0280` の eval retry は未実装。eval worker が稀に落ちた場合は、現時点では run が失敗する可能性がある。その場合は同じ checkpoint の eval retry または CQ-0280 実装を検討する。
