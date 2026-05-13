# Experiment Runbook: exp_035

作成日: 2026-05-10  
Stage: `Stage02_CallUnlock` / `Stage02b_OptionalActionUnlock`

## 1. 目的

`exp_035` は、CQ-0290 / CQ-0291 後の rule expansion smoke 実験である。

`exp_034` で Stage02a の安定 baseline は確認できたため、次に physical discard unlock 以外の本来選択可能行動を optional action として有効化し、学習 pipeline が壊れないことを確認する。

有効化する optional action:

- `Riichi / NoRiichi`
- `TsumoWin / Skip`
- `Ron / Skip`
- `Ankan / Skip`
- `Kakan / Skip`
- `Kyuushu / Skip`

Physical discard unlock はまだ行わない。34種打牌は維持し、CQ-0290 により赤牌/通常牌が同時にある場合は通常牌を優先する。

## 2. 背景

直近の安定 baseline は `exp_034`。

```text
policy lr          = 1e-4
value_semantic lr  = 1e-2
target_kl          = enabled
policy_ratio       = 1.0
ppo_mode           = separated
reward scale       = 0.0001
multi_cycle        = 60 cycles
```

`exp_034` 3seed aggregate:

| metric | mean | std |
|---|---:|---:|
| final avg_rank | 2.078 | 0.091 |
| best avg_rank | 1.998 | 0.029 |
| best10 | 2.048 | 0.014 |
| tail20 | 2.122 | 0.017 |

今回の CQ:

- CQ-0290: 34種打牌の concrete tile 解決で通常牌を赤牌より優先
- CQ-0291 batch 1: Riichi optional unlock
- CQ-0291 batch 2: TsumoWin / Ron optional unlock
- CQ-0291 batch 3: Ankan / Kakan / Kyuushu optional unlock

## 3. 実験方針

今回はいきなり 60 cycle 3seed には行かない。

まず `seed42` の short smoke で、以下を確認する。

- optional action sample が shard に出る
- learner / selector / model / checkpoint load が crash しない
- teacher が現行自動処理互換に近い挙動をしている
- PPO が optional branch を扱っても大崩れしない

その後、問題なければ同じ条件で 60 cycle 1seed を回す。

## 4. 条件

ベースは `exp_034` と同じ。

追加で以下を有効化する。

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

Teacher / baseline の期待挙動:

| family | teacher |
|---|---|
| `riichi` | Riichi |
| `tsumo` | TsumoWin |
| `ron` | Ron |
| `ankan` | Skip |
| `kakan` | Skip |
| `kyuushu` | Skip |

## 5. 実行コマンド

### 5.1 Short Smoke: 10 cycle seed42

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_core_minimal_mixed_s1_baseline.yaml \
  --base-dir runs \
  --override \
    experiment.name='"20260510_stage2b_optional_all_smoke_seed42"' \
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
    training.multi_cycle.num_cycles=10 \
    training.multi_cycle.selfplay_matches_per_cycle=200 \
    training.multi_cycle.eval_each_cycle=true
```

### 5.2 Probe: 60 cycle seed42

Short smoke が通った場合のみ実行する。

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

## 6. 確認項目

### 6.1 Smoke の確認

short smoke では性能よりも以下を優先する。

| item | expected |
|---|---|
| run completion | crash しない |
| shard read/write | optional family metadata が壊れない |
| learner | imitation / PPO が通る |
| checkpoint | save/load が通る |
| riichi samples | ある程度出る |
| ron/tsumo samples | ある程度出る |
| ankan/kakan/kyuushu samples | 10 cycle ではゼロでも即失敗扱いにしない |
| avg_rank | 極端な崩壊がない |

### 6.2 Probe の確認

60 cycle probe では `exp_034 seed42` と比較する。

`exp_034 seed42`:

| metric | value |
|---|---:|
| final | 1.960 |
| best | 1.960 |
| best10 | 2.032 |
| tail10 | 2.098 |
| tail20 | 2.098 |

optional unlock probe の暫定採用基準:

- crash しない
- final / tail20 が極端に悪化しない
- `tail20 <= 2.25` 程度なら 3seed 化を検討
- `tail20 > 2.35` なら optional family 別に分解して原因を見る

## 7. 想定されるリスク

### Riichi

現行は自動リーチ寄りだったため、optional 化で policy が NoRiichi を覚えると短期性能が落ちる可能性がある。

### Ron / Tsumo

teacher は和了を選ぶが、PPO で Skip 側に寄ると性能が大きく落ちる可能性がある。特に Ron skip はフリテンや見逃し状態への影響を見る必要がある。

### Kan

Ankan/Kakan はドラ・嶺上・槍槓に関係するため、頻度は低いが副作用が大きい。初期 teacher は Skip なので性能への影響は小さいはずだが、PPO で Kan を選び始める場合は注意する。

### Kyuushu

発生頻度が低く、10 cycle では十分な sample が出ない可能性が高い。これは smoke failure ではない。

## 8. 次の判断

Short smoke が成功:

```text
60 cycle seed42 probe へ進む。
```

60 cycle seed42 が許容範囲:

```text
seed43/44 を追加して exp_035 3seed 化する。
```

明確に悪化:

```text
optional family を分けて ablation する。
候補:
- Riichi only
- Riichi + win optional
- Kan/Kyuushu disabled
```

## 9. メモ

この実験は「強くするための feature 実験」ではなく、「本来ルールへ戻すための境界確認」である。  
多少の性能低下は許容するが、学習が壊れる・sample schema が不安定になる・teacher/top1 が破綻する場合は、ルール拡張前に修正する。
