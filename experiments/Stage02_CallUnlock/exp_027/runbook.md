# Experiment Runbook: exp_027

作成日: 2026-05-07  
Stage: `Stage02_CallUnlock`

## 1. 目的

`exp_027` の目的は、CQ-0285 によって terminal semantic loss を player-round 平均へ正規化した後、`terminal / yaku / value` の loss coefficient をまとめて上げる必要があるかを 1seed probe で確認することである。

CQ-0285 後の 60-cycle gradient norm probe では、terminal gradient 支配は大幅に解消した一方、policy performance は `exp_026 P100 scaled seed42` より悪化した。

比較対象:

```text
runs/20260503_stage2a_rewardscale_probe_P100_seed42_dd0b0c5d
runs/20260507_stage2a_gradnorm60_cq0285_P100_scaled_seed42_5aaf163a
```

## 2. 背景

CQ-0285 前の gradient norm probe では、terminal aux が value/semantic trunk を極端に支配していた。

```text
weighted_terminal / weighted_yaku  ~= 128-318
weighted_terminal / weighted_value ~= 238-1344
```

CQ-0285 で terminal loss を `sum / weight_sum` に変更した後は、60-cycle probe で以下まで下がった。

```text
all-cycle average:
weighted_terminal / weighted_yaku  ~= 8.0
weighted_terminal / weighted_value ~= 9.6

late_50_59 average:
weighted_terminal / weighted_yaku  ~= 6.7
weighted_terminal / weighted_value ~= 7.2
```

これは gradient balance としては改善だが、性能は悪化した。

```text
exp026 seed42 baseline:
final  1.970
best   1.970
tail10 2.124
tail20 2.137

CQ-0285 + base coef:
final  2.295
best   2.165
tail10 2.374
tail20 2.361
```

悪化の候補仮説:

1. CQ-0285 により terminal/value/semantic 系の実効gradientが弱くなりすぎた
2. ただし terminal だけを戻すと再び terminal 支配に寄りすぎる
3. よって、まずは `terminal / yaku / value` の係数を同倍率で上げ、policy loss に対する auxiliary/value 全体の押す力を戻す

## 3. なぜ global lr ではなく coef sweep か

`training.lr` は optimizer 全体に効くため、policy trunk / optional trunk / value trunk / semantic heads すべての更新を大きくする。

今回調べたいのは「CQ-0285で弱くなった value/semantic 系のloss信号を戻すべきか」であり、policy PPO ratio を直接荒らしたいわけではない。

したがって、本実験では global lr は固定し、以下の係数だけを同倍率で上げる。

- `training.value_loss_coef`
- `training.semantic_aux.terminal_loss_coef`
- `training.semantic_aux.yaku_loss_coef`

これにより、terminalだけを強めるのではなく、value/semantic aux 全体をpolicy lossに対して強める。

## 4. 注意点

10x/50x/100x はかなり攻めたprobeである。

特に 50x/100x では auxiliary/value loss が policy loss を大きく上回り、`max_grad_norm=0.5` による clipping が頻発する可能性がある。この場合、更新量の大きさというより gradient の方向が auxiliary/value に支配される。

この実験は最適値探索ではなく、以下を確認する破綻境界探索でもある。

- auxiliary/value をまとめて強めると性能が戻るか
- `clip_fraction` / `log_ratio` / entropy が崩れるか
- gradient norm 比は維持されるか、または terminal が再び支配するか
- value/yaku/terminal の loss が正常に学習されるか

## 5. 実験条件

Base: CQ-0285 適用後の `P100 scaled` 条件。

固定条件:

- config: `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- seed: `42`
- `reward.point_delta_scale = 0.0001`
- `policy_ratio = 1.0`
- `ppo_mode = "separated"`
- `policy_anchor.enabled = false`
- `training.lr = 0.0001`
- `clip_epsilon = 0.15`
- `entropy_coef = 0.0`
- `multi_cycle.num_cycles = 60`
- `selfplay_matches_per_cycle = 200`
- `gradient_norms.enabled = true`
- `gradient_norms.max_batches_per_epoch = 4`

Base coefficients:

```yaml
training.value_loss_coef: 0.125
training.semantic_aux.terminal_loss_coef: 0.1
training.semantic_aux.yaku_loss_coef: 0.05
```

Sweep:

| label | multiplier | value_loss_coef | terminal_loss_coef | yaku_loss_coef | intent |
|---|---:|---:|---:|---:|---|
| `COEF10x` | 10x | 1.25 | 1.0 | 0.5 | realistic upper probe |
| `COEF50x` | 50x | 6.25 | 5.0 | 2.5 | aggressive instability probe |
| `COEF100x` | 100x | 12.5 | 10.0 | 5.0 | likely-break boundary probe |

## 6. 実行コマンド

ログ出力先:

```bash
mkdir -p experiments/Stage02_CallUnlock/exp_027/driver_logs
```

### COEF10x

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_core_minimal_mixed_s1_baseline.yaml \
  --base-dir runs \
  --override \
    experiment.name='"stage2a_exp027_cq0285_coef10x_seed42"' \
    experiment.global_seed=42 \
    selfplay.policy_ratio=1.0 \
    selfplay.temperature=1.0 \
    training.rule_mix.policy_ratio=1.0 \
    training.rule_mix.save_baseline_actions=false \
    training.rule_mix_learner.ppo_mode='"separated"' \
    training.rule_mix_learner.baseline_imitation_epochs=0 \
    training.rule_mix_learner.policy_ppo_epochs=1 \
    training.rule_mix_learner.allow_mixed_offpolicy_baseline=false \
    training.policy_anchor.enabled=false \
    training.policy_anchor.coef=0.0 \
    training.lr=0.0001 \
    training.clip_epsilon=0.15 \
    training.entropy_coef=0.0 \
    training.value_loss_coef=1.25 \
    training.multi_cycle.num_cycles=60 \
    training.multi_cycle.selfplay_matches_per_cycle=200 \
    training.semantic_aux.enabled=true \
    training.semantic_aux.terminal_loss_coef=1.0 \
    training.semantic_aux.yaku_loss_coef=0.5 \
    training.diagnostics.gradient_norms.enabled=true \
    training.diagnostics.gradient_norms.max_batches_per_epoch=4 \
    training.diagnostics.gradient_norms.every_n_epochs=1 \
    feature_encoder.tile_presence_flags=true \
    model.value_hidden_dims='[256,128]' \
    model.semantic_aux.enabled=true \
    model.semantic_aux.policy_projection_dim=16 \
    model.semantic_aux.tile_presence_flags_semantic_only=false \
    reward.type='"point_delta"' \
    reward.point_delta_scale=0.0001 \
  2>&1 | tee experiments/Stage02_CallUnlock/exp_027/driver_logs/coef10x_seed42.log
```

### COEF50x

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_core_minimal_mixed_s1_baseline.yaml \
  --base-dir runs \
  --override \
    experiment.name='"stage2a_exp027_cq0285_coef50x_seed42"' \
    experiment.global_seed=42 \
    selfplay.policy_ratio=1.0 \
    selfplay.temperature=1.0 \
    training.rule_mix.policy_ratio=1.0 \
    training.rule_mix.save_baseline_actions=false \
    training.rule_mix_learner.ppo_mode='"separated"' \
    training.rule_mix_learner.baseline_imitation_epochs=0 \
    training.rule_mix_learner.policy_ppo_epochs=1 \
    training.rule_mix_learner.allow_mixed_offpolicy_baseline=false \
    training.policy_anchor.enabled=false \
    training.policy_anchor.coef=0.0 \
    training.lr=0.0001 \
    training.clip_epsilon=0.15 \
    training.entropy_coef=0.0 \
    training.value_loss_coef=6.25 \
    training.multi_cycle.num_cycles=60 \
    training.multi_cycle.selfplay_matches_per_cycle=200 \
    training.semantic_aux.enabled=true \
    training.semantic_aux.terminal_loss_coef=5.0 \
    training.semantic_aux.yaku_loss_coef=2.5 \
    training.diagnostics.gradient_norms.enabled=true \
    training.diagnostics.gradient_norms.max_batches_per_epoch=4 \
    training.diagnostics.gradient_norms.every_n_epochs=1 \
    feature_encoder.tile_presence_flags=true \
    model.value_hidden_dims='[256,128]' \
    model.semantic_aux.enabled=true \
    model.semantic_aux.policy_projection_dim=16 \
    model.semantic_aux.tile_presence_flags_semantic_only=false \
    reward.type='"point_delta"' \
    reward.point_delta_scale=0.0001 \
  2>&1 | tee experiments/Stage02_CallUnlock/exp_027/driver_logs/coef50x_seed42.log
```

### COEF100x

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_core_minimal_mixed_s1_baseline.yaml \
  --base-dir runs \
  --override \
    experiment.name='"stage2a_exp027_cq0285_coef100x_seed42"' \
    experiment.global_seed=42 \
    selfplay.policy_ratio=1.0 \
    selfplay.temperature=1.0 \
    training.rule_mix.policy_ratio=1.0 \
    training.rule_mix.save_baseline_actions=false \
    training.rule_mix_learner.ppo_mode='"separated"' \
    training.rule_mix_learner.baseline_imitation_epochs=0 \
    training.rule_mix_learner.policy_ppo_epochs=1 \
    training.rule_mix_learner.allow_mixed_offpolicy_baseline=false \
    training.policy_anchor.enabled=false \
    training.policy_anchor.coef=0.0 \
    training.lr=0.0001 \
    training.clip_epsilon=0.15 \
    training.entropy_coef=0.0 \
    training.value_loss_coef=12.5 \
    training.multi_cycle.num_cycles=60 \
    training.multi_cycle.selfplay_matches_per_cycle=200 \
    training.semantic_aux.enabled=true \
    training.semantic_aux.terminal_loss_coef=10.0 \
    training.semantic_aux.yaku_loss_coef=5.0 \
    training.diagnostics.gradient_norms.enabled=true \
    training.diagnostics.gradient_norms.max_batches_per_epoch=4 \
    training.diagnostics.gradient_norms.every_n_epochs=1 \
    feature_encoder.tile_presence_flags=true \
    model.value_hidden_dims='[256,128]' \
    model.semantic_aux.enabled=true \
    model.semantic_aux.policy_projection_dim=16 \
    model.semantic_aux.tile_presence_flags_semantic_only=false \
    reward.type='"point_delta"' \
    reward.point_delta_scale=0.0001 \
  2>&1 | tee experiments/Stage02_CallUnlock/exp_027/driver_logs/coef100x_seed42.log
```

## 7. 主評価

性能:

- `final avg_rank`
- `best avg_rank`
- `best10 avg_rank`
- `tail10 avg_rank`
- `tail20 avg_rank`
- `win_rate`
- `deal_in_rate`

安定性:

- `clip_fraction`
- `log_ratio_p01`, `log_ratio_p99`
- `ratio_max`
- `entropy`
- `max_prob_mean`
- NaN / crash の有無

Gradient balance:

- `gradient_norms.aggregate.ratios.value_semantic_weighted_terminal_to_weighted_yaku`
- `gradient_norms.aggregate.ratios.value_semantic_weighted_terminal_to_weighted_value`
- `gradient_norms.aggregate.ratios.value_semantic_weighted_yaku_to_weighted_value`
- `weighted_terminal_loss.value_semantic.mean`
- `weighted_yaku_loss.value_semantic.mean`
- `weighted_value_loss.value_semantic.mean`

## 8. 採用判断

このprobeで採用候補になる条件:

- `CQ-0285 + base coef` より final / tail10 / best10 が改善する
- `exp_026 seed42` に近づく、または超える
- `clip_fraction` が過度に高止まりしない
- `ratio_max` が極端に爆発しない
- entropy collapse が明確に早まらない
- terminal/yaku/value の gradient 比が極端に崩れない

想定:

- `COEF10x` が最も現実的な候補
- `COEF50x` / `COEF100x` は破綻境界確認の意味が強い
- もし全て改善しない場合、次は terminal-only ではなく、別 optimizer param-group lr または semantic architecture 側を検討する

## 9. 集計コマンド

```bash
python3 - <<'PY'
import json
from pathlib import Path

patterns = [
    ('COEF10x', '*stage2a_exp027_cq0285_coef10x_seed42*'),
    ('COEF50x', '*stage2a_exp027_cq0285_coef50x_seed42*'),
    ('COEF100x', '*stage2a_exp027_cq0285_coef100x_seed42*'),
]

runs = []
for label, pat in patterns:
    ms = sorted(Path('runs').glob(pat), key=lambda p: p.stat().st_mtime)
    if ms:
        runs.append((label, ms[-1]))

print('label,run,final,best,best10,tail10,tail20,win,deal,entropy_late,clip_late,T/Y_late,T/V_late,Y/V_late')

for label, run in runs:
    s = json.loads((run / 'summary.json').read_text())
    cycles = s['phase_stats']['cycles']
    ranks = [c.get('eval_metrics', {}).get('avg_rank') for c in cycles]
    def avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None
    def best_window(w):
        return min(avg(ranks[i:i+w]) for i in range(len(ranks)-w+1))
    def ratio(c, key):
        return c['learner_metrics']['ppo_diag']['gradient_norms']['aggregate']['ratios'].get(key)
    late = cycles[-10:]
    row = [
        label, str(run),
        ranks[-1], min(ranks), best_window(10), avg(ranks[-10:]), avg(ranks[-20:]),
        cycles[-1].get('eval_metrics', {}).get('win_rate'),
        cycles[-1].get('eval_metrics', {}).get('deal_in_rate'),
        avg([c['learner_metrics'].get('entropy') for c in late]),
        avg([c['learner_metrics'].get('ppo_diag', {}).get('clip_fraction') for c in late]),
        avg([ratio(c, 'value_semantic_weighted_terminal_to_weighted_yaku') for c in late]),
        avg([ratio(c, 'value_semantic_weighted_terminal_to_weighted_value') for c in late]),
        avg([ratio(c, 'value_semantic_weighted_yaku_to_weighted_value') for c in late]),
    ]
    print(','.join('' if v is None else str(v) for v in row))
PY
```
