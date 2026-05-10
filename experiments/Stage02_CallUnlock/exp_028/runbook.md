# Experiment Runbook: exp_028

作成日: 2026-05-07  
Stage: `Stage02_CallUnlock`

## 1. 目的

`exp_028` の目的は、CQ-0285 後に悪化した性能が、
**terminal semantic loss の絶対スケール低下**で説明できるかを切り分けることである。

今回は `terminal_loss_coef` だけを上げ、`yaku` と `value` は base のまま固定する。

これにより、次の仮説を直接検証する。

```text
CQ-0285 前は terminal gradient 支配が value_trunk を強く shaping しており、
その結果できた semantic summary が policy に有用だった。
CQ-0285 によりその terminal signal が弱くなり、policy performance が落ちた。
```

## 2. 背景

### 2.1 exp_026 baseline

`reward.point_delta_scale=0.0001` を入れた `P100 scaled` 条件では、
seed42 で以下の性能が出ていた。

```text
runs/20260503_stage2a_rewardscale_probe_P100_seed42_dd0b0c5d

final  1.970
best   1.970
tail10 2.124
tail20 2.137
```

### 2.2 CQ-0285

CQ-0285 では terminal semantic loss の weighted path を以下のように変更した。

旧:

```python
tl = (tl_per * terminal_weights).sum()
```

新:

```python
w_sum = terminal_weights.sum().clamp_min(1e-8)
tl = (tl_per * terminal_weights).sum() / w_sum
```

この修正により gradient balance は改善したが、60-cycle probe では性能が悪化した。

```text
runs/20260507_stage2a_gradnorm60_cq0285_P100_scaled_seed42_5aaf163a

final  2.295
best   2.165
tail10 2.374
tail20 2.361
```

### 2.3 exp_027

`terminal / yaku / value` の coef を同倍率で 10x / 50x / 100x 上げたが、
`COEF50x` が最良でも `exp_026 seed42 baseline` には届かなかった。

```text
COEF50x:
final  2.245
best   2.055
tail10 2.292
tail20 2.259
```

重要なのは、exp_027 では **ratio はほぼ動いていない** ことである。

```text
late T/Y:
CQ0285 base coef  6.75
COEF10x           5.96
COEF50x           7.15
COEF100x          7.43
```

つまり、coef を全部上げても

```text
terminal : yaku : value
```

の比率はほぼ変わらず、CQ-0285 前の

```text
terminal >>> yaku/value
```

の状態には戻らない。

### 2.4 ClaudeCode review

`experiments/Stage02_CallUnlock/exp_027/claude_code_review.md` の要点:

1. CQ-0285 前の terminal gradient 支配は scale 不整合だったが、同時に
   `value_trunk` を outcome-sensitive に shaping する主信号でもあった
2. `reward.point_delta_scale=0.0001` 後は `value_loss` が極小で、
   `value_trunk` を強く学習させる dense supervision は事実上 terminal に偏っている
3. 次の最重要 probe は、
   **CQ-0285 は維持したまま `terminal_loss_coef` だけを大きく上げること**

## 3. 実験の狙い

今回の実験では、`yaku` と `value` を動かさず、
**terminal だけの ratio restoration** を見る。

見たいこと:

1. `terminal_loss_coef` だけを上げると `exp_026` の性能に戻るか
2. gradient norm 比 `T/Y`, `T/V` が CQ-0285 前の水準に近づくか
3. terminal-driven shaping 仮説が支持されるか

支持された場合:

```text
「CQ-0285 の平均化自体は良いが、terminal signal を別途戻す必要がある」
```

支持されない場合:

```text
terminal absolute scale 以外の要因
（semantic summary 構造、semantic_proj dead weight、alpha 正規化など）
を次に疑う
```

## 4. 実験条件

Base は CQ-0285 適用後の `P100 scaled`。

固定条件:

- config: `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- seed: `42`
- `reward.point_delta_scale = 0.0001`
- `selfplay.policy_ratio = 1.0`
- `training.rule_mix_learner.ppo_mode = "separated"`
- `training.policy_anchor.enabled = false`
- `training.lr = 0.0001`
- `training.clip_epsilon = 0.15`
- `training.entropy_coef = 0.0`
- `training.value_loss_coef = 0.125`
- `training.semantic_aux.yaku_loss_coef = 0.05`
- `training.multi_cycle.num_cycles = 60`
- `training.multi_cycle.selfplay_matches_per_cycle = 200`
- `training.diagnostics.gradient_norms.enabled = true`
- `training.diagnostics.gradient_norms.max_batches_per_epoch = 4`

Sweep:

| label | terminal_loss_coef | intent |
|---|---:|---|
| `TERM30x` | `3.0` | ClaudeCode 推奨本命。CQ-0285 前の T/Y 水準に近づける |
| `TERM50x` | `5.0` | 少し強めの上限確認。TERM30x で足りない場合に備える |

base との関係:

```text
base terminal_loss_coef = 0.1
TERM30x = 30x
TERM50x = 50x
```

想定される late ratio の目安:

```text
base late T/Y ≈ 6.7
TERM30x なら概算で T/Y ≈ 200
TERM50x なら概算で T/Y ≈ 330
```

この値は CQ-0285 前 probe の

```text
T/Y ≈ 128-318
```

に近いか、やや上回る帯である。

## 5. 実行条件の位置づけ

この実験はまだ本採用条件の決定ではなく、**1seed hypothesis probe** である。

判定基準:

- `TERM30x` が `exp_026 seed42 baseline` に近づく、または上回る
  - 仮説支持。次は 3seed 実験化
- `TERM30x` が不十分で `TERM50x` のみ改善
  - terminal signal は必要だが、戻す量が多い
- 両方とも改善しない
  - terminal absolute scale だけでは説明できない
  - 次は alpha 正規化や semantic summary 構造を調べる

## 6. 実行コマンド

ログ出力先:

```bash
mkdir -p experiments/Stage02_CallUnlock/exp_028/driver_logs
```

### TERM30x

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_core_minimal_mixed_s1_baseline.yaml \
  --base-dir runs \
  --override \
    experiment.name='"stage2a_exp028_terminal30x_seed42"' \
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
    training.value_loss_coef=0.125 \
    training.multi_cycle.num_cycles=60 \
    training.multi_cycle.selfplay_matches_per_cycle=200 \
    training.semantic_aux.enabled=true \
    training.semantic_aux.terminal_loss_coef=3.0 \
    training.semantic_aux.yaku_loss_coef=0.05 \
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
  2>&1 | tee experiments/Stage02_CallUnlock/exp_028/driver_logs/term30x_seed42.log
```

### TERM50x

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_core_minimal_mixed_s1_baseline.yaml \
  --base-dir runs \
  --override \
    experiment.name='"stage2a_exp028_terminal50x_seed42"' \
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
    training.value_loss_coef=0.125 \
    training.multi_cycle.num_cycles=60 \
    training.multi_cycle.selfplay_matches_per_cycle=200 \
    training.semantic_aux.enabled=true \
    training.semantic_aux.terminal_loss_coef=5.0 \
    training.semantic_aux.yaku_loss_coef=0.05 \
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
  2>&1 | tee experiments/Stage02_CallUnlock/exp_028/driver_logs/term50x_seed42.log
```

## 7. 見るべき指標

### 7.1 Performance

- final `avg_rank`
- best `avg_rank`
- `tail10`, `tail20`
- final `win_rate`
- final `deal_in_rate`
- final `avg_score`

比較基準:

```text
exp026 seed42 base:
final  1.970
best   1.970
tail10 2.124
tail20 2.137
```

### 7.2 PPO diagnostics

- `clip_fraction`
- `log_ratio_p01 / p95 / p99`
- `ratio_max`
- `entropy`
- `max_prob_p95 / p99`

目的:

- terminal 強化で性能が戻っても PPO update が壊れていないか確認する

### 7.3 Gradient norm diagnostics

最重要:

- `gradient_norms.aggregate.ratios.value_semantic_weighted_terminal_to_weighted_yaku`
- `gradient_norms.aggregate.ratios.value_semantic_weighted_terminal_to_weighted_value`
- `weighted_terminal_loss.value_semantic.mean`
- `weighted_yaku_loss.value_semantic.mean`
- `weighted_value_loss.value_semantic.mean`

解釈:

- `TERM30x` / `TERM50x` で performance が戻るなら
  - terminal-driven trunk shaping 仮説を支持
- ratio が戻っても performance が戻らないなら
  - 係数ではなく構造要因が本命

## 8. 実行後の集計コマンド

```bash
python3 - <<'PY'
import json
from pathlib import Path

run_names = [
    ("exp026_base", Path("runs/20260503_stage2a_rewardscale_probe_P100_seed42_dd0b0c5d")),
    ("cq0285_base", Path("runs/20260507_stage2a_gradnorm60_cq0285_P100_scaled_seed42_5aaf163a")),
]

for pat, label in [
    ("*stage2a_exp028_terminal30x_seed42*", "term30x"),
    ("*stage2a_exp028_terminal50x_seed42*", "term50x"),
]:
    matches = sorted(Path("runs").glob(pat), key=lambda p: p.stat().st_mtime)
    if matches:
        run_names.append((label, matches[-1]))

def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

for label, run in run_names:
    s = json.loads((run / "summary.json").read_text())
    cycles = s["phase_stats"]["cycles"]
    ranks = [c.get("eval_metrics", {}).get("avg_rank") for c in cycles]
    wins = [c.get("eval_metrics", {}).get("win_rate") for c in cycles]
    deals = [c.get("eval_metrics", {}).get("deal_in_rate") for c in cycles]
    scores = [c.get("eval_metrics", {}).get("avg_score") for c in cycles]
    tail10 = avg(ranks[-10:])
    tail20 = avg(ranks[-20:])
    best = min(r for r in ranks if r is not None)
    final = ranks[-1]
    print()
    print(label, run)
    print(" final =", final)
    print(" best  =", best)
    print(" tail10=", tail10)
    print(" tail20=", tail20)
    print(" final_win =", wins[-1])
    print(" final_deal=", deals[-1])
    print(" final_score=", scores[-1])
    last = cycles[-1]["learner_metrics"]["ppo_diag"]["gradient_norms"]["aggregate"]
    ratios = last["ratios"]
    print(" late T/Y =", ratios.get("value_semantic_weighted_terminal_to_weighted_yaku"))
    print(" late T/V =", ratios.get("value_semantic_weighted_terminal_to_weighted_value"))
PY
```

## 9. 期待される次アクション

### ケースA: TERM30x が明確に改善

- `exp_026 seed42 baseline` に近づく / 上回る
- 次は `TERM30x` の 3seed 実験化

### ケースB: TERM50x のみ改善

- terminal signal は必要だが、戻す量がより大きい
- 次は `TERM50x` を基準に再確認

### ケースC: 両方ダメ

- absolute scale だけでは戻らない
- 次候補:
  1. CQ-0285 alpha 正規化
  2. `semantic_proj` の dead weight 修正
  3. semantic summary 構造見直し

## 10. 補足

今回の実験は、`semantic_proj` 修正や CQ-0285 revert 実験の前に、
**最小変更で terminal-driven shaping 仮説を切る**ためのものとして位置づける。

つまり今回は、

```text
「構造をいじる前に、terminal signal だけ戻せば性能が戻るのか」
```

を確認することが主目的である。
