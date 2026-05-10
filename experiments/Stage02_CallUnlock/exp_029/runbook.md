# Experiment Runbook: exp_029

作成日: 2026-05-08  
Stage: `Stage02_CallUnlock`

## 1. 目的

`exp_029` の目的は、CQ-0286 で実装した optimizer parameter group lr split を使い、`terminal_loss_coef=5.0` に頼らずに `value_semantic` 側の表現学習を速めることで `TERM50x` 相当の性能に近づけるかを確認することである。

`exp_028 TERM50x` は 3seed で強かったが、設計としては terminal loss だけを非常に大きくしている。

```text
TERM50x:
terminal_loss_coef = 5.0
yaku_loss_coef     = 0.05
value_loss_coef    = 0.125
```

今回の実験では、loss coefficient は CQ-0285 base に戻し、optimizer の learning rate だけを分ける。

```text
policy lr は固定
value_trunk / value_head / terminal_head / yaku_head / semantic_proj の lr だけを上げる
```

## 2. 背景

### 2.1 exp028 TERM50x

`TERM50x` は seed42/43/44 の 3seed で再現した。

```text
P100 scaled exp026 3seed:
final  2.0817
best   1.9600
best10 2.0422
tail10 2.1203
tail20 2.1292

TERM50x exp028 3seed:
final  2.1417
best   1.9333
best10 2.0263
tail10 2.1095
tail20 2.1017
```

`TERM50x` は final では弱いが、best / best10 / tail10 / tail20 では exp026 を上回った。

解釈:

```text
terminal-driven value_trunk shaping は performance に効いている。
```

### 2.2 ただし設計上の違和感

`terminal_loss_coef=5.0` は、実効スケールを戻す補正としては理解できるが、設定だけ見ると terminal を value/yaku より極端に重く扱っている。

できれば以下のように説明できる形へ寄せたい。

```text
terminal を異常に重くするのではなく、
value/outcome representation 側の学習速度を上げる。
```

### 2.3 CQ-0286

CQ-0286 で Stage2a learner に optimizer parameter group lr split を追加した。

config:

```yaml
training:
  lr: 0.0001
  lr_groups:
    enabled: true
    policy: 0.0001
    value_semantic: 0.0003
```

group:

```text
policy:
  discard_trunk
  discard_head
  optional_trunk
  candidate_encoder
  optional_scorer
  direct hint scorer/gate系

value_semantic:
  value_trunk
  value_head
  terminal_head
  yaku_head
  semantic_proj
```

## 3. 実験仮説

仮説:

```text
CQ-0285 base coef のままでも、
value_semantic_lr を上げれば value/outcome representation が速く育ち、
TERM50x に近い性能が得られる。
```

成功した場合:

```text
TERM50x の強さは「terminalを重くする必要がある」ではなく、
「value_semantic側の表現学習速度が足りなかった」と解釈できる。
```

失敗した場合:

```text
単に value_semantic 側を速くするだけでは足りず、
terminal-specific loss signal の比率そのものが重要だった可能性が高い。
```

## 4. 実験条件

Base: CQ-0285 適用後の `P100 scaled`。

固定:

- seed: `42`
- `policy_ratio = 1.0`
- `ppo_mode = "separated"`
- `policy_anchor.enabled = false`
- `reward.point_delta_scale = 0.0001`
- `training.lr = 0.0001`
- `training.lr_groups.enabled = true`
- `training.lr_groups.policy = 0.0001`
- `clip_epsilon = 0.15`
- `entropy_coef = 0.0`
- `value_loss_coef = 0.125`
- `terminal_loss_coef = 0.1`
- `yaku_loss_coef = 0.05`
- `multi_cycle.num_cycles = 60`
- `selfplay_matches_per_cycle = 200`
- `gradient_norms.enabled = true`
- `gradient_norms.max_batches_per_epoch = 4`

Sweep:

| label | policy_lr | value_semantic_lr | multiplier | intent |
|---|---:|---:|---:|---|
| `VS_LR3x` | `1e-4` | `3e-4` | 3x | mild split |
| `VS_LR10x` | `1e-4` | `1e-3` | 10x | main probe |
| `VS_LR30x` | `1e-4` | `3e-3` | 30x | aggressive boundary |

## 5. 比較対象

主比較:

```text
exp026 P100 scaled 3seed
exp028 TERM50x 3seed
```

seed42 直接比較:

```text
exp026 seed42:
runs/20260503_stage2a_rewardscale_probe_P100_seed42_dd0b0c5d

CQ0285 base seed42:
runs/20260507_stage2a_gradnorm60_cq0285_P100_scaled_seed42_5aaf163a

TERM50x seed42:
runs/20260507_stage2a_exp028_terminal50x_seed42_df9be40a
```

## 6. 実行コマンド

driver:

```bash
./.venv/bin/python scripts/local/stage2/exp_029_driver.py
```

単発:

```bash
EXP029_ONLY=VS_LR3x ./.venv/bin/python scripts/local/stage2/exp_029_driver.py
EXP029_ONLY=VS_LR10x ./.venv/bin/python scripts/local/stage2/exp_029_driver.py
EXP029_ONLY=VS_LR30x ./.venv/bin/python scripts/local/stage2/exp_029_driver.py
```

## 7. 見るべき指標

### 7.1 Performance

- final avg_rank
- best avg_rank
- best10
- tail10
- tail20
- final win_rate
- final deal_in_rate
- final avg_score

`TERM50x` との比較では、final だけでなく `best10 / tail10 / tail20` を重視する。

### 7.2 Optimizer lr group diagnostics

`learner_metrics.optimizer_lr_groups` で以下を確認する。

- `enabled = true`
- `policy.lr = 0.0001`
- `value_semantic.lr` が条件通り
- param_count / tensor_count が nonzero

### 7.3 PPO / gradient diagnostics

- `clip_fraction`
- `ratio_max`
- `max_prob`
- `T/Y`, `T/V`, `Y/V`
- `weighted_terminal_loss.value_semantic.mean`
- `weighted_yaku_loss.value_semantic.mean`
- `weighted_value_loss.value_semantic.mean`

重要:

```text
lr split は loss coefficient を変えないため、T/Y や T/V の loss由来比率は
TERM50xほど大きくならないはず。
```

それでも性能が戻るなら、terminal比率ではなく value_semantic 側の更新速度が重要だったと読める。

## 8. 判定

### 成功

`VS_LR10x` または `VS_LR30x` が `TERM50x seed42` に近づく。

目安:

```text
best10 <= 2.05
tail10 <= 2.15
tail20 <= 2.15
```

この場合、次は最良条件を 3seed 化する。

### 部分成功

CQ0285 base よりは改善するが、TERM50x には届かない。

この場合:

```text
value_semantic lr split は有効だが、terminal-specific signal も必要。
```

次は `TERM50x` より小さい terminal coef と lr split の組み合わせを見る。

### 失敗

全条件が CQ0285 base と同等または悪化。

この場合:

```text
単に value_semantic 側を速くしても戻らない。
terminal-specific loss ratio が重要。
```

次は `TERM50x` を正式baseline候補として扱いつつ、semantic_proj や alpha 正規化を検討する。
