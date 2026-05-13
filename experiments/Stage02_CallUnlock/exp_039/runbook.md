# Experiment Runbook: exp_039

作成日: 2026-05-12  
Stage: `Stage02_CallUnlock` / `Stage02b_OptionalOffRegressionCheck`

## 1. 目的

`exp_039` は、CQ-0290 以降の現コードで、`exp_034` 相当の optional-off baseline が再現するかを確認する sanity check である。

`exp_036` / `exp_037` / `exp_038` では optional action unlock 後の条件が `exp_034` より弱かった。ただし、その低下が optional action 自体によるものなのか、CQ-0288 以降のモデル/encoder/learner 周辺差分による regression なのかはまだ完全には切れていない。

そのため、まず optional action をすべて OFF に戻し、モデルサイズも `exp_034` 相当に戻して、現コードで性能水準が再現するかを見る。

## 2. 条件

1seed probe として seed42 を 60 cycle 回す。

| parameter | value |
|---|---:|
| seed | 42 |
| cycles | 60 |
| selfplay_matches_per_cycle | 200 |
| optional_riichi / tsumo / ron / ankan / kakan / kyuushu | all false |
| policy_ratio | 1.0 |
| ppo_mode | separated |
| reward.point_delta_scale | 0.0001 |
| policy lr | 0.0001 |
| value/semantic lr | 0.01 |
| model.value_hidden_dims | `[256,128]` |
| target_kl | 0.03 |
| target_kl stop_multiplier | 1.5 |
| target_kl skip | true |
| feature_encoder.tile_presence_flags | true |
| feature_encoder.riichi_discard_mask | false |
| semantic aux | enabled |
| gradient_norms | enabled |

`riichi_discard_mask=false` にする理由: `exp_034` 時点にはこの特徴量がなく、optional-off baseline の再現性を見るには入力次元をなるべく戻す必要があるため。

## 3. 比較対象

主比較は `exp_034` seed42。

| run | cycles | final | best | tail10 | tail20 |
|---|---:|---:|---:|---:|---:|
| exp034 seed42 | 60 | 1.960 | 1.960 | 2.098 | 2.098 |
| exp034 seed42, first30 | 30 | 2.310 | 2.030 | 2.215 | 2.266 |

補助比較:

| run | cycles | final | best | tail10 |
|---|---:|---:|---:|---:|
| exp038 OFF_WIDE1 | 30 | 2.300 | 2.170 | 2.269 |
| exp038 OFF_WIDE2 | 30 | 2.265 | 2.140 | 2.272 |

## 4. 判定基準

| 結果 | 解釈 |
|---|---|
| exp034 seed42 に近い (`tail10≈2.10`, final≈2.0 前後) | 現コードの optional-off baseline は健在。性能低下は optional unlock 側が主因。 |
| 2.20 台で停滞 | optional-off でも現コード差分が影響している可能性。CQ-0288/0290以降の regression 調査が必要。 |
| 2.30 近辺で停滞 | optional 以前に現 baseline が壊れている可能性が高い。ルール拡張を止めて regression 優先。 |

注意: CQ-0288 で `semantic_proj` が削除され、CQ-0291 以降で candidate action type embedding 行数が増えているため、完全な seed 再現ではない。見るべきは曲線の完全一致ではなく、性能水準が戻るかどうか。

## 5. Driver

実行:

```bash
./.venv/bin/python scripts/local/stage2/exp_039_driver.py
```

validate-only:

```bash
EXP039_VALIDATE_ONLY=1 ./.venv/bin/python scripts/local/stage2/exp_039_driver.py
```

再実行:

```bash
EXP039_FORCE_RERUN=1 ./.venv/bin/python scripts/local/stage2/exp_039_driver.py
```

Run map:

```text
experiments/Stage02_CallUnlock/exp_039/run_map.json
```

Driver logs:

```text
experiments/Stage02_CallUnlock/exp_039/driver_logs/
```

## 6. 完了後に見るもの

- final / best / tail10 / tail20 avg_rank
- 10-cycle block mean
- optional_decision_count が 0 であること
- riichi_opportunity は存在しても optional_opened / bypassed が 0 であること
- PPO diagnostics: target_kl stop / clip_fraction / max_prob / entropy
- semantic eval は必要に応じて final checkpoint で実施
