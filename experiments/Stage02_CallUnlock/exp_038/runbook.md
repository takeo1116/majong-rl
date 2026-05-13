# Experiment Runbook: exp_038

作成日: 2026-05-11  
Stage: `Stage02_CallUnlock` / `Stage02b_ModelCapacityProbe`

## 1. 目的

`exp_038` は、ルール拡張後の性能低下がモデル容量不足で説明できるかを調べる 1seed probe である。

`exp_036` / `exp_037` では optional action を開放すると optional-off baseline に届かなかった。CQ-0295 audit では optional binary family 自体はほぼ完全に teacher label を学べており、明確な実装バグというより、decision process / input / branch が増えたことによる表現・最適化負荷が疑われる。

そこで、まず value/semantic trunk の容量を増やして、以下を切り分ける。

- optional-off でも単純に容量増で伸びるのか
- RII_ONLY のような軽いルール忠実度拡張が容量増で回復するのか
- WIDE1 と WIDE2 の差があるのか

## 2. 実験条件

3 条件を seed42 で 1 本ずつ回す。

| label | optional flags | value_hidden_dims | 目的 |
|---|---|---:|---|
| `OFF_WIDE1_seed42` | all off | `[384,192]` | optional-off baseline が容量増で伸びるか |
| `OFF_WIDE2_seed42` | all off | `[512,256]` | さらに大きい容量で伸びる/壊れるか |
| `RII_WIDE1_seed42` | riichi only | `[384,192]` | Riichi optional が容量増で baseline に近づくか |

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
| value/semantic lr | 0.01 |
| target_kl | 0.03 |
| target_kl stop_multiplier | 1.5 |
| target_kl skip | true |
| gradient_norms | enabled |
| `feature_encoder.tile_presence_flags` | true |
| `feature_encoder.riichi_discard_mask` | true |
| semantic aux | enabled |

注意: 今回の WIDE は `model.value_hidden_dims` の拡張であり、discard/optional policy trunk の hidden dims は既存 config のままにする。まず value/semantic 容量が律速かを見るため。

## 3. 比較対象

主な比較対象:

| run | condition | cycles | final | best | tail10 |
|---|---|---:|---:|---:|---:|
| exp034 seed42 | optional off, `[256,128]` | first 30 | 2.310 | 2.030 | 2.215 |
| exp037 RII_ONLY | riichi only, `[256,128]` | 30 | 2.315 | 2.140 | 2.277 |

exp034 full 60cycle seed42:

| metric | value |
|---|---:|
| final avg_rank | 1.960 |
| best avg_rank | 1.960 |
| tail10 | 2.098 |
| tail20 | 2.098 |

## 4. 判定基準

| 結果 | 解釈 |
|---|---|
| OFF_WIDE1/2 が exp034 first30 を上回る | ルール拡張前から容量不足の可能性 |
| RII_WIDE1 だけ改善 | optional_riichi の追加負荷が容量で吸収できる可能性 |
| OFF は改善、RII は改善しない | optional_riichi の design/curriculum 問題が濃い |
| WIDE2 が WIDE1 より悪化 | 容量増による過学習/最適化悪化または時間不足 |
| どれも改善しない | 容量より optional decision design が主因 |

この probe は 30cycle なので、最終判断ではなく候補選別として扱う。

## 5. Driver

連続実行:

```bash
./.venv/bin/python scripts/local/stage2/exp_038_driver.py
```

個別実行:

```bash
EXP038_ONLY=OFF_WIDE1_seed42 ./.venv/bin/python scripts/local/stage2/exp_038_driver.py
EXP038_ONLY=OFF_WIDE2_seed42 ./.venv/bin/python scripts/local/stage2/exp_038_driver.py
EXP038_ONLY=RII_WIDE1_seed42 ./.venv/bin/python scripts/local/stage2/exp_038_driver.py
```

validate-only:

```bash
EXP038_VALIDATE_ONLY=1 ./.venv/bin/python scripts/local/stage2/exp_038_driver.py
```

失敗しても次へ進める場合:

```bash
EXP038_STOP_ON_ERROR=0 ./.venv/bin/python scripts/local/stage2/exp_038_driver.py
```

Run map:

```text
experiments/Stage02_CallUnlock/exp_038/run_map.json
```

Driver logs:

```text
experiments/Stage02_CallUnlock/exp_038/driver_logs/
```

## 6. 確認項目

完了後に見るもの:

- final / best / tail10 / tail20 avg_rank
- selfplay stats の total_steps / num_rounds / call_count
- RII_WIDE1 の `riichi_opportunity_*` / bypass rate
- learner `ppo_diag.decision_family` があれば family 別 update 状況
- gradient_norms の value/semantic group が WIDE で極端に暴れていないか

## 7. 次の分岐

- OFF_WIDE1 が良く、OFF_WIDE2 が同等以上なら WIDE1/2 どちらかを 60cycle または 3seed 化する。
- RII_WIDE1 が明確に良ければ、Riichi optional を残す方向で 3seed 化する。
- RII_WIDE1 が悪い場合は、モデル容量より Riichi optional curriculum / automation hybrid を優先する。
