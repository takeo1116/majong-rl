# Experiment Runbook: exp_040

作成日: 2026-05-13  
Stage: `Stage02_CallUnlock` / `Stage02b_OptionalAllGamma095`

## 1. 目的

`exp_040` は、optional action unlock 後の性能低下について、`gamma=0.5` が小さすぎるという仮説を検証する。

ClaudeCode review (`experiments/Stage02_CallUnlock/exp_039/claude_code_review.md`) では、`RIICHI_OPTIONAL` などの optional branch 挿入により、実際の engine step reward が従来の discard decision から 1 decision 後ろへずれる可能性が指摘された。

`gamma=0.5` では 1 decision 遅れただけで reward signal が半減するため、optional action unlock 後の性能低下を過剰に悪化させている可能性がある。そこで `gamma=0.95 / gae_lambda=0.95` に戻して、optional-all 条件の性能がどこまで回復するかを見る。

## 2. 事前結果: seed42 probe

seed42 は runbook 作成前に直接コマンドで実行済み。

| label | seed | run_dir |
|---|---:|---|
| `GAMMA095_OPTIONAL_ALL_seed42` | 42 | `runs/20260513_20260513_stage2b_optional_all_gamma095_probe_seed42_cf98d62a` |

結果:

| metric | value |
|---|---:|
| final avg_rank | 2.235 |
| best avg_rank | 2.065 |
| best cycle | 54 |
| tail10 avg_rank | 2.2465 |
| tail20 avg_rank | 2.2518 |
| final win_rate | 0.2258 |
| final deal_in_rate | 0.2041 |

この seed42 結果は、旧 optional-all `gamma=0.5` より明確に良い。  
一方で、ルール拡張前の `exp_034` や現コード optional-off の `exp_039` にはまだ少し届いていない。

## 3. 今回回す条件

seed43/44 を追加で回し、seed42 と合わせて 3seed 評価にする。

| parameter | value |
|---|---:|
| seeds | 43, 44 |
| cycles | 60 |
| selfplay_matches_per_cycle | 200 |
| optional_riichi / tsumo / ron / ankan / kakan / kyuushu | all true |
| gamma | 0.95 |
| gae_lambda | 0.95 |
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
| feature_encoder.riichi_discard_mask | true |
| semantic aux | enabled |

## 4. 比較対象

### ルール拡張前 baseline (`exp_034`)

| metric | 3seed mean |
|---|---:|
| final | 2.078 |
| best | 1.998 |
| tail10 | 2.113 |
| tail20 | 2.122 |

### 現コード optional-off baseline (`exp_039`)

| metric | 3seed mean |
|---|---:|
| final | 2.128 |
| best | 2.018 |
| tail10 | 2.156 |
| tail20 | 2.169 |

### 旧 optional-all (`exp_036`, gamma=0.5, seed42)

| metric | value |
|---|---:|
| final | 2.360 |
| best | 2.125 |
| tail10 | 2.330 |
| tail20 | 2.319 |

## 5. 判定基準

| 結果 | 解釈 |
|---|---|
| 3seed tail20 が `2.15-2.20` 近辺まで戻る | `gamma=0.5` が主因だった可能性が高い。optional-all を維持して次へ進める。 |
| best は良いが tail/final が弱い | `gamma` は改善するが late drift / optional residual cost が残る。追加調整が必要。 |
| seed42 だけ良く、seed43/44 が弱い | gamma 効果は不安定。3seed平均で判断する。 |
| `exp036` と大差ない | S1仮説は弱い。reward attribution の構造修正や optional family 別設計を再検討。 |

## 6. Driver

seed43/44 だけを回す。

```bash
./.venv/bin/python scripts/local/stage2/exp_040_driver.py
```

validate-only:

```bash
EXP040_VALIDATE_ONLY=1 ./.venv/bin/python scripts/local/stage2/exp_040_driver.py
```

再実行:

```bash
EXP040_FORCE_RERUN=1 ./.venv/bin/python scripts/local/stage2/exp_040_driver.py
```

単体実行:

```bash
EXP040_ONLY=GAMMA095_OPTIONAL_ALL_seed43 ./.venv/bin/python scripts/local/stage2/exp_040_driver.py
EXP040_ONLY=GAMMA095_OPTIONAL_ALL_seed44 ./.venv/bin/python scripts/local/stage2/exp_040_driver.py
```

Run map:

```text
experiments/Stage02_CallUnlock/exp_040/run_map.json
```

Driver logs:

```text
experiments/Stage02_CallUnlock/exp_040/driver_logs/
```

## 7. 完了後に見るもの

- final / best / tail10 / tail20 avg_rank
- seed42/43/44 の 3seed mean/std
- optional_decision_count と family breakdown
- riichi_bypass_rate が引き続き低いか
- target_kl stop count / clip_fraction / max_prob / entropy
- win_rate と deal_in_rate の内訳

