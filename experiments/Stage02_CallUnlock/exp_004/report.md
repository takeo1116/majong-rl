# Experiment Report: exp_004

作成日: 2026-03-30  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_004/runbook.md`
- `experiments/Stage02_CallUnlock/exp_004/run_map.json`
- `experiments/Stage02_CallUnlock/exp_002/report.md`
- `experiments/Stage02_CallUnlock/exp_003/report.md`

## 1. 要約

`exp_004` では、Stage02a A `core_minimal` を固定して、`mixed` PPO を安定化させる条件を探索した。

比較条件:

1. M1 `policy_ratio=0.50, baseline_sample_weight=0.50`
2. M2 `policy_ratio=0.50, baseline_sample_weight=0.25`
3. M3 `M2 + policy_anchor.coef=1.0`
4. M4 `M3 + lr=1e-4 + clip_epsilon=0.10 + max_grad_norm=0.30`

結論は以下。

- M1 / M2 は明確に失敗した
- M3 は部分改善したが、後半 cycle で still unstable だった
- **M4 は 20 cycle を通して stable に mixed PPO を回せた**
- 今回の mixed 安定化に一番効いたのは、`policy_ratio` や `baseline_sample_weight` 単独ではなく、**PPO update strength を下げたこと**と読むのが自然
- M4 は 1 seed では `separated` control よりも良い最終 eval を示しており、以後の mixed baseline 候補として扱う価値が高い

## 2. 実験目的

`exp_002` では、Stage02a A `core_minimal` を用いて

- `ppo_mode="mixed"`
- `ppo_mode="separated"`

を比較した結果、`separated` は stable、`mixed` は unstable だった。

`exp_003` ではその stable `separated` baseline 上で A/B/C feature 比較を再開できたが、Stage1 parity の観点では、
「baseline 打牌を含む mixed PPO を stable に回せること」はまだ未達だった。

したがって `exp_004` の目的は、**Stage02a で mixed PPO を stable に回す最初の条件を見つけること**である。

## 3. 実行条件

固定条件:

- A `core_minimal`
- `training.rule_mix_learner.ppo_mode = "mixed"`
- `training.multi_cycle.num_cycles = 20`
- `training.multi_cycle.eval_each_cycle = true`
- `training.imitation_eval.enabled = true`
- `training.imitation_eval.num_matches = 50`
- `training.policy_anchor.reference = "imitation_fixed"`

共通 config:

- `configs/stage2a_core_minimal_mixed_search_baseline.yaml`

実行管理:

- `experiments/Stage02_CallUnlock/exp_004/run_map.json`
- `scripts/local/stage2/exp_004_driver.py`

参照点として、以下の既存 run を流用した。

### Stable separated control

- source: `exp_002` の A2 `separated` control
- final `avg_rank=2.555`, `win_rate=0.2312`
- final `ratio_mean=1.0035`, `clip_fraction=0.2462`, `anchor_kl_discard=0.0688`

### Unstable mixed reference

- source: `exp_002` の A1 `mixed` reference
- final `avg_rank=3.45`, `win_rate=0.0484`
- final `ratio_mean=413438`, `clip_fraction=0.4898`, `anchor_kl_discard=5.6866`

## 4. 対象 run

### M1 `pr050_bsw05`

- run label: `M1_pr050_bsw05` （対応は `experiments/Stage02_CallUnlock/exp_004/run_map.json` を参照）

### M2 `pr050_bsw025`

- run label: `M2_pr050_bsw025` （対応は `experiments/Stage02_CallUnlock/exp_004/run_map.json` を参照）

### M3 `pr050_bsw025_anchor10`

- run label: `M3_pr050_bsw025_anchor10` （対応は `experiments/Stage02_CallUnlock/exp_004/run_map.json` を参照）

### M4 `pr050_bsw025_anchor10_lr1e4_clip010_gn03`

- run label: `M4_pr050_bsw025_anchor10_lr1e4_clip010_gn03` （対応は `experiments/Stage02_CallUnlock/exp_004/run_map.json` を参照）

## 5. 主結果

### imitation 直後 eval と final eval

| Condition | imitation avg_rank | imitation win_rate | final avg_rank | final win_rate |
|---|---:|---:|---:|---:|
| M1 | 2.505 | 0.2293 | 3.100 | 0.0931 |
| M2 | 2.505 | 0.2293 | 3.370 | 0.0718 |
| M3 | 2.505 | 0.2293 | 3.330 | 0.0905 |
| M4 | 2.370 | 0.2317 | 2.230 | 0.2487 |

解釈:

- M1 / M2 は imitation 直後は悪くないが、PPO を回すほど大きく悪化した
- M3 は `cycle_00` までは改善傾向を示したが、後半で崩れた
- M4 は imitation 直後でも悪くなく、**PPO 後にさらに改善した**

### cycle_00 の確認

| Condition | cycle_00 avg_rank | cycle_00 win_rate |
|---|---:|---:|
| M1 | 2.570 | 0.2134 |
| M2 | 2.565 | 0.2161 |
| M3 | 2.395 | 0.2312 |
| M4 | 2.360 | 0.2506 |

M3 / M4 は早い段階では改善しているが、M3 は維持に失敗し、M4 だけが維持と上積みの両方に成功した。

### 後半 5 cycle 平均

| Condition | tail-5 avg_rank | tail-5 win_rate | tail-5 policy_loss |
|---|---:|---:|---:|
| M1 | 3.124 | 0.1064 | 910.53 |
| M2 | 3.231 | 0.0937 | 1406.33 |
| M3 | 3.089 | 0.1200 | 19.62 |
| M4 | 2.395 | 0.2386 | 0.00775 |

この比較から、M4 だけが後半でも PPO を正常域に保っていることが分かる。

## 6. PPO 安定性

### final PPO diagnostics

| Condition | ratio_mean | clip_fraction | anchor_kl_discard | anchor_kl_optional |
|---|---:|---:|---:|---:|
| M1 | 8.40e8 | 0.4047 | 4.3720 | 0.00769 |
| M2 | 7.04e7 | 0.4007 | 4.1895 | 0.00905 |
| M3 | 1.17e4 | 0.3791 | 2.9179 | 0.00461 |
| M4 | 1.0280 | 0.3053 | 0.0165 | 0.00101 |

解釈:

- `policy_ratio=0.50` への引き上げだけでは安定化しない
- `baseline_sample_weight=0.25` へ下げても still unstable
- `anchor=1.0` は崩壊を少し遅らせるが、十分ではない
- **`lr`, `clip_epsilon`, `max_grad_norm` を落とした M4 だけが PPO diagnostics を正常域に戻した**

### policy_loss の挙動

final `policy_loss`:

- M1: `608.52`
- M2: `3410.06`
- M3: `9.22`
- M4: `0.00675`

`M3` でも一見かなり改善しているが、eval と KL が still unstable であり、安定 run と呼ぶには足りない。
M4 は `policy_loss` も `ratio_mean` も `separated` control に近い水準まで戻っている。

## 7. separated control との比較

### final 値

| Condition | final avg_rank | final win_rate | final ratio_mean | final anchor_kl_discard |
|---|---:|---:|---:|---:|
| separated control | 2.555 | 0.2312 | 1.0035 | 0.0688 |
| unstable mixed reference | 3.450 | 0.0484 | 413438 | 5.6866 |
| M4 | 2.230 | 0.2487 | 1.0280 | 0.0165 |

1 seed のため断定は避けるべきだが、M4 は少なくとも

- unstable mixed reference を明確に脱した
- separated control よりも見た目上は良い

という結果になった。

したがって、**M4 は Stage02a mixed の実用的な第一候補**として扱う価値がある。

## 8. throughput 観点

imitation chunk train 時間は全条件でほぼ同水準だった。

- M1: `106s, 109s, 113s`
- M2: `112s, 118s, 108s`
- M3: `103s, 104s, 110s`
- M4: `104s, 106s, 108s`

つまり、M4 の安定化は大きな wall-clock penalty を伴っていない。
今回 mixed を救ったのは、速度を大きく落とす安全策ではなく、妥当な update strength の調整だったと見てよい。

## 9. 結論

今回の `exp_004` から得られた結論は以下。

1. `mixed` の安定化には、`policy_ratio` や `baseline_sample_weight` の調整だけでは足りない
2. `anchor` 強化は補助的には効くが、それだけでは十分でない
3. **決定打は PPO update strength を下げることだった**
4. Stage02a mixed の暫定有効条件としては、現時点では **M4** が最有力

より具体的には、以下の組み合わせが mixed 安定化候補となる。

- `policy_ratio=0.50`
- `baseline_sample_weight=0.25`
- `policy_anchor.coef=1.0`
- `lr=1e-4`
- `clip_epsilon=0.10`
- `max_grad_norm=0.30`

## 10. 次のアクション

この結果を受けて、次に自然なのは以下のどちらかである。

1. **M4 の最小有効条件を探す**
   - `lr` だけ戻す
   - `clip_epsilon` だけ戻す
   - `max_grad_norm` だけ戻す
   を順に試し、どの knob が本当に効いているかを切る

2. **M4 を 2-3 seed で再確認する**
   - mixed が本当に再現性を持って stable か
   - separated control に対する優位が seed を変えても残るか
   を確認する

改善を進める目的に照らすと、まずは **最小有効条件の切り分け** を先にやる方が価値が高い。
M4 は 3 つの update-strength 系 knob を同時に動かしているため、ここを分解できると、今後の mixed baseline をより強く、より説明可能にできる。
