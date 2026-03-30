# Experiment Report: exp_005

作成日: 2026-03-30  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_005/runbook.md`
- `experiments/Stage02_CallUnlock/exp_005/run_map.json`
- `experiments/Stage02_CallUnlock/exp_004/report.md`

## 1. 要約

`exp_005` では、`exp_004` で stable だった M4 の mixed PPO 条件について、
どの update-strength ノブが本当に必要かを切り分けた。

比較したのは以下の 6 条件である。

1. 必要性テスト
   - N1 `no_low_lr`
   - N2 `no_low_clip`
   - N3 `no_tight_gradclip`
2. 単独有効性テスト
   - S1 `low_lr_only`
   - S2 `low_clip_only`
   - S3 `tight_gradclip_only`

結論はかなり明確だった。

- **`low_lr` は必要**
- **`low_lr` は単独でも十分**
- `low_clip` は必要ではなく、単独でも不十分
- `tight_gradclip` も必要ではなく、単独でも不十分

したがって、今回の mixed 安定化の主因は **`lr=3e-4 -> 1e-4`** への引き下げと判断するのが自然である。

現時点の最小有効条件は、実質的に **S1** である。

## 2. 実験目的

`exp_004` では、Stage02a A `core_minimal` を固定した mixed PPO の安定化探索を行い、
以下の M4 条件が stable であることを確認した。

- `policy_ratio=0.50`
- `baseline_sample_weight=0.25`
- `policy_anchor.coef=1.0`
- `lr=1e-4`
- `clip_epsilon=0.10`
- `max_grad_norm=0.30`

ただし、M4 は 3 つの update-strength 系ノブを同時に変更していた。

1. `lr`
2. `clip_epsilon`
3. `max_grad_norm`

そこで `exp_005` では、

- どのノブが **必要条件** か
- どのノブが **単独でも効く** か

を切ることを目的とした。

## 3. 実行条件

固定条件:

- A `core_minimal`
- `training.rule_mix_learner.ppo_mode = "mixed"`
- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix_learner.baseline_sample_weight = 0.25`
- `training.policy_anchor.coef = 1.0`
- `training.policy_anchor.reference = "imitation_fixed"`
- `training.multi_cycle.num_cycles = 20`
- `training.multi_cycle.eval_each_cycle = true`
- `training.imitation_eval.enabled = true`

共通 config:

- `configs/stage2a_core_minimal_mixed_minimality_baseline.yaml`

実行管理:

- `experiments/Stage02_CallUnlock/exp_005/run_map.json`
- `scripts/local/stage2/exp_005_driver.py`

参照点:

### unstable anchor-only reference (M3)

- `exp_004` の `M3_pr050_bsw025_anchor10`
- final `avg_rank=3.33`, `win_rate=0.0905`
- final `ratio_mean=1.17e4`, `anchor_kl_discard=2.9179`

### stable mixed candidate (M4)

- `exp_004` の `M4_pr050_bsw025_anchor10_lr1e4_clip010_gn03`
- final `avg_rank=2.23`, `win_rate=0.2487`
- final `ratio_mean=1.0280`, `anchor_kl_discard=0.0165`

## 4. 対象 run

### Necessary tests

- N1: `N1_no_low_lr`
- N2: `N2_no_low_clip`
- N3: `N3_no_tight_gradclip`
  - 対応は `experiments/Stage02_CallUnlock/exp_005/run_map.json` を参照

### Single-effect tests

- S1: `S1_low_lr_only`
- S2: `S2_low_clip_only`
- S3: `S3_tight_gradclip_only`
  - 対応は `experiments/Stage02_CallUnlock/exp_005/run_map.json` を参照

## 5. 主結果

### 最終値

| Condition | final avg_rank | final win_rate | final policy_loss | ratio_mean | clip_fraction | anchor_kl_discard |
|---|---:|---:|---:|---:|---:|---:|
| N1 no_low_lr | 2.600 | 0.2043 | 0.8641 | 42.59 | 0.3960 | 1.5770 |
| N2 no_low_clip | 2.200 | 0.2471 | 0.0070 | 1.0220 | 0.2211 | 0.0186 |
| N3 no_tight_gradclip | 2.260 | 0.2573 | 0.0065 | 1.0145 | 0.2932 | 0.0161 |
| S1 low_lr_only | 2.325 | 0.2568 | 0.0082 | 1.0310 | 0.2340 | 0.0198 |
| S2 low_clip_only | 2.930 | 0.1578 | 0.1713 | 20.72 | 0.3906 | 1.3492 |
| S3 tight_gradclip_only | 2.650 | 0.1781 | 160.1680 | 1620.80 | 0.3671 | 1.1405 |
| M4 reference | 2.230 | 0.2487 | 0.0068 | 1.0280 | 0.3053 | 0.0165 |

### imitation 直後との比較

| Condition | imitation avg_rank | imitation win_rate | final avg_rank | final win_rate |
|---|---:|---:|---:|---:|
| N1 | 2.595 | 0.2207 | 2.600 | 0.2043 |
| N2 | 2.370 | 0.2317 | 2.200 | 0.2471 |
| N3 | 2.315 | 0.2394 | 2.260 | 0.2573 |
| S1 | 2.315 | 0.2394 | 2.325 | 0.2568 |
| S2 | 2.505 | 0.2293 | 2.930 | 0.1578 |
| S3 | 2.595 | 0.2207 | 2.650 | 0.1781 |

ここから見えることは次の通り。

- N2 / N3 / S1 は imitation 直後から大崩れせず、PPO 後も維持または改善
- N1 は imitation 直後から PPO による改善が弱く、後半で drift
- S2 / S3 は imitation 直後から final にかけて明確に悪化

## 6. 必要性テストの読み取り

### N1 `no_low_lr`

`low_lr` を外した条件。

結果:

- final `ratio_mean=42.59`
- final `anchor_kl_discard=1.577`
- final `avg_rank=2.60`, `win_rate=0.204`

これは hard collapse ほどではないが、stable mixed と呼ぶには不十分である。
したがって、**`low_lr` は必要**とみなすのが自然である。

### N2 `no_low_clip`

`low_clip` を外した条件。

結果:

- final `ratio_mean=1.022`
- final `anchor_kl_discard=0.0186`
- final `avg_rank=2.20`, `win_rate=0.247`

M4 とほぼ同水準であり、**`low_clip` は必要条件ではない**と読める。

### N3 `no_tight_gradclip`

`tight_gradclip` を外した条件。

結果:

- final `ratio_mean=1.0145`
- final `anchor_kl_discard=0.0161`
- final `avg_rank=2.26`, `win_rate=0.257`

これも M4 と同水準であり、**`tight_gradclip` も必要条件ではない**。

## 7. 単独有効性テストの読み取り

### S1 `low_lr_only`

`low_lr` だけを M3 に加えた条件。

結果:

- final `ratio_mean=1.031`
- final `anchor_kl_discard=0.0198`
- final `avg_rank=2.325`, `win_rate=0.257`

これは stable mixed とみなしてよい水準であり、**`low_lr` は単独でも十分に効く**。

### S2 `low_clip_only`

`low_clip` だけを M3 に加えた条件。

結果:

- final `ratio_mean=20.72`
- final `anchor_kl_discard=1.349`
- final `avg_rank=2.93`, `win_rate=0.158`

これは unstable であり、**`low_clip` 単独では mixed を救えない**。

### S3 `tight_gradclip_only`

`tight_gradclip` だけを M3 に加えた条件。

結果:

- final `ratio_mean=1620.80`
- final `anchor_kl_discard=1.140`
- final `avg_rank=2.65`, `win_rate=0.178`

こちらも unstable であり、**`tight_gradclip` 単独でも mixed を救えない**。

## 8. 結論

今回の `exp_005` から得られた結論はかなり明確である。

1. **`low_lr` は必要**
2. **`low_lr` は単独でも十分**
3. `low_clip` は必要ではない
4. `tight_gradclip` も必要ではない

したがって、`exp_004` の M4 を mixed 安定化の第一候補としていたが、
`exp_005` を踏まえると **最小有効条件は S1** と整理するのが自然である。

現時点の canonical mixed baseline 候補:

- `policy_ratio=0.50`
- `baseline_sample_weight=0.25`
- `policy_anchor.coef=1.0`
- `lr=1e-4`
- `clip_epsilon=0.15`
- `max_grad_norm=0.50`

この条件は、M4 より軽く、かつ今回の 1 seed では十分 stable だった。

## 9. 解釈

今回の結果は、Stage02a mixed の主な不安定要因が
**`lr=3e-4` の update size** にあったことを示唆している。

`clip_epsilon` と `max_grad_norm` は補助的な safety knob ではあるが、
今回の evidence では主犯ではなかった。

したがって、今後の mixed 実験ではまず

- `low_lr`

を前提にし、

- `low_clip`
- `tight_gradclip`

は必要が出たときだけ追加で締める方がよい。

## 10. 次のアクション

この結果を踏まえて、次に自然なのは以下。

1. **S1 を mixed baseline 候補として branch-swap eval を行う**
   - discard / optional のどちらが本当に改善しているかを切る

2. **S1 を 2-3 seed で再確認する**
   - mixed が再現性を持って stable かを確認する

3. その後、必要なら
   - full -> partial
   - ルール拡張
   - 完全麻雀化
   へ進む

現時点では、改善を前に進める目的に照らして、
**次は S1 で branch-swap eval をやる**のが最も価値が高い。
