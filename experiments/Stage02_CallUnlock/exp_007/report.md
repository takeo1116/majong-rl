# Experiment Report: exp_007

作成日: 2026-03-30  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_007/runbook.md`
- `experiments/Stage02_CallUnlock/exp_007/run_map.json`
- `experiments/Stage02_CallUnlock/exp_006/report.md`
- `experiments/Stage02_CallUnlock/exp_005/report.md`

## 1. 要約

`exp_007` では、S1 mixed baseline から 1 ノブだけを動かして、
**安定を壊さず PPO の改善量を少し強められるか**を確認した。

比較した条件:

- R1: `lr=1.5e-4`
- R2: `lr=2.0e-4`
- R3: `policy_anchor.coef=0.75`

結論はかなり明確だった。

- **R3 が最良**
- `lr` を上げる方向は、`1.5e-4` でも旨みが薄く、`2.0e-4` ではかなり怪しくなる
- 一方で **anchor を `1.0 -> 0.75` に緩める**と、安定性を保ったまま改善量を増やせる可能性が高い

したがって、現時点の mixed baseline 候補は
**S1 から一段進めて R3** とみなすのが自然である。

## 2. 実験目的

`exp_005` により、Stage02a mixed PPO の最小有効条件は S1 と分かった。

S1 条件:

- `policy_ratio=0.50`
- `baseline_sample_weight=0.25`
- `policy_anchor.coef=1.0`
- `lr=1e-4`
- `clip_epsilon=0.15`
- `max_grad_norm=0.50`

`exp_006` では S1 の branch-swap eval を行ったが、
discard / optional の branch 単位改善はかなり小さく、
「PPO が明確に効いている」とまでは言いにくかった。

そこで `exp_007` では、

- 安定性は維持しつつ
- imitation 直後より final がもう少しはっきり良くなる条件

を探すことを目的とした。

## 3. 実行条件

基準 baseline:

- S1 mixed
- config: `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`

固定条件:

- A `core_minimal`
- `training.rule_mix_learner.ppo_mode = "mixed"`
- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix_learner.baseline_sample_weight = 0.25`
- `training.policy_anchor.reference = "imitation_fixed"`
- `training.clip_epsilon = 0.15`
- `training.max_grad_norm = 0.50`
- `training.multi_cycle.num_cycles = 20`
- `training.multi_cycle.eval_each_cycle = true`
- `training.imitation_eval.enabled = true`

実行管理:

- `scripts/local/stage2/exp_007_driver.py`
- `experiments/Stage02_CallUnlock/exp_007/run_map.json`

基準 run:

- S1: `exp_005` の `S1_low_lr_only`（対応は `experiments/Stage02_CallUnlock/exp_005/run_map.json` を参照）

## 4. 対象 run

- R1: `R1_mid_lr_1p5e4`
- R2: `R2_mid_lr_2e4`
- R3: `R3_lower_anchor_075`
  - 対応は `experiments/Stage02_CallUnlock/exp_007/run_map.json` を参照

## 5. 主結果

### imitation 直後と final の比較

| Condition | imitation avg_rank | imitation win_rate | final avg_rank | final win_rate |
|---|---:|---:|---:|---:|
| S1 reference | 2.315 | 0.2394 | 2.325 | 0.2568 |
| R1 mid_lr_1p5e4 | 2.440 | 0.2379 | 2.355 | 0.2422 |
| R2 mid_lr_2e4 | 2.495 | 0.2395 | 2.475 | 0.2404 |
| R3 lower_anchor_075 | 2.315 | 0.2394 | 2.235 | 0.2523 |

### 最終 PPO diagnostics

| Condition | policy_loss | ratio_mean | clip_fraction | anchor_kl_discard | anchor_kl_optional |
|---|---:|---:|---:|---:|---:|
| S1 reference | 0.0082 | 1.0310 | 0.2340 | 0.0198 | 0.00083 |
| R1 mid_lr_1p5e4 | 0.0162 | 1.0561 | 0.2984 | 0.0341 | 0.00187 |
| R2 mid_lr_2e4 | 0.1427 | 1.9983 | 0.3440 | 0.4331 | 0.00257 |
| R3 lower_anchor_075 | 0.0074 | 1.0163 | 0.2265 | 0.0223 | 0.00079 |

## 6. 読み取り

### 6.1 R1 `mid_lr_1p5e4`

R1 は stable ではある。

- `avg_rank`: `2.440 -> 2.355`
- `win_rate`: `0.2379 -> 0.2422`

したがって、PPO は一応効いている。

ただし S1 final と比べると、

- S1 final `avg_rank=2.325`
- R1 final `avg_rank=2.355`
- S1 final `win_rate=0.2568`
- R1 final `win_rate=0.2422`

であり、**S1 を明確に上回ってはいない**。

また PPO diagnostics も S1 より少し荒れている。

- `ratio_mean=1.0561`
- `clip_fraction=0.2984`
- `anchor_kl_discard=0.0341`

つまり、`lr=1.5e-4` は「安全ではあるが、旨みは薄い」条件と読める。

### 6.2 R2 `mid_lr_2e4`

R2 はかなり悪い。

- `avg_rank`: `2.495 -> 2.475`
- `win_rate`: `0.2395 -> 0.2404`

改善量は極めて小さい。

しかも PPO diagnostics は明確に悪化した。

- `policy_loss=0.1427`
- `ratio_mean=1.9983`
- `clip_fraction=0.3440`
- `anchor_kl_discard=0.4331`

`exp_005` の N1 ほど hard collapse ではないが、
**`2.0e-4` は今の mixed に対して既に強すぎる**と見るのが自然である。

### 6.3 R3 `lower_anchor_075`

R3 が今回の best だった。

- `avg_rank`: `2.315 -> 2.235`
- `win_rate`: `0.2394 -> 0.2523`

final `win_rate` は S1 よりわずかに低いが、
final `avg_rank` は明確に良い。

- S1 final `avg_rank=2.325`
- R3 final `avg_rank=2.235`

加えて PPO diagnostics もかなり健全である。

- `policy_loss=0.0074`
- `ratio_mean=1.0163`
- `clip_fraction=0.2265`
- `anchor_kl_discard=0.0223`

これは、**anchor を少し緩めることが、安定を保ったまま PPO の効きを強める方向として有望**であることを示している。

## 7. 結論

今回の `exp_007` から得られる結論は次の通り。

1. `lr` を上げる方向は、少なくとも今の mixed では得策ではない
2. `1.5e-4` は stable だが S1 を超えない
3. `2.0e-4` はほぼ行き過ぎ
4. **`policy_anchor.coef=0.75` は有望**

したがって、今後の mixed baseline 候補は

- S1: 「最小有効条件」
- R3: 「改善量を少し強めた有望条件」

という 2 層で整理するのがきれいである。

## 8. R3 の follow-up branch-swap

`exp_007` 実行後、R3 run を source にして branch-swap eval を追加で実施した。

結果:

- `II`: `avg_rank=2.4235`, `win_rate=0.2415`, `deal_in=0.1744`
- `FI`: `avg_rank=2.3920`, `win_rate=0.2404`, `deal_in=0.1733`
- `IF`: `avg_rank=2.4165`, `win_rate=0.2423`, `deal_in=0.1756`
- `FF`: `avg_rank=2.3960`, `win_rate=0.2397`, `deal_in=0.1744`

ここからの読みは、

- **discard 側の改善はかなり見える**
- optional 側はまだ効果が弱く、指標が一貫しない
- 相乗効果はまだ強くない

である。

つまり、R3 は少なくとも
**「PPO が何も学んでいない」状態ではなく、discard 側では改善が観測できる段階**
まで来ている。

## 9. 次アクション

次に自然なのは以下のどちらかである。

1. **R3 を mixed の新しい baseline 候補として採用する**
   - optional がまだ弱くても、ルール拡張へ進むには十分前向きな材料がある
2. **optional 側だけもう少し改善余地を探る**
   - ただしこれは「性能の頂点探索」ではなく、「optional が学習できる状態確認」に留める

現時点では、

- Stage02a mixed PPO は stable
- R3 では discard 改善も見えた

ので、**完全麻雀方向へ進む土台としてはかなり良い状態**と評価してよい。
