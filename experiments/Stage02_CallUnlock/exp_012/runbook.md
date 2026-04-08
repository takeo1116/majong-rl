# Experiment Runbook: exp_012

作成日: 2026-04-02  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_011/report.md`
- `experiments/Stage02_CallUnlock/exp_011/runbook.md`
- `reference/stage2/stage2a_semantic_aux_trunk_design.md`

## 1. 背景

`exp_011` では、feature 修正後の条件で semantic auxiliary を再評価した。
その結果、次のことが分かった。

- actor-relative + `riichi/menzen` 追加により、`A1/A2` は `C0` より明確に改善した
- 特に `A1_semaux_default_featurefix` は final `avg_rank` が最良で、現時点の本線になった
- 一方で semantic diagnostics を見ると、`checkpoint_imitation.pt` では `win_called` confidence が改善しているのに、`checkpoint_cycle_05.pt` と final では terminal head が再び `ron_bystander` 偏重に collapse していた
- yaku head も imitation から PPO を経るにつれて全体として悪化していた

ここから、現状の主問題は

- feature 欠損ではなく
- **PPO 中に value / semantic 共有表現がどう引っ張られるか**

にあると考えられる。

最初の切り分けとして、今回は trunk 構造や loss 設計を変えず、**`training.value_loss_coef` のみを sweep** して PPO 中の semantic collapse が緩和するかを見る。

## 2. 今回の問い

今回の runbook で答えたい問いは次の 4 つである。

1. `value_loss_coef` を下げると、`A1` の semantic head collapse は緩和するか
2. 特に `win_called` の confidence (`mean_p`, `top3_hit_rate`, `mean_rank`) は改善するか
3. semantic head の保持が改善しても、policy performance は維持できるか
4. `value_loss_coef` を下げすぎると PPO 安定性や final performance は崩れるか

## 3. 実験方針

今回は **`A1_semaux_default_featurefix` を固定し、`training.value_loss_coef` だけを変える**。

固定するもの:

- semantic aux 有効
- `policy_projection_dim = 16`
- `terminal_loss_coef = 0.2`
- `yaku_loss_coef = 0.1`
- feature 条件は `exp_011` と同じ
  - actor-relative full observation
  - full path `riichi`
  - full path `menzen`
- seed `42`
- `num_cycles = 20`

動かすもの:

- `training.value_loss_coef`

## 4. 比較条件

### A1_base_v025

- `training.value_loss_coef = 0.25`
- `exp_011` の `A1_semaux_default_featurefix` と同一
- 比較基準

### A1_vhalf_v0125

- `training.value_loss_coef = 0.125`
- PPO value loss の寄与を半減

### A1_vlow_v005

- `training.value_loss_coef = 0.05`
- PPO value loss の寄与をかなり弱める

## 5. 共通固定条件

全条件共通:

- `core_minimal`
- `full observation`
- mixed PPO
- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix_learner.baseline_sample_weight = 0.25`
- `training.policy_anchor.reference = "imitation_fixed"`
- `training.policy_anchor.coef = 0.75`
- `training.lr = 1e-4`
- `training.clip_epsilon = 0.15`
- `training.max_grad_norm = 0.50`
- `training.multi_cycle.num_cycles = 20`
- `training.multi_cycle.eval_each_cycle = true`
- `training.imitation_eval.enabled = true`
- seed `42`

encoder 側で有効のままにするもの:

- `shanten_hint`
- `discard_ukeire_hint`
- `current_shanten`
- `shape_hint`
- `turn_context`
- full path actor-relative `riichi`
- full path actor-relative `menzen`

## 6. 観測項目

### 6.1 性能

- `imitation_eval`
- `final`
- `best cycle`
- `tail-5 average`

特に見る差分:

- `final - imitation_eval`
- `tail-5`
- 3 条件間の final `avg_rank`

### 6.2 PPO 安定性

- `ratio_mean`
- `clip_fraction`
- `anchor_kl_discard`
- learner loss の暴れ方
- 20 cycle 完走可否

### 6.3 semantic diagnostics

少なくとも `A1_base_v025` と、最も良かった条件について以下を取る。
必要なら 3 条件すべて取る。

評価点:

- `checkpoint_imitation.pt`
- `checkpoint_cycle_05.pt`
- `checkpoint_learner.pt`

特に見たいもの:

- terminal accuracy
- `win_called` support / recall
- `win_called` の `mean_p`, `p50`, `p90`, `top1_hit_rate`, `top3_hit_rate`, `mean_rank`
- `top1_confusers`
- yaku micro / macro F1
- `Riichi`, `Yakuhai`, `Tanyao` の positive-conditioned confidence

## 7. 成功判定

今回の sweep は、少なくとも次を満たせば成功とみなす。

1. `value_loss_coef` を下げた条件で semantic collapse が緩和する
   - 例: `cycle_05` や final で `win_called mean_p` / `top3_hit_rate` が base より改善
2. PPO 安定性が大きく悪化しない
3. final performance が base と同程度以上で維持される、または改善する

特に今回は、**policy performance を極端に落とさずに semantic head を保てるか** が主眼である。

## 8. 典型的な解釈

### ケース 1: `value_loss_coef` を下げると semantic も policy も改善

解釈:

- PPO 中は value loss の寄与が強すぎ、semantic を潰していた
- 次はこの方向で finer sweep か multi-seed に進める

### ケース 2: semantic は改善するが policy が落ちる

解釈:

- critic が弱くなりすぎている
- value と semantic の tradeoff がある
- 次は PPO 専用 semantic coef や anchor replay を検討する

### ケース 3: ほとんど変わらない

解釈:

- 問題は単純な `value/semantic` 比率ではない
- class imbalance や semantic anchor の方が本命

## 9. 実行コマンド

全条件:

```bash
./.venv/bin/python scripts/local/stage2/exp_012_driver.py
```

1 条件だけ:

```bash
EXP012_ONLY=A1_vhalf_v0125 ./.venv/bin/python scripts/local/stage2/exp_012_driver.py
```
