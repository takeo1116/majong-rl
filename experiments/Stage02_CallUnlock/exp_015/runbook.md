# Experiment Runbook: exp_015

作成日: 2026-04-08  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_013/report.md`
- `experiments/Stage02_CallUnlock/exp_014/report.md`
- `experiments/Stage02_CallUnlock/exp_014/runbook.md`
- `reference/stage2/stage2a_semantic_aux_trunk_design.md`

## 1. 背景

`exp_014` では、次の構造修正を反映した状態で `C0 / A2` を再比較した。

- `CQ-0265`: `shanten_hint` / `discard_ukeire_hint` を discard direct hint に戻す
- `CQ-0266`: terminal semantic を 5-class に整理する
- `CQ-0267`: `self_tenpai_flag` / `remaining_draws_norm` を追加する

その結果、

- policy 性能は `C0` / `A2` ともに改善した
- 特に `A2` は実用基準として引き続き最良だった
- semantic terminal では `draw_tenpai` が改善した
- 一方で `win_called` / `win_menzen` は大きく悪化した

という、かなり明確な trade-off が出た。

この状況から、次の仮説が立った。

- 今の terminal teacher は、同じ `player-round` の全 decision に同じ終局ラベルを貼っている
- そのため長い局ほど terminal loss の総量が大きくなり、row ベースの empirical 分布が歪んでいる
- これが terminal class の不安定さ、特に `draw_tenpai` と win 系の綱引きを悪化させている可能性が高い

これを受けて `CQ-0268` を実装した。

## 2. 今回の再実験前に入った修正

### 2.1 terminal semantic loss の player-round 正規化 (`CQ-0268`)

Stage2a の terminal semantic loss に対して、`player-round` 単位の正規化重みを導入した。

- group key: `episode_id / round_id / player_id`
- 同じ group に属する decision 群の terminal weight 合計を `1.0` にそろえる
- decision 数 `n` の group では各 row 重みは `1/n`

重要:
- 今回は **正規化のみ**
- 巡目による late weighting は入れていない
- `yaku` / `policy` / `value` / reward / GAE は据え置き

狙い:
- 長い局ほど terminal label が強く入りすぎる bias を抑える
- `draw_tenpai` を保ちながら `win_called` / `win_menzen` が戻るかを見る

## 3. 今回の問い

今回の `exp_015` で答えたい問いは次の 4 つである。

1. `CQ-0268` により `draw_tenpai` の改善を保ったまま `win_called` / `win_menzen` が回復するか
2. `A2_semaux_light` は terminal 正規化後も引き続き最良条件か
3. `A1_semaux_default` の retain 問題が terminal 正規化で改善するか
4. terminal loss の duplicated-label bias 補正だけで semantic terminal がどこまで安定するか

## 4. 実験方針

今回は `exp_014` の現基準をそのまま維持し、差分を `CQ-0268` の terminal player-round 正規化だけに絞る。

固定するもの:

- `training.value_loss_coef = 0.125`
- latest direct hint branch (`CQ-0265`)
- terminal 5-class (`CQ-0266`)
- `self_tenpai_flag` / `remaining_draws_norm` (`CQ-0267`)
- seed `42`

主比較は **`A2` と `A1` の 2 条件**とする。

理由:

- `CQ-0268` は terminal semantic loss のみに効く
- `semantic_aux` 無効の `C0` には本質的に効かない
- 今回は semantic terminal の stabilizing 効果を見たいので、`A2 / A1` に絞る

## 5. 比較条件

### A2: semantic aux light + vhalf + tenpaifix + prnorm

- semantic aux 有効
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.terminal_loss_coef = 0.1`
- `training.semantic_aux.yaku_loss_coef = 0.05`
- `training.value_loss_coef = 0.125`
- terminal semantic loss は player-round 正規化あり

### A1: semantic aux default + vhalf + tenpaifix + prnorm

- semantic aux 有効
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.terminal_loss_coef = 0.2`
- `training.semantic_aux.yaku_loss_coef = 0.1`
- `training.value_loss_coef = 0.125`
- terminal semantic loss は player-round 正規化あり

## 6. 共通固定条件

全条件共通:

- `core_minimal`
- `full observation`
- mixed PPO
- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix_learner.baseline_sample_weight = 0.25`
- `training.policy_anchor.reference = "imitation_fixed"`
- `training.policy_anchor.coef = 0.75`
- `training.value_loss_coef = 0.125`
- `training.lr = 1e-4`
- `training.clip_epsilon = 0.15`
- `training.max_grad_norm = 0.50`
- `training.multi_cycle.num_cycles = 20`
- `training.multi_cycle.eval_each_cycle = true`
- `training.imitation_eval.enabled = true`
- seed `42`

encoder / model 側では、次の構造を前提にする。

- actor-relative full observation
- full path `riichi`
- full path `menzen`
- Stage2a discard direct hints (`shanten_hint`, `discard_ukeire_hint`)
- terminal semantic 5-class
- `self_tenpai_flag`
- `remaining_draws_norm`
- terminal semantic loss player-round 正規化

## 7. 必須観測

### 7.1 通常性能

- `imitation_eval`
- `final`
- `best cycle`
- `tail-5 average`

特に見る差分:

- `final - imitation_eval`
- `A2 - A1`
- `exp_014 -> exp_015`

### 7.2 PPO 安定性

- `ratio_mean`
- `clip_fraction`
- `anchor_kl_discard`
- learner loss の暴れ方
- 20 cycle 完走可否

### 7.3 semantic diagnostics

少なくとも `A2` について、以下を取る。

- `checkpoint_imitation.pt`
- `best cycle checkpoint`
- `checkpoint_learner.pt`

余力があれば `A1` でも同じものを取る。

見るもの:

- terminal accuracy
- `win_called` / `win_menzen` / `draw_tenpai` / `deal_in` の support と recall
- label-conditioned confidence
- relevant subset の平均予測分布

### 7.4 重点 subset 診断

今回の本命はここである。

少なくとも次を見たい。

- `actual_draw_tenpai_and_self_tenpai_and_late`
- `actual_win_called_and_tenpai_and_open`
- `actual_other_non_dealin_and_self_tenpai_and_late`

比較観点:

- `draw_tenpai` が `exp_014` 並みに保たれるか
- `win_called` が `exp_014` より回復するか
- `win_menzen` が `exp_014` より回復するか

## 8. 成功条件

今回の成功条件は、次のどれかを満たすこととする。

1. `A2` で `draw_tenpai` の改善を維持したまま `win_called` が回復する
2. `A2` で `win_menzen` も回復する
3. `A1` の final retain が改善する
4. policy 性能が `exp_014` 並み以上を維持する

## 9. 失敗条件と次の手

もし `CQ-0268` 後も

- `draw_tenpai` は上がるが win 系が戻らない
- または terminal 全体がまだ unstable

なら、次は構造ではなく **loss 設計**を疑う。

候補:

1. terminal class の mild class weight
2. progress-based terminal weighting
3. semantic 用 sampling / replay 設計

今回はそこまでは入れず、まず `player-round` 正規化単独の効果だけを見る。

## 10. 実行コマンド

既定の 2 条件を回す:

```bash
./.venv/bin/python scripts/local/stage2/exp_015_driver.py
```

1 条件だけ回す:

```bash
EXP015_ONLY=A2_semaux_light_vhalf_tenpaifix_prnorm ./.venv/bin/python scripts/local/stage2/exp_015_driver.py
```
