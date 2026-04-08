# Experiment Runbook: exp_016

作成日: 2026-04-09  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_014/report.md`
- `experiments/Stage02_CallUnlock/exp_015/report.md`
- `experiments/Stage02_CallUnlock/exp_015/runbook.md`
- `reference/stage2/stage2a_semantic_aux_trunk_design.md`

## 1. 背景

`exp_015` では、次の修正を反映した状態で `A2 / A1` を比較した。

- `CQ-0268`: terminal semantic loss の player-round 正規化

その結果、

- terminal 学習の安定性は明確に改善した
- `A2_semaux_light_vhalf_tenpaifix_prnorm` が practical baseline になった
- `deal_in` は top-1 class ではなく risk signal として見る方が自然だと整理された
- yaku は `Riichi` / `Yakuhai` は比較的見えている一方、
  `Tanyao` / `Pinfu` / `MenzenTsumo` などは late な局面でもほぼ立っていない

という状況になった。

その後、次の 3 点を実装した。

- `CQ-0269`: `deal_in` risk diagnostics を `semantic_eval` に追加
- `CQ-0270`: self tile-presence flags を encoder に追加
- `CQ-0271`: Stage2a discard の legal snapshot / action 解決を同一化

このうち、学習挙動そのものに効く本命は `CQ-0270` である。

## 2. 今回の再実験前に入った修正

### 2.1 self tile-presence flags の追加 (`CQ-0270`)

共有 encoder feature に次の 6 特徴を追加した。

- `self_has_honor`
- `self_has_terminal`
- `self_has_simple`
- `self_has_man`
- `self_has_pin`
- `self_has_sou`

意味:
- 手牌または副露に、対応する牌種が 1 枚以上あれば `1.0`

狙い:
- `Tanyao` のような「字牌 / 么九牌が存在しないこと」が条件になる役を、
  34次元 hand counts だけよりも MLP が直接読みやすくする
- `Yakuhai` のような存在条件との非対称性を少し緩和する

### 2.2 `deal_in` risk diagnostics の追加 (`CQ-0269`)

これは学習条件ではなく評価改善である。

`semantic_eval` に次を追加した。

- `mean_p(deal_in | y=deal_in)`
- `mean_p(deal_in | y!=deal_in)`
- `roc_auc`
- `pr_auc`
- subset:
  - `late_and_noten`
  - `early_and_tenpai`

狙い:
- `deal_in` を top-1 class ではなく危険度確率として評価する

### 2.3 illegal discard バグ修正 (`CQ-0271`)

discard decision で

- legal mask 作成
- tile_type 選択
- concrete `Action` 解決

が同一 legal snapshot 上で完結するようにした。

これは主に run の信頼性向上であり、学習方針の差分ではない。

## 3. 今回の問い

`exp_016` で答えたい問いは次の 4 つである。

1. `CQ-0270` により yaku、特に `Tanyao` が少しでも立つようになるか
2. yaku 改善が terminal / policy を壊さずに得られるか
3. `A2_semaux_light_vhalf_tenpaifix_prnorm` は `yakuflags` 後も実用基準を維持できるか
4. 追加特徴の効果は `semantic_aux` ありで主に出るのか、それとも control にも出るのか

## 4. 実験方針

今回は `exp_015` の現基準をそのまま維持し、差分を `CQ-0270` の presence flags に絞る。

固定するもの:

- `training.value_loss_coef = 0.125`
- latest direct hint branch (`CQ-0265`)
- terminal 5-class (`CQ-0266`)
- `self_tenpai_flag` / `remaining_draws_norm` (`CQ-0267`)
- terminal player-round normalization (`CQ-0268`)
- seed `42`

主比較は **`C0` と `A2` の 2 条件**とする。

理由:

- `A2` は現在の practical baseline
- `CQ-0270` は shared feature 追加なので、policy / terminal / yaku の全部に影響しうる
- `C0` を入れることで、
  - 単純な特徴量改善
  - semantic aux と組み合わさった改善
  を切り分けやすい

時間があれば `A2` の seed 追加で再現性を見る。

## 5. 比較条件

### C0: control + vhalf + tenpaifix + prnorm + yakuflags

- semantic aux 無効
- `training.value_loss_coef = 0.125`
- 最新の
  - actor-relative full observation
  - full path `riichi` / `menzen`
  - Stage2a discard direct hints
  - terminal 5-class
  - `self_tenpai_flag`
  - `remaining_draws_norm`
  - terminal player-round normalization
  - self tile-presence flags
  を使う control

### A2: semantic aux light + vhalf + tenpaifix + prnorm + yakuflags

- semantic aux 有効
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.terminal_loss_coef = 0.1`
- `training.semantic_aux.yaku_loss_coef = 0.05`
- `training.value_loss_coef = 0.125`
- 上記 C0 と同じ最新特徴量

### A2 seed43（任意）

- `A2` と同一条件
- seed のみ `43`

目的:
- `CQ-0270` の効果が seed42 の偶然でないかを軽く確認する

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

encoder / model 側では、次の構造を前提にする。

- actor-relative full observation
- full path `riichi`
- full path `menzen`
- Stage2a discard direct hints (`shanten_hint`, `discard_ukeire_hint`)
- terminal semantic 5-class
- `self_tenpai_flag`
- `remaining_draws_norm`
- terminal player-round 正規化
- self tile-presence flags

## 7. 必須観測

### 7.1 通常性能

- `imitation_eval`
- `final`
- `best cycle`
- `tail-5 average`

特に見る差分:

- `final - imitation_eval`
- `A2 - C0`
- `exp_015 -> exp_016`

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

見るもの:

- terminal accuracy
- `win_called` / `win_menzen` / `draw_tenpai` の support と recall
- `deal_in` risk diagnostics
- yaku micro / macro F1
- yaku の label-conditioned confidence

### 7.4 yaku 重点観測

今回の本命はここである。

最低限、次を見たい。

- `Riichi`
- `Yakuhai`
- `Tanyao`

余力があれば:

- `Pinfu`
- `MenzenTsumo`

加えて、`winner player-round の最後の 3 decision` に限定した yaku 集計も再度取る。

見たいもの:

- overall winner-only と last-3 only の差
- `Tanyao` が 0 から少しでも動くか
- `Riichi` / `Yakuhai` を壊していないか

### 7.5 `deal_in` risk diagnostics

今回から top-1 ではなく、次を主に見る。

- `mean_p_pos / mean_p_neg`
- `roc_auc / pr_auc`
- `late_and_noten`
- `early_and_tenpai`

## 8. 成功判定

今回の成功条件は、次のどれかを満たすこととする。

1. `A2` の policy 性能が `exp_015` と同等以上
2. yaku macro F1 が改善する
3. `Tanyao` の recall または confidence が 0 から動く
4. terminal が大きく崩れない
5. `deal_in` risk diagnostics の分離が改善する

## 9. 失敗条件と次の手

もし `CQ-0270` 後も

- `Tanyao` が全く立たない
- yaku macro F1 が動かない
- terminal / policy まで悪化する

なら、次は特徴量追加ではなく **yaku loss 側**を疑う。

候補:

1. yaku の mild label-wise weight
2. rare すぎない役だけ positive weight を入れる
3. coarse yaku 補助 head

今回はそこまでは入れず、まず presence flags 単独の効果だけを見る。

## 10. 実行コマンド

既定の 2 条件を回す:

```bash
./.venv/bin/python scripts/local/stage2/exp_016_driver.py
```

`A2 seed43` も追加する:

```bash
EXP016_INCLUDE_SEED43=1 ./.venv/bin/python scripts/local/stage2/exp_016_driver.py
```

1 条件だけ回す:

```bash
EXP016_ONLY=A2_semaux_light_vhalf_tenpaifix_prnorm_yakuflags ./.venv/bin/python scripts/local/stage2/exp_016_driver.py
```
