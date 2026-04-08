# Experiment Report: exp_014

作成日: 2026-04-08  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_014/runbook.md`
- `experiments/Stage02_CallUnlock/exp_013/report.md`
- `experiments/Stage02_CallUnlock/exp_014/semantic_eval_a2_cycle19_confidence/semantic_eval_imitation_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_014/semantic_eval_a2_cycle19_confidence/semantic_eval_cycle18_on_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_014/semantic_eval_a2_cycle19_confidence/semantic_eval_final_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_014/semantic_eval_a2_cycle19_confidence/draw_tenpai_subset_final_cycle19.json`

## 1. 要約

`exp_014` は、`exp_013` の基準条件に対して `CQ-0267` を加えた再実験である。

追加したもの:

- `self_tenpai_flag`
- `remaining_draws_norm`

目的は、5-class terminal のうち特に弱かった `draw_tenpai` を改善できるかを見ることだった。

結論は次の通り。

- **policy performance は改善した**
- 特に `A2_semaux_light_vhalf_tenpaifix` は `exp_013` より final が少し良くなった
- `C0` も大きく改善しており、今回の追加特徴は semantic aux の有無に関係なく有効そう
- semantic では、**`draw_tenpai` は実際に改善した**
- ただしその一方で、**`win_called` / `win_menzen` は大きく悪化した**
- `deal_in` は依然として top-1 ではほぼ学べていない

したがって `exp_014` は、

- `self_tenpai_flag` / `remaining_draws_norm` が有効な特徴であること
- しかし terminal head 全体としては、今の loss / sampling では trade-off が出ること

を確認した実験と整理できる。

## 2. 背景

`exp_013` までで、次の構造整理は完了していた。

- `CQ-0265`
  - `shanten_hint` / `discard_ukeire_hint` を shared trunk から外し、
    discard direct hint branch に戻した
- `CQ-0266`
  - terminal semantic を 5-class に整理した
  - `win_menzen / win_called / draw_tenpai / deal_in / other_non_dealin`

その結果、`A2` では `win_called` がかなりまともに学べるようになった一方、
`draw_tenpai` はまだほぼ学べていなかった。

`draw_tenpai` は、

- 自分がテンパイしているか
- 局終盤か

に強く依存するため、`CQ-0267` ではその 2 点を direct な特徴として入れた。

## 3. 条件

共通固定:

- `value_loss_coef = 0.125`
- latest direct hint branch (`CQ-0265`)
- terminal 5-class (`CQ-0266`)
- seed `42`
- `num_cycles = 20`

比較条件:

### C0: control + vhalf + tenpaifix

- semantic aux 無効

### A2: semantic aux light + vhalf + tenpaifix

- semantic aux 有効
- `policy_projection_dim = 16`
- `terminal_loss_coef = 0.1`
- `yaku_loss_coef = 0.05`

今回は `draw_tenpai` の効果確認を優先し、既定条件は `C0 + A2` の 2 本とした。

## 4. 主結果

### 4.1 imitation と final

#### C0

- imitation: `avg_rank=2.465`, `win_rate=0.2335`, `deal_in_rate=0.1802`
- final: `avg_rank=2.435`, `win_rate=0.2345`, `deal_in_rate=0.1974`

#### A2

- imitation: `avg_rank=2.500`, `win_rate=0.2380`, `deal_in_rate=0.1945`
- final: `avg_rank=2.385`, `win_rate=0.2508`, `deal_in_rate=0.1876`

読み:

- `A2` は final `avg_rank=2.385` で、`exp_013` の `2.410` より改善
- `win_rate` も `0.2432 -> 0.2508` と改善
- `deal_in_rate` も `0.2011 -> 0.1876` と改善
- `C0` も `2.595 -> 2.435` と大きく改善しており、今回の特徴追加はかなり強い

### 4.2 cycle の形

driver log から見ると:

#### C0

- imitation: `avg_rank=2.46`
- best cycle: `cycle 0`, `avg_rank=2.29`
- final: `avg_rank=2.44`
- tail-5 average: `2.474`

#### A2

- imitation: `avg_rank=2.50`
- best cycle: `cycle 18`, `avg_rank=2.37`
- final: `avg_rank=2.39`
- tail-5 average: `2.412`

読み:

- `A2` は peak と final の差が小さく、保持がかなり良い
- `C0` も前回より保持は改善したが、tail-5 では `A2` に劣る
- 現時点の practical baseline は引き続き `A2`

### 4.3 PPO 安定性

#### C0

- `ratio_mean=0.9950`
- `clip_fraction=0.0598`
- `anchor_kl_discard=0.00449`

#### A2

- `ratio_mean=0.9939`
- `clip_fraction=0.0891`
- `anchor_kl_discard=0.00514`

読み:

- どちらも PPO 指標としては健全
- 今回の変化は「不安定化」ではなく、**学習される semantic の重心が変わった**と見るのが自然

## 5. semantic diagnostics (A2)

対象 checkpoint:

- `checkpoint_imitation.pt`
- `checkpoint_cycle_18.pt`
- `checkpoint_learner.pt`

評価 shard:

- `cycle_19/selfplay`

出力先:

- `experiments/Stage02_CallUnlock/exp_014/semantic_eval_a2_cycle19_confidence/`

## 6. terminal の変化

### 6.1 `draw_tenpai` は改善した

#### 全体

- imitation
  - recall `0.0000`
  - `mean_p = 0.0404`
  - `top1_hit = 0.0000`
- cycle 18
  - recall `0.0163`
  - `mean_p = 0.0454`
  - `top1_hit = 0.0163`
- final
  - recall `0.0382`
  - `mean_p = 0.0694`
  - `top1_hit = 0.0382`

`draw_tenpai` はまだ強くはないが、**0 から動いている**。

#### relevant subset: `actual_draw_tenpai_and_self_tenpai_and_late`

ここが今回の本命である。

- support: `582`
- imitation
  - `p(draw_tenpai) = 0.0735`
  - `top1(draw_tenpai) = 0.0000`
- cycle 18
  - `p(draw_tenpai) = 0.1334`
  - `top1(draw_tenpai) = 0.0945`
- final
  - `p(draw_tenpai) = 0.2115`
  - `top1(draw_tenpai) = 0.2079`

比較対象 `actual_other_non_dealin_and_self_tenpai_and_late` の final は:

- `p(draw_tenpai) = 0.1721`
- `top1(draw_tenpai) = 0.1410`

読み:

- `self_tenpai_flag` と `remaining_draws_norm` は、狙いどおり `draw_tenpai` を持ち上げている
- 終盤テンパイ局面では、`draw_tenpai` を relevant class として認識できるようになった
- 差はまだ十分大きいとは言えないが、**以前の 0 状態からは明確に前進**

### 6.2 `win_called` は大きく悪化した

`exp_013 final -> exp_014 final`

- recall: `0.2155 -> 0.0004`
- `mean_p = 0.2445 -> 0.1214`
- `top1_hit = 0.2155 -> 0.0004`

top-3 は `0.9978 -> 0.9781` で依然高いが、
**top-1 で `win_called` を選べる状態からは大きく後退**している。

### 6.3 `win_menzen` も悪化した

`exp_013 final -> exp_014 final`

- recall: `0.0473 -> 0.0025`
- `mean_p = 0.1383 -> 0.0703`

こちらも win 系 terminal としてはかなり弱くなった。

### 6.4 `deal_in` は依然弱い

- imitation: recall `0.0000`, `mean_p = 0.1767`
- cycle 18: recall `0.0000`, `mean_p = 0.2176`
- final: recall `0.0000`, `mean_p = 0.2119`

`deal_in` は probability mass 自体は持つが、top-1 class としては出ていない。

## 7. yaku の変化

ここは意外だったが、`exp_014` では yaku がかなり改善した。

`exp_013 final -> exp_014 final`

- micro F1: `0.3030 -> 0.4716`
- macro F1: `0.0641 -> 0.0840`
- exact match: `0.1566 -> 0.3256`

terminal の `draw_tenpai` 改善と同時に、yaku も持ち直している。

ただし terminal win 系が崩れているので、
**semantic 全体が一様に良くなったわけではなく、重心が `draw_tenpai` / `other_non_dealin` 寄りに移った**と見る方が正確である。

## 8. 解釈

今回の `CQ-0267` は、仮説どおり

- `draw_tenpai`
- 終盤テンパイ局面

には効いた。

一方で shared trunk / semantic loss 全体としては、

- `draw_tenpai`
- `other_non_dealin`

の識別を強める方向に寄り、

- `win_called`
- `win_menzen`

をかなり犠牲にした。

つまり、今回確認できたのは次の 2 点である。

1. `self_tenpai_flag` / `remaining_draws_norm` は有効な特徴である
2. ただし今の loss / sampling のままだと、terminal 5-class の中でトレードオフが起きる

## 9. 結論

`exp_014` の結論は次の通り。

- **policy 観点では前進**
  - `A2` は final `avg_rank=2.385` まで改善
  - `C0` も大きく改善
- **semantic 観点では半分成功**
  - `draw_tenpai` は改善した
  - しかし `win_called / win_menzen` は大きく悪化
- `deal_in` は引き続き弱い

したがって次にやるべきことは、特徴量追加ではなく、

- terminal 5-class の class weight
  または
- sampling 調整

によって、

- `draw_tenpai` の改善を保ちつつ
- `win_called` を戻す

ことである。

現時点の practical baseline は依然として `A2` だが、
semantic terminal のバランスを取るには、**次は学習設計側に踏み込む段階**と判断する。
