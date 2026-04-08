# Experiment Report: exp_012

作成日: 2026-04-02  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_012/runbook.md`
- `experiments/Stage02_CallUnlock/exp_011/report.md`
- `experiments/Stage02_CallUnlock/exp_012/semantic_eval_valuecoef/base_cycle05/semantic_eval_base_cycle05_summary.md`
- `experiments/Stage02_CallUnlock/exp_012/semantic_eval_valuecoef/vhalf_cycle05/semantic_eval_vhalf_cycle05_summary.md`
- `experiments/Stage02_CallUnlock/exp_012/semantic_eval_valuecoef/vlow_cycle05/semantic_eval_vlow_cycle05_summary.md`
- `experiments/Stage02_CallUnlock/exp_012/semantic_eval_valuecoef/base_final/semantic_eval_base_final_summary.md`
- `experiments/Stage02_CallUnlock/exp_012/semantic_eval_valuecoef/vhalf_final/semantic_eval_vhalf_final_summary.md`
- `experiments/Stage02_CallUnlock/exp_012/semantic_eval_valuecoef/vlow_final/semantic_eval_vlow_final_summary.md`

## 1. 要約

`exp_012` は、`exp_011` で本線となった `A1_semaux_default_featurefix` を固定し、
PPO の `training.value_loss_coef` だけを

- `0.25` (`A1_base_v025`)
- `0.125` (`A1_vhalf_v0125`)
- `0.05` (`A1_vlow_v005`)

に振った sweep である。

結論は次の通り。

- **最有力は `A1_vhalf_v0125`**
- `value_loss_coef` を少し下げると、policy performance は改善する
- ただし、**semantic head collapse を根本的に止める効果は確認できなかった**
- 特に `checkpoint_cycle_05` の semantic diagnostics では、`vhalf` / `vlow` は `base` より `win_called` を強く持てていない
- 一方で final checkpoint では、`vhalf` / `vlow` の方が `win_called mean_p` と `top3_hit_rate` はやや改善する
- したがって、`value_loss_coef` の調整は有効ではあるが、**semantic collapse の主因はこれだけではない**

つまり今回の sweep は、

- `A1` の新しい実験基準として `vhalf` を採用する価値がある
- ただし次の本命は `semantic anchor` や `class imbalance` 対策など、別の保持策である

と整理するのが自然である。

## 2. 背景

`exp_011` では、feature 修正

- actor-relative full observation
- full path `riichi`
- full path `menzen`

の後で、`A1` が再び有望になった。

一方で semantic diagnostics では、

- imitation checkpoint では `win_called` confidence が改善
- しかし `checkpoint_cycle_05` と final では terminal head が `ron_bystander` に collapse

という問題が残っていた。

この段階での仮説は、

- PPO 中に shared `value/semantic` 表現が value 側に強く引っ張られ、semantic head が保てないのではないか

というものだった。

そのため今回は、構造変更や loss 変更を入れず、まず `value_loss_coef` のみを触って切り分けた。

## 3. 条件

固定:

- `A1_semaux_default_featurefix` と同じ条件
- semantic aux 有効
- `policy_projection_dim = 16`
- `terminal_loss_coef = 0.2`
- `yaku_loss_coef = 0.1`
- feature 条件は `exp_011` と同じ
- seed `42`
- `num_cycles = 20`

差分:

### A1_base_v025

- `training.value_loss_coef = 0.25`

### A1_vhalf_v0125

- `training.value_loss_coef = 0.125`

### A1_vlow_v005

- `training.value_loss_coef = 0.05`

## 4. 主結果

### 4.1 imitation と final

全条件で imitation は同じ。

- imitation: `avg_rank=2.500`, `win_rate=0.2333`, `deal_in_rate=0.1961`

final:

- `A1_base_v025`: `avg_rank=2.415`, `win_rate=0.2396`, `deal_in_rate=0.1921`
- `A1_vhalf_v0125`: **`avg_rank=2.380`**, `win_rate=0.2398`, `deal_in_rate=0.1817`
- `A1_vlow_v005`: `avg_rank=2.470`, `win_rate=0.2443`, `deal_in_rate=0.2041`

読み:

- `vhalf` が final `avg_rank` 最良
- `vlow` は `win_rate` は高いが、`avg_rank` と `deal_in` が悪い
- `value_loss_coef` を下げすぎると、policy quality は少し崩れる

### 4.2 best cycle

- `A1_base_v025`: cycle 0, `avg_rank=2.385`, `win_rate=0.2465`, `deal_in=0.1896`
- `A1_vhalf_v0125`: cycle 17, `avg_rank=2.365`, `win_rate=0.2701`, `deal_in=0.1905`
- `A1_vlow_v005`: cycle 16, `avg_rank=2.395`, `win_rate=0.2446`, `deal_in=0.1807`

読み:

- `vhalf` は best cycle でも最良
- `vlow` は final より best cycle でも弱い

### 4.3 tail-5 average

- `A1_base_v025`: `avg_rank=2.442`, `win_rate=0.2465`, `deal_in=0.1909`
- `A1_vhalf_v0125`: `avg_rank=2.457`, `win_rate=0.2454`, `deal_in=0.1907`
- `A1_vlow_v005`: `avg_rank=2.464`, `win_rate=0.2421`, `deal_in=0.1923`

読み:

- tail-5 は `base` がわずかに良い
- `vhalf` は final は最良だが、終盤平均で base を明確には上回っていない
- それでも `vlow` よりは安定している

### 4.4 PPO 安定性

- `A1_base_v025`
  - `ratio_mean=1.0175`
  - `clip_fraction=0.2443`
  - `anchor_kl_discard=0.0192`
- `A1_vhalf_v0125`
  - `ratio_mean=1.0097`
  - `clip_fraction=0.2258`
  - `anchor_kl_discard=0.0191`
- `A1_vlow_v005`
  - `ratio_mean=1.0062`
  - `clip_fraction=0.2252`
  - `anchor_kl_discard=0.0178`

読み:

- `value_loss_coef` を下げても PPO は不安定化していない
- むしろ `vhalf/vlow` は PPO 指標だけ見ると素直
- したがって `vhalf` の改善は偶然ではなく、少なくとも PPO 挙動としては自然

## 5. semantic diagnostics

今回の主眼は、`value_loss_coef` 調整で semantic head collapse が緩和するかである。
そのため `cycle_05` と final を比較した。

### 5.1 cycle_05

`win_called` confidence:

- `base`
  - `mean_p = 0.0958`
  - `top3_hit_rate = 0.0203`
  - `mean_rank = 4.0`
- `vhalf`
  - `mean_p = 0.0729`
  - `top3_hit_rate = 0.0087`
  - `mean_rank = 4.5`
- `vlow`
  - `mean_p = 0.0798`
  - `top3_hit_rate = 0.0062`
  - `mean_rank = 4.1`

terminal accuracy:

- `base = 0.4371`
- `vhalf = 0.4442`
- `vlow = 0.4512`

ただし中身は共通して

- `win_called recall = 0`
- `ron_bystander recall ≈ 0.996 - 0.997`

であり、collapse は維持されている。

読み:

- **cycle_05 の時点では、`base` が一番 `win_called` を見ている**
- `value_loss_coef` を下げるだけでは、early collapse は改善していない

### 5.2 final

`win_called` confidence:

- `base`
  - `mean_p = 0.1004`
  - `top3_hit_rate = 0.0066`
  - `mean_rank = 4.0`
- `vhalf`
  - **`mean_p = 0.1219`**
  - **`top3_hit_rate = 0.0177`**
  - `mean_rank = 4.0`
- `vlow`
  - `mean_p = 0.1193`
  - `top3_hit_rate = 0.0128`
  - `mean_rank = 4.0`

こちらも

- `win_called recall = 0`
- `ron_bystander recall ≈ 0.993 - 0.999`

で、top-1 collapse 自体は解消していない。

読み:

- final では `vhalf/vlow` の方が `win_called` を少し見ている
- ただし改善は限定的で、top-1 に上がるところまでは全く届いていない

### 5.3 yaku の傾向

cycle_05:

- `base`
  - micro F1 `0.3341`
  - `Riichi hit@0.5 = 0.7425`
  - `Yakuhai hit@0.5 = 0.1252`
- `vhalf`
  - micro F1 `0.3112`
  - `Riichi hit@0.5 = 0.6376`
  - `Yakuhai hit@0.5 = 0.1800`
- `vlow`
  - micro F1 `0.3236`
  - `Riichi hit@0.5 = 0.6472`
  - `Yakuhai hit@0.5 = 0.1595`

final:

- `base`
  - micro F1 `0.3055`
  - `Riichi hit@0.5 = 0.6231`
  - `Yakuhai hit@0.5 = 0.1063`
- `vhalf`
  - micro F1 `0.2262`
  - `Riichi hit@0.5 = 0.1913`
  - `Yakuhai hit@0.5 = 0.2619`
- `vlow`
  - micro F1 `0.3326`
  - `Riichi hit@0.5 = 0.3514`
  - `Yakuhai hit@0.5 = 0.3569`

読み:

- `vhalf` は terminal final が良い一方、yaku はかなり崩れている
- `vlow` は yaku の保持では一番良いが、policy performance が落ちる
- ここにも `value / semantic` 以外の tradeoff が見えている

### 5.4 条件付き分布の追加確認

`win_called mean_p` の読みを補強するため、`A1_vhalf_v0125` final checkpoint について

- `actual_win_called_and_tenpai_and_open`
- `actual_ron_bystander` 全体
- `all_samples` 全体

の terminal 予測平均を比較した。

参照:

- `experiments/Stage02_CallUnlock/exp_012/win_called_tenpai_distribution_vhalf_final.json`
- `experiments/Stage02_CallUnlock/exp_012/win_called_vs_overall_distribution_vhalf_final.json`

`actual_win_called_and_tenpai_and_open`:

- count `2854`
- `win_menzen = 0.0721`
- `win_called = 0.1192`
- `ron_loss = 0.1779`
- `tsumo_loss = 0.2132`
- `ron_bystander = 0.3282`

`actual_ron_bystander` 全体:

- `win_menzen = 0.0446`
- `win_called = 0.1163`
- `ron_loss = 0.1888`
- `tsumo_loss = 0.2273`
- `ron_bystander = 0.3728`

`all_samples` 全体:

- `win_menzen = 0.0483`
- `win_called = 0.1164`
- `ron_loss = 0.1878`
- `tsumo_loss = 0.2259`
- `ron_bystander = 0.3695`

読み:

- `win_called` は、**actual `win_called` で終わった open-tenpai 局面でも `0.1192`**
- 一方で
  - `actual_ron_bystander` 全体では `0.1163`
  - `all_samples` 全体でも `0.1164`
- つまり、**relevant な `win_called` 局面でも `p(win_called)` がほとんど上がっていない**

さらに、`actual_win_called_and_tenpai_and_open` での top-1 は

- `ron_bystander = 0.9832`
- `win_menzen = 0.0168`
- `win_called = 0.0`

であり、top-1 collapse も維持されている。

この比較から、今の terminal head は

- open / tenpai らしい雰囲気を少しは見ている
- しかし **`win_called` になりやすい局面を特異的には識別できていない**

と読むのが自然である。

## 6. 解釈

### 6.1 `value_loss_coef` を少し下げるのは有効

`0.25 -> 0.125` は、少なくとも policy 側には効いている。

- final `avg_rank` は最良
- deal-in も最良
- PPO 安定性も良い

したがって、**実験の新基準としては `A1_vhalf_v0125` を採用してよい**。

### 6.2 ただし semantic collapse の主因ではない

今回もっとも重要なのは、`cycle_05` diagnostics で

- `base` が最も `win_called` を持てていた
- `vhalf/vlow` はそこを改善していない

ことである。

つまり、`value_loss_coef` を下げても

- early semantic collapse は止まらない
- `ron_bystander` top-1 collapse も止まらない
- relevant な `win_called` 局面で `p(win_called)` を押し上げるところまでも届かない

したがって、**「value が強すぎるから semantic が壊れる」だけが主因ではない**。

### 6.3 次に見るべきもの

今回の結果を踏まえると、次の本命は

- semantic anchor replay
- terminal / yaku の class imbalance 対策

のどちらか、あるいは両方である。

特に `win_called` は

- support は十分ある
- feature 修正後も imitation では改善する
- でも PPO で保持できない

ので、**保持機構そのもの** を入れる必要がある可能性が高い。

## 7. まとめ

`exp_012` の時点で言えることは次の通り。

1. `value_loss_coef` sweep はやる価値があった
2. 新しい実験基準としては **`A1_vhalf_v0125`** が最有力
3. ただし semantic collapse を止める主手段にはならなかった
4. 次の焦点は、`semantic anchor` または `class imbalance` 対策である

したがって、今後は

- 基準条件を `A1_vhalf_v0125` に更新しつつ
- semantic head を PPO 中に保つ仕組みを試す

のが自然である。
