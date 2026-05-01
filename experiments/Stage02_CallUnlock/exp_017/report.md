# Experiment Report: exp_017

作成日: 2026-04-10  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_017/runbook.md`
- `experiments/Stage02_CallUnlock/exp_015/report.md`
- `experiments/Stage02_CallUnlock/exp_016/report.md`
- `experiments/Stage02_CallUnlock/exp_017/semantic_eval_off_narrow_cycle19/semantic_eval_final_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_017/semantic_eval_off_wide_cycle19/semantic_eval_final_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_017/semantic_eval_on_narrow_cycle19/semantic_eval_cycle8_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_017/semantic_eval_on_narrow_cycle19/semantic_eval_final_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_017/semantic_eval_on_wide_cycle19/semantic_eval_cycle10_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_017/semantic_eval_on_wide_cycle19/semantic_eval_final_cycle19_summary.md`

## 1. 要約

`exp_017` は、`CQ-0270` の self tile-presence flags (`yakuflags`) が
`exp_016` で悪化した理由を、

- `tile_presence_flags`: `off / on`
- `value_hidden_dims`: `narrow / wide`

の 2x2 factorial で切り分けた実験である。

結論は次の通り。

- **capacity 仮説はかなり支持された**
- `yakuflags` は `narrow` だと悪いが、`wide` にすると policy も diagnostics もかなり回復する
- 特に `Tanyao` は recall こそ立たないままでも、
  **`on_wide` で確率信号 (`mean_p`, `hit@0.2`) が明確に改善**した
- 一方で、`on_wide` でも practical baseline (`exp_015 A2`) を超えたとは言えない
- したがって、**baseline は引き続き `exp_015 A2` 維持**
- ただし `CQ-0270` の発想自体は強く再評価され、
  「shared input に常時入れるだけでは不十分だが、容量があれば活きる」
  という整理になった

## 2. 背景

`exp_015` までで、

- direct hint branch (`CQ-0265`)
- terminal 5-class (`CQ-0266`)
- `self_tenpai_flag / remaining_draws_norm` (`CQ-0267`)
- terminal player-round normalization (`CQ-0268`)

が入り、

- `A2_semaux_light_vhalf_tenpaifix_prnorm`

が practical baseline になっていた。

その後 `exp_016` では、`CQ-0270` により

- `self_has_honor`
- `self_has_terminal`
- `self_has_simple`
- `self_has_man`
- `self_has_pin`
- `self_has_sou`

を shared encoder input に追加したが、

- policy は悪化
- 本命の `Tanyao` は改善しない

という結果になった。

ただし、この結果だけで特徴量アイデア自体を棄却するのは早いと判断し、
今回は

- `yakuflags` の有無
- `value / terminal / yaku` 側 trunk 幅

の相互作用を切り分けることにした。

## 3. 条件

すべて `A2_semaux_light_vhalf_tenpaifix_prnorm` 系で固定し、
次の 4 条件を比較した。

### off + narrow

- `A2_semaux_light_vhalf_tenpaifix_prnorm_off_narrow`
- `tile_presence_flags = false`
- `value_hidden_dims = [128, 64]`

### off + wide

- `A2_semaux_light_vhalf_tenpaifix_prnorm_off_widevalue`
- `tile_presence_flags = false`
- `value_hidden_dims = [256, 128]`

### on + narrow

- `A2_semaux_light_vhalf_tenpaifix_prnorm_on_narrow`
- `tile_presence_flags = true`
- `value_hidden_dims = [128, 64]`

### on + wide

- `A2_semaux_light_vhalf_tenpaifix_prnorm_on_widevalue`
- `tile_presence_flags = true`
- `value_hidden_dims = [256, 128]`

共通:

- seed `42`
- `value_loss_coef = 0.125`
- `terminal_loss_coef = 0.1`
- `yaku_loss_coef = 0.05`

## 4. 主結果

### 4.1 final policy

#### `exp_015 A2` 参考

- final: `avg_rank=2.345`
- `win_rate=0.2540`
- `deal_in_rate=0.1781`

#### `exp_017 off_narrow`

- imitation: `avg_rank=2.355`
- final: `avg_rank=2.375`
- `win_rate=0.2589`
- `deal_in_rate=0.1864`

#### `exp_017 off_wide`

- imitation: `avg_rank=2.535`
- final: `avg_rank=2.375`
- `win_rate=0.2396`
- `deal_in_rate=0.1808`

#### `exp_017 on_narrow`

- imitation: `avg_rank=2.570`
- final: `avg_rank=2.460`
- `win_rate=0.2333`
- `deal_in_rate=0.1908`

#### `exp_017 on_wide`

- imitation: `avg_rank=2.495`
- final: `avg_rank=2.420`
- `win_rate=0.2576`
- `deal_in_rate=0.1809`

読み:

- `on_narrow -> on_wide` で、
  - `avg_rank 2.460 -> 2.420`
  - `win_rate 0.2333 -> 0.2576`
  - `deal_in 0.1908 -> 0.1809`
  とかなり回復
- つまり、`yakuflags` の悪化は単純な「特徴量が悪い」ではなく、
  **narrow trunk では処理しきれない**成分が大きい
- 一方で `on_wide` でも `exp_015 A2` (`2.345`) には届いていない

### 4.2 best cycle / retain

#### off_narrow

- best cycle: `cycle 5`
- best `avg_rank=2.365`
- final `2.375`

#### off_wide

- best cycle: `cycle 19`
- best `avg_rank=2.375`
- final `2.375`

#### on_narrow

- best cycle: `cycle 8`
- best `avg_rank=2.305`
- final `2.460`

#### on_wide

- best cycle: `cycle 10`
- best `avg_rank=2.345`
- final `2.420`

読み:

- `on_narrow` は peak は高いが retain が悪い
- `on_wide` は peak を大きく上げるというより、
  **retain を改善して final を持ち上げる**方向に効いている

## 5. diagnostics

今回は各 run の `cycle_19/selfplay` shard を固定し、

- `imitation`
- `best cycle`
- `final`

を同じ shard 上で比較した。

### 5.1 terminal

#### off_narrow final

- terminal accuracy: `0.6599`
- `win_menzen recall = 0.1155`
- `win_called recall = 0.0703`
- `draw_tenpai recall = 0.0149`

#### off_wide final

- terminal accuracy: `0.6660`
- `win_menzen recall = 0.0803`
- `win_called recall = 0.0553`
- `draw_tenpai recall = 0.0119`

#### on_narrow final

- terminal accuracy: `0.6627`
- `win_menzen recall = 0.0917`
- `win_called recall = 0.0556`
- `draw_tenpai recall = 0.0037`

#### on_wide final

- terminal accuracy: `0.6614`
- `win_menzen recall = 0.0329`
- `win_called recall = 0.0907`
- `draw_tenpai recall = 0.0040`

読み:

- `on_narrow -> on_wide` で `win_called` は回復
- ただし `win_menzen` は大きく落ちる
- `draw_tenpai` はどちらも弱いまま

つまり、wide 化は semantic 全体を均等に改善したわけではなく、
**`yakuflags` が拾う方向の class を持ち上げつつ、別の class と再配分している**
と見るのが自然である。

### 5.2 yaku overall

#### off_narrow final

- micro F1: `0.4780`
- macro F1: `0.1114`
- exact match: `0.4229`

#### off_wide final

- micro F1: `0.5779`
- macro F1: `0.1062`
- exact match: `0.4374`

#### on_narrow final

- micro F1: `0.5534`
- macro F1: `0.1030`
- exact match: `0.4077`

#### on_wide final

- micro F1: `0.5177`
- macro F1: `0.0983`
- exact match: `0.3973`

読み:

- `on_wide` は **overall yaku 指標ではまだ最良ではない**
- したがって、今回の wide 化は
  - `yakuflags` を活かす
  - しかし overall yaku 最適化まではまだ到達していない
という段階である

### 5.3 `Tanyao`

今回の本命はここである。

#### off_narrow

- imitation:
  - recall `0.0575`
  - `mean_p=0.1741`
  - `hit@0.2=0.2632`
- final:
  - recall `0.0000`
  - `mean_p=0.1440`
  - `hit@0.2=0.1487`

#### off_wide

- imitation:
  - recall `0.0068`
  - `mean_p=0.1790`
  - `hit@0.2=0.4044`
- final:
  - recall `0.0000`
  - `mean_p=0.0997`
  - `hit@0.2=0.0328`

#### on_narrow

- imitation:
  - recall `0.0032`
  - `mean_p=0.1022`
  - `hit@0.2=0.0933`
- final:
  - recall `0.0000`
  - `mean_p=0.0981`
  - `hit@0.2=0.0823`

#### on_wide

- imitation:
  - recall `0.0194`
  - `mean_p=0.1810`
  - `hit@0.2=0.3835`
- final:
  - recall `0.0000`
  - `mean_p=0.2089`
  - `hit@0.2=0.4920`

読み:

- recall は依然として `0.0`
- しかし `on_narrow -> on_wide` で、
  - `mean_p 0.0981 -> 0.2089`
  - `hit@0.2 0.0823 -> 0.4920`
  と **非常に大きく改善**
- これは、`yakuflags` が `Tanyao` に対して完全に無意味なのではなく、
  **wide trunk があって初めて確率信号として立ち上がる**
ことを強く示している

ここが今回の最重要発見である。

### 5.4 `Riichi` / `Yakuhai`

#### on_narrow final

- `Riichi recall = 0.7621`
- `Yakuhai recall = 0.6532`

#### on_wide final

- `Riichi recall = 0.4472`
- `Yakuhai recall = 0.8109`

読み:

- wide 化は `Riichi` と `Yakuhai` の間でも配分を変えている
- `Tanyao` signal を持ち上げることと、
  既存の簡単な役とのバランスがまだ取れていない

### 5.5 `deal_in` risk

#### off_narrow final

- overall `pr_auc = 0.1992`
- `late_and_noten = 0.1750`
- `early_and_tenpai = 0.1061`

#### off_wide final

- overall `pr_auc = 0.1996`
- `late_and_noten = 0.1710`
- `early_and_tenpai = 0.1125`

#### on_narrow final

- overall `pr_auc = 0.1915`
- `late_and_noten = 0.1751`
- `early_and_tenpai = 0.1011`

#### on_wide final

- overall `pr_auc = 0.2031`
- `late_and_noten = 0.1946`
- `early_and_tenpai = 0.1101`

読み:

- `on_wide` は `deal_in` risk の分離も改善している
- ここでも、wide 化は `yakuflags` の副作用を減らすだけでなく、
  semantic 側の一部 signal をちゃんと使えるようにしている

## 6. 解釈

今回の結果から言えることは次の通り。

1. `CQ-0270` のアイデアは死んでいない  
   `exp_016` だけを見ると失敗に見えたが、`exp_017` では
   `on_narrow -> on_wide` で `Tanyao` の確率信号が明確に改善した

2. ただし wide にしただけではまだ足りない  
   `on_wide` は `Tanyao` signal を持ち上げるが、
   overall yaku / policy の最良条件にはなっていない

3. 問題は「特徴量が悪い」より「今の shared integration が不十分」寄り  
   少なくとも、
   - `yakuflags` は narrow では悪い
   - wide ならかなりマシ
   という結果は capacity 依存の相互作用を示している

## 7. 実務判断

現時点の practical baseline は、引き続き

- `A2_semaux_light_vhalf_tenpaifix_prnorm` (`exp_015`)

とする。

理由:

- `on_wide` でも final policy は `exp_015` baseline に届いていない
- overall yaku もまだ明確な勝ちにはなっていない

ただし、

- `CQ-0270` を完全に棄却する理由はなくなった
- 今後の yaku 改善では、
  - wide trunk
  - yaku 専用接続
  - yaku loss の再配分

のいずれかと組み合わせて再検証する価値が高い

## 8. 次の候補

自然な次の一手は次のどちらかである。

1. `on_wide` を土台にして、
   `yakuflags` を shared 入力ではなく yaku 側だけに効く形へ移す
2. `on_wide` を土台にして、
   yaku loss を少しだけ調整し、
   `Tanyao` signal を overall 指標につなげられるかを見る

今回の `exp_017` は、
**`yakuflags` が本質的に悪いわけではなく、今の narrow/shared 条件では活かしきれない**
ということを示した実験だった。

