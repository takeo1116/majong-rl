# Experiment Report: exp_016

作成日: 2026-04-09  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_016/runbook.md`
- `experiments/Stage02_CallUnlock/exp_015/report.md`
- `experiments/Stage02_CallUnlock/exp_016/compare_exp015_A2_cycle19/semantic_eval_imitation_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_016/compare_exp015_A2_cycle19/semantic_eval_final_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_016/semantic_eval_A2_semaux_light_vhalf_tenpaifix_prnorm_yakuflags_cycle19/semantic_eval_imitation_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_016/semantic_eval_A2_semaux_light_vhalf_tenpaifix_prnorm_yakuflags_cycle19/semantic_eval_cycle10_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_016/semantic_eval_A2_semaux_light_vhalf_tenpaifix_prnorm_yakuflags_cycle19/semantic_eval_final_cycle19_summary.md`

## 1. 要約

`exp_016` は、`exp_015` の practical baseline

- `A2_semaux_light_vhalf_tenpaifix_prnorm`

に対して、`CQ-0270` の self tile-presence flags を加えた再実験である。

今回の結論は次の通り。

- `CQ-0270` は、**現状の shared input への常時追加という形では baseline 更新に失敗**
- policy 指標は `exp_015` より悪化した
- diagnostics 上も、今回の本命だった **`Tanyao` 改善は確認できなかった**
- 一方で
  - `Riichi`
  - `Yakuhai`
  - terminal の `win_menzen / win_called`
  は改善しており、追加特徴が完全に無意味だったとは言いにくい
- したがって、**baseline は `exp_015 A2` 維持**
- `CQ-0270` の特徴量アイデア自体は保留しつつ、今後は
  - on/off を切り替えて比較できるようにする
  - trunk 幅や接続位置を変えて再検証する
のが自然

## 2. 背景

`exp_015` までで、次の整理は反映済みだった。

- `CQ-0265`
  - `shanten_hint / discard_ukeire_hint` を shared trunk から外し、discard direct hint に戻した
- `CQ-0266`
  - terminal semantic を 5-class に整理した
- `CQ-0267`
  - `self_tenpai_flag / remaining_draws_norm` を追加した
- `CQ-0268`
  - terminal semantic loss に player-round 正規化を導入した

その結果、`exp_015` では

- `A2_semaux_light_vhalf_tenpaifix_prnorm`

が practical baseline になった。

一方で yaku は、

- `Riichi`
- `Yakuhai`

は比較的学べているが、

- `Tanyao`
- `Pinfu`
- `MenzenTsumo`

などは late な局面でもほぼ立っていなかった。

この問題に対して、`CQ-0270` では

- `self_has_honor`
- `self_has_terminal`
- `self_has_simple`
- `self_has_man`
- `self_has_pin`
- `self_has_sou`

の 6 特徴を追加し、特に `Tanyao` のような「存在しないこと」が条件になる役を MLP が読みやすくすることを狙った。

## 3. 今回の差分

`exp_016` の学習差分として重要なのは実質 `CQ-0270` である。

加えて、同時点のコードには次も入っている。

- `CQ-0269`
  - `deal_in` risk diagnostics を `semantic_eval` に追加
- `CQ-0271`
  - Stage2a discard の legal snapshot と concrete action 解決を同一化

ただし、

- `CQ-0269` は評価改善
- `CQ-0271` は run 信頼性改善

であり、学習挙動そのものの差分は主に `CQ-0270` と見てよい。

## 4. 条件

比較条件:

### C0: control + yakuflags

- semantic aux 無効
- `value_loss_coef = 0.125`
- `CQ-0270` の tile-presence flags あり

### A2: semantic aux light + yakuflags

- semantic aux 有効
- `terminal_loss_coef = 0.1`
- `yaku_loss_coef = 0.05`
- `value_loss_coef = 0.125`
- `CQ-0270` の tile-presence flags あり

### A2 seed43

- A2 と同条件
- seed のみ `43`

共通:

- direct hint branch (`CQ-0265`)
- terminal 5-class (`CQ-0266`)
- `self_tenpai_flag / remaining_draws_norm` (`CQ-0267`)
- terminal player-round 正規化 (`CQ-0268`)

## 5. 主結果

### 5.1 final

#### `exp_015 A2` 比較基準

- final: `avg_rank=2.345`
- `win_rate=0.2540`
- `deal_in_rate=0.1781`

#### `exp_016 C0`

- final: `avg_rank=2.500`
- `win_rate=0.2361`
- `deal_in_rate=0.1891`

#### `exp_016 A2 seed42`

- final: `avg_rank=2.470`
- `win_rate=0.2483`
- `deal_in_rate=0.1938`

#### `exp_016 A2 seed43`

- final: `avg_rank=2.505`
- `win_rate=0.2466`
- `deal_in_rate=0.1928`

読み:

- `A2 seed42` は `C0` よりは少し良い
- しかし `exp_015 A2` よりは **明確に悪化**
- `seed43` でも同じ傾向で、seed を変えても改善は見えない
- `deal_in_rate` も `exp_015` より悪い

したがって、**policy の観点では `yakuflags` は採用失敗**と見るのが妥当である。

### 5.2 imitation / best cycle / tail-5

#### C0

- imitation: `avg_rank=2.40`
- best cycle: `cycle 15`, `avg_rank=2.260`
- tail-5 average: `2.431`
- final: `2.500`

#### A2 seed42

- imitation: `avg_rank=2.57`
- best cycle: `cycle 10`, `avg_rank=2.345`
- tail-5 average: `2.436`
- final: `2.470`

#### A2 seed43

- imitation: `avg_rank=2.43`
- best cycle: `cycle 10`, `avg_rank=2.290`
- tail-5 average: `2.441`
- final: `2.505`

読み:

- `A2 seed42` は imitation 時点でかなり悪い
- 途中で一時的に戻る局面はあるが、`exp_015` のような final までの保持はない
- `A2 seed43` も best cycle は悪くないが、final まで保持できていない

## 6. diagnostics

比較は次のように揃えた。

- `exp_015 A2 final`
- `exp_016 A2 imitation`
- `exp_016 A2 cycle10` (best cycle)
- `exp_016 A2 final`

すべて `cycle_19/selfplay` shard 上で評価した。

### 6.1 terminal

#### `exp_015 A2 final`

- terminal accuracy: `0.6541`
- `win_menzen recall = 0.0884`
- `win_called recall = 0.0779`
- `draw_tenpai recall = 0.0166`
- `deal_in recall = 0.0000`

#### `exp_016 A2 final`

- terminal accuracy: `0.6646`
- `win_menzen recall = 0.0951`
- `win_called recall = 0.0871`
- `draw_tenpai recall = 0.0085`
- `deal_in recall = 0.0000`

読み:

- terminal 全体の accuracy は少し改善
- `win_menzen / win_called` も少し改善
- ただし `draw_tenpai` は後退

つまり、`yakuflags` は terminal 側にも影響しており、
**一部 class を少し持ち上げる代わりに `draw_tenpai` を削る**方向に働いている。

### 6.2 yaku

#### `exp_015 A2 final`

- micro F1: `0.5007`
- macro F1: `0.0965`
- exact match: `0.4339`

主要役:

- `Riichi recall = 0.3918`
- `Yakuhai recall = 0.9121`
- `Tanyao recall = 0.0000`
- `Pinfu recall = 0.0000`
- `MenzenTsumo recall = 0.0000`

#### `exp_016 A2 final`

- micro F1: `0.5966`
- macro F1: `0.1069`
- exact match: `0.4443`

主要役:

- `Riichi recall = 0.8672`
- `Yakuhai recall = 0.8003`
- `Tanyao recall = 0.0000`
- `Pinfu recall = 0.0000`
- `MenzenTsumo recall = 0.0000`

読み:

- yaku 全体の aggregate 指標は改善している
- ただし改善の中心は **`Riichi` の大幅上昇**
- `Tanyao` は **最終的に 0 のまま**
- `Pinfu` / `MenzenTsumo` も立っていない

つまり、今回の追加特徴は

- yaku を広く改善した

というより、

- **既に見えやすい役をさらに強くした**

と解釈する方が自然である。

### 6.3 `Tanyao` の見え方

今回の一番重要な点はここである。

#### `exp_015 imitation`

- `Tanyao recall = 0.0444`
- `mean_p = 0.1652`

#### `exp_016 imitation`

- `Tanyao recall = 0.0030`
- `mean_p = 0.1039`

#### `exp_016 final`

- `Tanyao recall = 0.0000`
- `mean_p = 0.0816`

読み:

- `CQ-0270` の狙いだった `Tanyao` 改善は **起きていない**
- むしろ imitation 時点から悪化しており、
  PPO で壊れたというより **最初から期待した学習方向に乗っていない**

この点が、今回 baseline 更新を見送る最大の理由である。

### 6.4 `deal_in` risk

`deal_in` は top-1 ではなく risk signal として見る。

#### `exp_015 A2 final`

- overall `pr_auc = 0.2018`
- `mean_p_pos = 0.1766`
- `mean_p_neg = 0.1737`
- `late_and_noten pr_auc = 0.1662`
- `early_and_tenpai pr_auc = 0.0970`

#### `exp_016 A2 final`

- overall `pr_auc = 0.1963`
- `mean_p_pos = 0.1655`
- `mean_p_neg = 0.1638`
- `late_and_noten pr_auc = 0.1767`
- `early_and_tenpai pr_auc = 0.0896`

読み:

- overall では微悪化
- `late_and_noten` だけ少し改善
- しかし総合すると **横ばいか微悪化**

`CQ-0270` は `deal_in` risk を良くしたとも言いにくい。

## 7. 解釈

今回の結果から言えることは次の通り。

### 7.1 `CQ-0270` の特徴量アイデア自体は即棄却ではない

追加した presence flags は、

- `has_honor`
- `has_terminal`
- `has_simple`

のように、役判断に意味のある情報であること自体は自然である。

したがって、**特徴量の発想そのものが誤りだったとはまだ言わない**。

### 7.2 ただし「shared input にそのまま常時入れる」形は筋が悪い

現実に起きたことは、

- `Tanyao` は改善しない
- `Riichi` はかなり改善する
- terminal の一部も少し改善する
- しかし policy は悪化する

であった。

これは、

- 情報が完全に無意味

というより、

- **今の trunk / 最適化条件では、学びやすい signal に吸われた**

と見る方が自然である。

### 7.3 practical baseline は `exp_015 A2` 維持

したがって現時点では、

- **`A2_semaux_light_vhalf_tenpaifix_prnorm`**

を baseline として維持するのが妥当である。

`CQ-0270` をそのまま常時有効化した `exp_016` 系は、現時点では採用しない。

## 8. 次の方針

今回の結果を踏まえた自然な次の手は次の通り。

1. `tile_presence_flags` を config で on/off できるようにする
2. `exp_017` で
   - `yakuflags なし`
   - `yakuflags あり`
   - trunk 幅そのまま
   - trunk 幅拡張
   を比較する
3. 特徴量アイデア自体は保留しつつ、
   - 今の shared input が悪いのか
   - semantic/value trunk の容量不足が悪いのか
   を切り分ける

つまり、`exp_016` は

- **`CQ-0270` を baseline 採用しない**

という判断を与えた実験であると同時に、

- **特徴量アイデア自体は trunk 容量や接続位置を変えて再検証する価値がある**

ことを示した実験でもある。
