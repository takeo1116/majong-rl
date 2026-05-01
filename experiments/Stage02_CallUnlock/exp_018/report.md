# Experiment Report: exp_018

作成日: 2026-04-10  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_018/runbook.md`
- `experiments/Stage02_CallUnlock/exp_015/report.md`
- `experiments/Stage02_CallUnlock/exp_016/report.md`
- `experiments/Stage02_CallUnlock/exp_017/report.md`

## 1. 要約

`exp_018` は、`CQ-0273` で追加した

- `tile_presence_flags_semantic_only=true`

を初めて試した実験である。

狙いは、

- `tile_presence_flags` を raw で policy trunk に直接入れるのは重い
- ただし semantic/value 側には有用かもしれない

という仮説を切ることだった。

結論は次の通り。

- **今回の semantic-only routing はうまくいかなかった**
- `narrow` / `wide` のどちらでも、policy は `exp_017 on_wide` より悪く、baseline (`exp_015 A2`) からも大きく離れた
- 意図せず duplicate run が 2 本ずつ走ったが、**4 本とも同じ方向の悪化**を示しており、結論はかなり頑健
- したがって、少なくとも現状の
  - `tile_presence_flags` を semantic/value 側だけに残し
  - policy には semantic summary 経由でしか渡さない
 という設計は採用しない

実務判断としては、

- practical baseline は引き続き `exp_015 A2`
- `exp_017 on_wide` は「有望だが未採用」の候補
- `exp_018 semantic_only` はいったん見送り

が自然である。

## 2. 背景

`exp_016` では、`CQ-0270` の self tile-presence flags

- `self_has_honor`
- `self_has_terminal`
- `self_has_simple`
- `self_has_man`
- `self_has_pin`
- `self_has_sou`

を shared encoder input に追加したが、policy は悪化した。

ただし `exp_017` の 2x2 実験で、

- `yakuflags on + narrow` は悪い
- `yakuflags on + wide` はかなり回復
- 特に `Tanyao` の確率信号は改善

という結果が出たため、

- 特徴量アイデア自体は完全には死んでいない
- 問題は「どこに入れるか」かもしれない

という整理になった。

そこで `CQ-0273` では、

- encoder は従来どおり `tile_presence_flags` を出す
- `value / terminal / yaku` 側には raw のまま入れる
- `discard / optional` の raw policy 入力からは除外する

という `semantic_only` routing を実装し、今回それを検証した。

## 3. 条件

今回は新規 2 条件のみを回し、比較アンカーとして `exp_017` を再利用する設計にした。

### 3.1 新規条件

#### on + semantic_only + narrow

- `A2_semaux_light_vhalf_tenpaifix_prnorm_on_semonly_narrow`
- `feature_encoder.tile_presence_flags = true`
- `model.semantic_aux.tile_presence_flags_semantic_only = true`
- `model.value_hidden_dims = [128, 64]`

#### on + semantic_only + wide

- `A2_semaux_light_vhalf_tenpaifix_prnorm_on_semonly_widevalue`
- `feature_encoder.tile_presence_flags = true`
- `model.semantic_aux.tile_presence_flags_semantic_only = true`
- `model.value_hidden_dims = [256, 128]`

### 3.2 比較アンカー

- `exp_015 A2`
- `exp_017 on_narrow`
- `exp_017 on_wide`

## 4. duplicate run について

今回は意図せず、同じ `exp_018` driver が二重起動していたため、
各条件が 2 回ずつ走った。

### narrow

- run1: `20260410_001540_A2_semaux_light_vhalf_tenpaifix_prnorm_on_semonly_narrow.log`
- run2: `20260410_002938_A2_semaux_light_vhalf_tenpaifix_prnorm_on_semonly_narrow.log`

### wide

- run1: `20260410_011120_A2_semaux_light_vhalf_tenpaifix_prnorm_on_semonly_widevalue.log`
- run2: `20260410_014032_A2_semaux_light_vhalf_tenpaifix_prnorm_on_semonly_widevalue.log`

`run_map.json` は後発の 2 本を採用 run として記録しているが、
分析上は 4 本とも同方向の結果を示しているため、むしろ結論の頑健性確認に使えた。

## 5. 主結果

### 5.1 比較基準

#### `exp_015 A2`

- imitation: `avg_rank=2.355`
- final: `avg_rank=2.345`
- `win_rate=0.2540`
- `deal_in_rate=0.1781`

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

### 5.2 `exp_018 on_semonly_narrow`

#### run1

- imitation: `avg_rank=2.520`
- final: `avg_rank=2.540`
- `win_rate=0.2362`
- `deal_in_rate=0.2025`

#### run2

- imitation: `avg_rank=2.490`
- final: `avg_rank=2.565`
- `win_rate=0.2377`
- `deal_in_rate=0.1825`

読み:

- imitation は `exp_017 on_narrow` より改善して見える
- しかし final はむしろ大きく悪化している
- つまり、**semantic_only routing は imitation の見え方を少し良くしても、
  PPO ではうまく保持されない**

### 5.3 `exp_018 on_semonly_wide`

#### run1

- imitation: `avg_rank=2.470`
- final: `avg_rank=2.525`
- `win_rate=0.2310`
- `deal_in_rate=0.2024`

#### run2

- imitation: `avg_rank=2.560`
- final: `avg_rank=2.600`
- `win_rate=0.2441`
- `deal_in_rate=0.1914`

読み:

- `wide` にしても改善しない
- `exp_017 on_wide` (`final 2.420`) と比べて明確に悪い
- したがって、**semantic/value 側だけに残せば `on_wide` の良さを保てる**
  という仮説は支持されなかった

## 6. 解釈

今回の結果から、少なくとも次のことが言える。

### 6.1 raw policy 入力を完全に切るのは弱すぎる

`exp_017 on_wide` では、

- raw flag を policy trunk にも入れる
- かつ value/semantic 側も wide

という条件で、policy はかなり回復していた。

一方 `exp_018` では、

- raw flag を policy trunk から完全に外し
- semantic summary 経由でしか policy に渡らない

ようにしたところ、`narrow` / `wide` を問わず悪化した。

このことから、

- raw flag を policy に直接入れるのは重い
- しかし semantic summary だけに任せるのは弱すぎる

という、**中間の接続が必要**な状態だと考えられる。

### 6.2 問題は単純な容量不足だけではない

`exp_017` では `wide` が有効だったので、容量仮説は部分的に正しかった。

しかし `exp_018` では `wide` にしてもダメだったため、今回の routing では

- signal の伝達経路
- policy への見せ方

の方がボトルネックになっている可能性が高い。

## 7. 実務判断

現時点の整理は次の通り。

- practical baseline: **`exp_015 A2` 維持**
- `exp_017 on_wide`: 有望だが未採用
- `exp_018 semantic_only`: 見送り

今回の結果だけで `tile_presence_flags` の発想自体を捨てる必要はないが、
少なくとも

- `semantic/value only`

という routing は、現状の設計では採らない方がよい。

## 8. 次の候補

次に試すなら、自然なのは以下のどちらかである。

1. **yaku head 直前だけに追加入力する**
   - semantic summary より直接的
   - raw policy trunk ほど広くない

2. **policy には限定的に見せる中間接続を作る**
   - 例: tile_presence_flags 専用の小 projection を policy に渡す
   - raw 全入力でも summary のみでもない中間形

少なくとも今回で、

- all-trunks raw
- semantic-only

の両極は見えたので、次はその中間を設計するのが自然である。

