# Experiment Report: exp_019

作成日: 2026-04-10  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_019/runbook.md`
- `experiments/Stage02_CallUnlock/exp_015/report.md`
- `experiments/Stage02_CallUnlock/exp_017/report.md`
- `experiments/Stage02_CallUnlock/exp_018/report.md`

## 1. 要約

`exp_019` は、

- `exp_015 A2` baseline
- `exp_017 on_wide`

の比較を **paired 3-seed** でやり直し、
`yakuflags on + wide` が practical baseline を更新できるかを確かめる実験である。

結論は次の通り。

- **3 seed で見ても `on_wide` は baseline を上回らなかった**
- final `avg_rank` は **3/3 seed で baseline の方が良い**
- final `win_rate` はかなり拮抗しているが、`avg_rank` と `deal_in_rate` を含めると
  practical な優位は baseline 側に残る
- したがって、**現在の実験環境では `yakuflags` は性能向上に寄与しない**
  という整理でよい

この結果により、

- practical baseline は引き続き `exp_015 A2`
- `exp_017 on_wide` は「面白い diagnostics を示した研究候補」ではあるが、
  現時点では採用しない

という判断がかなり頑健になった。

## 2. 背景

`exp_016` では shared input として `tile_presence_flags` を追加したが、
policy は悪化した。

その後 `exp_017` の 2x2 実験で、

- `yakuflags on + narrow` は悪い
- `yakuflags on + wide` ではかなり回復
- 特に `Tanyao` の確率信号は改善

という結果が出た。

ただしその時点では、

- baseline 側も
- `on_wide` 側も

ほぼ seed42 の 1 点比較に依存していた。

そのため今回、seed43 / seed44 を baseline と `on_wide` の両方で追加し、
seed42 を含めた **paired 3-seed 比較**にした。

## 3. 条件

### 3.1 baseline 条件

- `A2_semaux_light_vhalf_tenpaifix_prnorm`
- `tile_presence_flags = false`
- `value_hidden_dims = [128, 64]`

### 3.2 on_wide 条件

- `A2_semaux_light_vhalf_tenpaifix_prnorm_on_widevalue`
- `tile_presence_flags = true`
- `value_hidden_dims = [256, 128]`
- `tile_presence_flags_semantic_only = false`

### 3.3 seed

比較 seed:

- `42` (既存アンカー)
- `43`
- `44`

## 4. 主結果

### 4.1 seed ごとの final

#### seed42

baseline:

- final `avg_rank = 2.345`
- `win_rate = 0.2540`
- `deal_in_rate = 0.1781`

on_wide:

- final `avg_rank = 2.420`
- `win_rate = 0.2576`
- `deal_in_rate = 0.1809`

差分 (`on_wide - baseline`):

- `avg_rank +0.075`
- `win_rate +0.0037`
- `deal_in +0.0028`

#### seed43

baseline:

- final `avg_rank = 2.455`
- `win_rate = 0.2461`
- `deal_in_rate = 0.2028`

on_wide:

- final `avg_rank = 2.600`
- `win_rate = 0.2464`
- `deal_in_rate = 0.2028`

差分:

- `avg_rank +0.145`
- `win_rate +0.0003`
- `deal_in +0.0000`

#### seed44

baseline:

- final `avg_rank = 2.400`
- `win_rate = 0.2511`
- `deal_in_rate = 0.1865`

on_wide:

- final `avg_rank = 2.450`
- `win_rate = 0.2447`
- `deal_in_rate = 0.1969`

差分:

- `avg_rank +0.050`
- `win_rate -0.0064`
- `deal_in +0.0103`

読み:

- **3/3 seed で final `avg_rank` は baseline の方が良い**
- `win_rate` は seed42 でわずかに `on_wide` が上だが、seed43 はほぼ同等、seed44 は baseline が上
- `deal_in_rate` は overall に `on_wide` が少し悪い

### 4.2 3 seed 平均

baseline:

- imitation `avg_rank = 2.393`
- final `avg_rank = 2.400`
- final `win_rate = 0.2504`
- final `deal_in_rate = 0.1891`

on_wide:

- imitation `avg_rank = 2.553`
- final `avg_rank = 2.490`
- final `win_rate = 0.2496`
- final `deal_in_rate = 0.1935`

読み:

- final `avg_rank` は **`2.400 -> 2.490`** で `on_wide` が明確に悪い
- `win_rate` はほぼ同じ
- しかし `deal_in_rate` も少し悪化

したがって、practical な総合評価では baseline 維持が自然である。

## 5. 解釈

今回の multi-seed 比較で、次のことがかなりはっきりした。

### 5.1 `on_wide` は「偶然 baseline に届かなかった」わけではない

`exp_017` seed42 単体では、

- `on_wide` は baseline にかなり近い
- diagnostics では `Tanyao` signal が良い

ため、「seed を増やせば逆転するかもしれない」という余地があった。

しかし今回、

- seed43
- seed44

を追加しても、むしろ `avg_rank` 差は安定して baseline 側に寄った。

つまり、

- `on_wide` は偶然不利だった

という解釈は採りにくくなった。

### 5.2 diagnostics の良さと practical performance は分かれた

`exp_017` で見えた

- `Tanyao mean_p`
- `hit@0.2`

の改善自体は、依然として興味深い。

ただし今回は、

- final `avg_rank`
- `deal_in_rate`
- paired seed 比較

を見た結果、**それが実用性能の向上には結びついていない**
という整理が妥当になった。

つまり現時点では、

- `yakuflags` は semantic/yaku 的には面白い信号を作る
- しかしこの環境では policy 改善にはつながらない

と考えるのが自然である。

## 6. 実務判断

現時点の整理は次の通り。

- practical baseline: **`exp_015 A2` 維持**
- `exp_017 on_wide`: 不採用
- `exp_018 semantic_only`: 不採用

今回の `exp_019` で、少なくともこの実験系においては

- `tile_presence_flags` を入れる工夫

は、いまのところ **性能向上に寄与しない** という結論でよい。

## 7. 今後の含意

今回の結論は、

- `tile_presence_flags` の発想が論理的に間違っている

という意味ではない。

ただし、

- shared raw input
- semantic_only
- wide 化
- multi-seed

まで見たうえでなお practical 改善が出ていないため、
**しばらく `yakuflags` 系の追試を優先する必要は薄い**
と判断してよい。

今後もし再開するなら、

1. yaku head 直前の局所接続
2. 小さい専用 projection
3. loss 側の調整

のような、かなり別の設計になってからで十分である。

