# Experiment Report: exp_067

作成日: 2026-03-22  
参照:
- `experiments/exp_067/runbook.md`
- `experiments/exp_067/run_map.json`

注記:
- この report は、`runs/` 配下の実験成果物を削除しても後から困りにくいよう、通常より詳細に数値を転記して残す。
- 本 report の数値はすべて `seed=42` の単発結果である。
- `deal_in_rate` は現在の evaluator 実装上、純粋な「ロン放銃率」ではない。  
  正確には、**局終了時に policy 席が失点し、かつ他家の誰かが得点した局の割合**である。  
  したがって、他家ツモによる失点や流局ノーテン罰符も混ざりうる。

## 1. 目的

CQ-0210 / CQ-0211 修正後の corrected semantics 上で、baseline から 1 ノブだけ変えた単発実験を広く打ち、次に 3 seeds で詰める候補を探す。

今回の問いは主に次の 3 点だった。

1. 修正後 semantics でも `gamma=0.75, gae=0.3` が妥当か
2. `anchor`, `clip`, `policy_ratio`, `shanten_scale` のどれが final 保持に効くか
3. `improve / same / worsen` の向きの自然さを保ったまま final を改善できるか

## 2. 条件一覧

基準条件は corrected baseline `REF g075_gae030_ref` で、そこから 1 個だけ変更した。

| 条件 | 変更内容 |
|---|---|
| `REF` | `gamma=0.75, gae=0.3, clip=0.15, anchor=0.5, policy_ratio=0.0, shanten_scale=0.003` |
| `A gamma090` | `gamma=0.90` |
| `B gae000` | `gae_lambda=0.0` |
| `C clip010` | `clip_epsilon=0.10` |
| `D anchor075` | `policy_anchor.coef=0.75` |
| `E ratio005` | `policy_ratio=0.05` |
| `F shape001` | `shanten_scale=0.001` |
| `G clip020` | `clip_epsilon=0.20` |
| `H anchor025` | `policy_anchor.coef=0.25` |
| `I ratio010` | `policy_ratio=0.10` |
| `J shape000` | `shanten_scale=0.0` |

## 3. 主結果

### 3.1 `init / best / final` の全結果

ここでの `init_score` は `cycle0.eval_before.avg_score`、つまり **imitation 直後・PPO 更新前** の評価である。

| cond | knob | init_score | best_score | best_cycle | final_score | final-init | drawdown | final_rank | final_win | final_deal_in |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| REF | g075_gae030_ref | 2175.75 | 2547.25 | 24 | 1249.50 | -926.25 | 1297.75 | 2.4250 | 0.2799 | 0.4618 |
| A | gamma090 | 2601.25 | 2937.50 | 13 | 859.00 | -1742.25 | 2078.50 | 2.4400 | 0.2808 | 0.4551 |
| B | gae000 | 2572.00 | 3095.50 | 4 | 1318.00 | -1254.00 | 1777.50 | 2.3700 | 0.2812 | 0.4634 |
| C | clip010 | 2175.75 | 3262.75 | 14 | 2032.25 | -143.50 | 1230.50 | 2.3925 | 0.2893 | 0.4555 |
| D | anchor075 | 2175.75 | 2729.00 | 11 | 2076.00 | -99.75 | 653.00 | 2.3750 | 0.2932 | 0.4511 |
| E | ratio005 | 2175.75 | 2849.25 | 8 | 1449.75 | -726.00 | 1399.50 | 2.4050 | 0.2939 | 0.4577 |
| F | shape001 | 2692.50 | 3069.00 | 14 | 1220.75 | -1471.75 | 1848.25 | 2.4225 | 0.2874 | 0.4561 |
| G | clip020 | 2175.75 | 2673.50 | 4 | 1721.25 | -454.50 | 952.25 | 2.3625 | 0.2838 | 0.4587 |
| H | anchor025 | 2175.75 | 2356.00 | 1 | 1024.25 | -1151.50 | 1331.75 | 2.4275 | 0.2770 | 0.4588 |
| I | ratio010 | 2175.75 | 3431.00 | 4 | 1792.25 | -383.50 | 1638.75 | 2.3700 | 0.2879 | 0.4498 |
| J | shape000 | 2205.00 | 2869.50 | 12 | 1335.00 | -870.00 | 1534.50 | 2.4250 | 0.2767 | 0.4624 |

### 3.2 final learner 診断

| cond | clip_fraction | ratio_std | best_set_after | value_error_mean | policy_samples | baseline_samples | improve_adv | same_adv | worsen_adv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| REF | 0.1912 | 0.1236 | 0.9003 | 0.0081 | 0 | 88212 | 0.1050 | -0.0318 | -0.0087 |
| A | 0.1950 | 0.1236 | 0.9004 | 0.0009 | 0 | 88212 | 0.1251 | -0.0359 | -0.0263 |
| B | 0.2094 | 0.1347 | 0.9018 | 0.0049 | 0 | 88212 | 0.1104 | -0.0347 | 0.0012 |
| C | 0.2677 | 0.1082 | 0.9060 | 0.0072 | 0 | 88212 | 0.1048 | -0.0319 | -0.0075 |
| D | 0.1511 | 0.1109 | 0.9088 | 0.0083 | 0 | 88212 | 0.1064 | -0.0318 | -0.0120 |
| E | 0.1975 | 0.1295 | 0.9019 | 0.0087 | 4038 | 83441 | 0.1006 | -0.0307 | -0.0066 |
| F | 0.1946 | 0.1271 | 0.8998 | 0.0114 | 0 | 88212 | 0.0974 | -0.0331 | 0.0241 |
| G | 0.1649 | 0.1516 | 0.8952 | 0.0059 | 0 | 88212 | 0.1062 | -0.0318 | -0.0111 |
| H | 0.2486 | 0.1487 | 0.8861 | 0.0086 | 0 | 88212 | 0.1044 | -0.0324 | -0.0015 |
| I | 0.1749 | 0.1146 | 0.9019 | -0.0002 | 8001 | 80234 | 0.1095 | -0.0305 | -0.0312 |
| J | 0.1964 | 0.1256 | 0.9009 | 0.0042 | 0 | 88212 | 0.0990 | -0.0309 | 0.0033 |

## 4. まず言えること

### 4.1 すべての条件で `best > init`

これは今回もっとも重要な観察である。

- どの条件でも、PPO は一度は imitation 直後を超えている
- したがって、**「imitation から先に強くなる余地がない」わけではない**
- 問題は **改善不能** ではなく、**改善の保持失敗** である

### 4.2 すべての条件で `final < init`

一方で、全条件で最終的には imitation 直後を下回った。

したがって corrected semantics で現時点の baseline は、

- 一時的には改善できる
- しかし 30 cycle 続けると、その改善を失う

という挙動になっている。

## 5. 条件ごとの読み

### 5.1 良かった条件

#### D `anchor075`

- final 最良: `2076.0`
- `final - init = -99.75`
- drawdown 最小: `653.0`
- `best_set_after = 0.9088` で全条件中最高
- `deal_in_rate` も `0.4551 -> 0.4511` で改善維持

読み:
- corrected semantics では、**anchor を強める方向が最も効いた**
- 改善不能ではなく、**drift / overshoot を抑えると final はかなり戻る**

#### C `clip010`

- final 次点: `2032.25`
- `final - init = -143.5`
- best は全条件中かなり高い: `3262.75`
- `ratio_std = 0.1082` でかなり小さい

読み:
- `clip=0.10` は **改善も作れて、final も比較的保てる**
- `clip_fraction` は高いが、threshold 依存なので単純比較は難しい
- 実質的には **更新強度を下げる方向がまだ効く**

#### I `ratio010`

- final `1792.25`
- `deal_in_rate = 0.4498` でかなり良い
- `worsen_adv = -0.0312` で最も自然
- `policy_samples = 8001`, `baseline_samples = 80234`

読み:
- 少量 actor mix は still 効いていそう
- ただし final score は D/C ほどは伸びない
- **distribution mismatch は残っているが、最大要因ではない**

### 5.2 悪かった条件

#### A `gamma090`

- final 最悪: `859.0`
- `final - init = -1742.25`
- drawdown も最大: `2078.5`

読み:
- post-fix でも `gamma=0.90` は長すぎる寄り
- 少なくとも current reward / target の組では本命ではない

#### H `anchor025`

- final `1024.25`
- `best_set_after = 0.8861` まで悪化
- `ratio_std = 0.1487`

読み:
- anchor を弱める方向はかなり悪い
- corrected semantics でも、**imitation anchor はまだ強く必要**

#### F/J `shanten_scale` 弱化・除去

- `shape001`: final `1220.75`
- `shape000`: final `1335.0`
- どちらも baseline `0.003` より悪い

読み:
- shaping はまだ必要
- corrected semantics になっても、いまは **shanten shaping を弱めるフェーズではない**

## 6. `final < init` の原因

今回のデータから、`final < init` の原因はかなり絞れる。

### 6.1 主因は「改善不能」ではなく「保持失敗」

根拠:
- 全条件で `best > init`
- つまり PPO は一度は良い方向に動けている
- その後の更新で、良い領域から外れている

### 6.2 `win_rate` は全条件で低下

`init -> final` で、全条件で `win_rate` が下がっている。

例:
- REF: `0.2989 -> 0.2799`
- D `anchor075`: `0.2989 -> 0.2932`
- C `clip010`: `0.2989 -> 0.2893`
- I `ratio010`: `0.2989 -> 0.2879`

これは重要で、**final < init の説明には攻撃性能の低下が必ず入っている。**

### 6.3 `deal_in_rate` は多くの条件で悪化、ただしすべてではない

`deal_in_rate` は現在の実装では純粋な放銃率ではなく、  
**局終了時の失点イベント率**に近い指標である。

良い条件:
- D `anchor075`: `0.4551 -> 0.4511`
- I `ratio010`: `0.4551 -> 0.4498`

悪い条件:
- REF: `0.4551 -> 0.4618`
- B `gae000`: `0.4522 -> 0.4634`
- J `shape000`: `0.4534 -> 0.4624`

読み:
- 悪い条件では **失点イベント率も悪化** している
- ただし D/I のように **失点イベント率は改善しても final は init を超えない** 条件もある

したがって、

- 一部の条件では「守備崩壊」が final 悪化の原因
- しかし良い条件でも、**守備だけでは init を超え切れない**

### 6.4 teacher rail の緩やかな喪失

今回、良い条件ほど final `best_set_after` が高い。

- D `anchor075`: `0.9088`
- C `clip010`: `0.9060`
- REF: `0.9003`
- H `anchor025`: `0.8861`

また、良い条件ほど `ratio_std` が小さい。

- C `clip010`: `0.1082`
- D `anchor075`: `0.1109`
- H `anchor025`: `0.1487`
- G `clip020`: `0.1516`

読み:
- corrected semantics にしても、**最終的な performance はまだ residual drift / overshoot に強く支配されている**
- その意味で、今の問題は target より **optimization / training regime** に近い

## 7. `improve / same / worsen` の向き

今回のデータでは、以前の不自然さはかなり解消している。

ほぼ全条件で:
- `improve_adv > 0`
- `same_adv < 0`

例:
- REF: `+0.1050 / -0.0318 / -0.0087`
- D: `+0.1064 / -0.0318 / -0.0120`
- C: `+0.1048 / -0.0319 / -0.0075`

例外的に、
- B `gae000`
- F `shape001`
- J `shape000`

では `worsen_adv` が 0 付近または正に寄っている。

読み:
- **signal の向きは、bugfix によりかなり健全化した**
- したがって現時点では、まずは target / advantage の再定義より **hyperparameter / training regime の見直し** を優先すべきである

## 8. いま何が本丸か

今回の 11 条件から見る限り、

1. corrected semantics でも **改善余地はある**
2. しかし **30 cycle 続けると保持に失敗する**
3. 保持には
   - `anchor` を強める
   - `clip` を下げる
   が最も効く
4. `policy_ratio` も一定の効果はあるが、主因ではなさそう
5. `gamma=0.90` や shaping 弱化は悪い

このため、現時点の優先順位は

1. **hyperparameter / regime 再調整**
2. それでもダメなら target / reward / advantage 再設計

である。

## 9. 実務的な次の一手

今回の結果だけで次に見る価値が高い組み合わせは次の通り。

1. `anchor=0.75` + `clip=0.10`
2. `anchor=0.75` + `policy_ratio=0.10`
3. `clip=0.10` + `policy_ratio=0.10`

理由:
- D と C が最も強かった
- I も secondary improvement を示した
- corrected semantics で最初に詰めるべきは **保持寄り設定の組み合わせ** である

## 10. 短いまとめ

- bugfix 後、`improve / same / worsen` の向きはかなり自然になった
- しかし `final < init` はまだ全条件で残る
- その主因は **学習不能ではなく、改善保持の失敗**
- 今回最も効いたのは
  - `anchor=0.75`
  - `clip=0.10`
- `policy_ratio=0.10` も一定の効果
- `gamma=0.90` と shaping 弱化は悪い
- したがって、次は **保持寄り hyperparameter の組み合わせ** を先に見るべきである
