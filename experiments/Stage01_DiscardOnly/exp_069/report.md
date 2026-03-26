# Experiment Report: exp_069

作成日: 2026-03-23  
参照:
- `experiments/exp_069/runbook.md`
- `experiments/exp_069/run_map.json`
- `experiments/exp_068/report.md`

注記:
- この report は、`runs/` 配下の実験成果物を削除しても後から困りにくいよう、通常より詳細に数値を転記して残す。
- `REF` は `exp_068` の B `anchor075_ratio010` 3-seed 集計を参照し、`exp_069` では新規再実行していない。
- `deal_in_rate` は現在の evaluator 実装上、純粋な「ロン放銃率」ではない。  
  正確には、**局終了時に policy 席が失点し、かつ他家の誰かが得点した局の割合**である。  
  したがって、他家ツモによる失点や流局ノーテン罰符も混ざりうる。

## 1. 目的

`exp_068` で current best 候補となった

- `anchor=0.75`
- `policy_ratio=0.10`

から出発し、`training.rule_mix.policy_ratio` を **より大きい帯 (`0.30 / 0.50 / 0.70`)** に振って、

1. actor mix をさらに増やすと plateau / final が改善するか
2. `policy_ratio` の最適帯が `0.10` 近傍なのか、それとももっと高い側なのか

を 3 seeds (`42, 43, 44`) で確認した。

## 2. 条件

共通固定:
- 新モデル (`policy_direct_hints + context_gate`)
- `gamma=0.75`
- `gae_lambda=0.3`
- `clip_epsilon=0.15`
- `policy_anchor.coef=0.75`
- `value_loss_coef=0.25`
- `shanten_scale=0.003`
- imitation `1000 x 3 chunks`
- PPO `200 x 30 cycles`
- seeds `42, 43, 44`

比較条件:
- `REF anchor075_ratio010`  
  `exp_068` B の 3-seed 集計を参照
- A `anchor075_ratio030`
- B `anchor075_ratio050`
- C `anchor075_ratio070`

## 3. 主結果

### 3.1 3-seed 平均比較

| 条件 | policy_ratio | init mean | best mean | final mean | final-init mean | drawdown mean | cycle20-29 mean | rank mean | win mean | deal_in mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| REF `anchor075_ratio010` | 0.10 | 2346.58 | 3230.92 | **2348.58** | **+2.00** | 882.33 | **2494.93** | 2.3558 | 0.2933 | 0.4516 |
| A `anchor075_ratio030` | 0.30 | 2034.83 | 2809.58 | 2050.58 | +15.75 | **759.00** | 2076.78 | **2.3408** | 0.2997 | **0.4458** |
| B `anchor075_ratio050` | 0.50 | 1393.08 | 3000.92 | 2169.08 | +776.00 | 831.83 | 2108.14 | 2.3542 | 0.2975 | 0.4493 |
| C `anchor075_ratio070` | 0.70 | 2220.00 | 2987.67 | 2055.08 | -164.92 | 932.58 | 2126.95 | 2.3542 | **0.3023** | 0.4488 |

まず一番大きい事実は、**score 系では `policy_ratio=0.10` が依然として最良**だったこと。

特に、noise に強い plateau 指標として見ていた `cycle 20-29 mean` は

- `0.10`: `2494.93`
- `0.30`: `2076.78`
- `0.50`: `2108.14`
- `0.70`: `2126.95`

で、`0.10` が明確に勝っている。

したがって、少なくとも今回の current setup では、**`policy_ratio` を 0.30 以上へ大きく上げる方向は score 改善につながらなかった**。

### 3.2 final learner 診断の比較

| 条件 | clip_fraction mean | ratio_std mean | value_error mean | best_set_after mean | num_policy_samples mean | num_baseline_samples mean | improve adv | same adv | worsen adv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| REF `ratio010` | 0.1444 | 0.1052 | 0.00061 | 0.9130 | 8604.33 | 80288.00 | +0.1014 | -0.0295 | -0.0193 |
| A `ratio030` | 0.1331 | 0.1041 | 0.00214 | 0.9201 | 25926.33 | 63738.00 | +0.0945 | -0.0267 | -0.0226 |
| B `ratio050` | 0.1072 | 0.0937 | 0.00265 | 0.9247 | 43270.00 | 45513.33 | +0.0873 | -0.0248 | -0.0194 |
| C `ratio070` | 0.0816 | 0.0823 | 0.00034 | 0.9255* | 61213.33 | 28531.67 | +0.0885 | -0.0244 | -0.0242 |

`*` 注: `ratio070` では 3 seeds のうち 2 本で一部 cycle の `num_best_set_samples=0` となり、その cycle の `best_set_hit_rate_before/after` は `None` だった。表の平均は `None` でない値だけから計算している。

この表から分かることはかなりはっきりしている。

- `policy_ratio` を上げるほど、`clip_fraction` は下がる
- `ratio_std` も下がる
- つまり **high ratio は update を荒くするのではなく、むしろ穏やかにする**
- それでも score が上がらないので、問題は「学習が暴れている」ことではない

今の自然な解釈は、

**actor の状態分布を増やしすぎると、baseline による良い誘導が薄れ、score の高い plateau を維持しにくくなる**

というもの。

一方で、signal の向きは壊れていない。

- 全条件で `improve_adv > 0`
- 全条件で `same_adv < 0`
- 全条件で平均的には `worsen_adv < 0`

なので、今回の差は **target/advantage の破綻ではなく regime の差** と読める。

## 4. 条件ごとの詳細

### REF `anchor075_ratio010`（参照点）

3-seed 平均:
- `final_score = 2348.58`
- `cycle20-29 mean = 2494.93`
- `drawdown = 882.33`
- `best_set_after = 0.9130`

`exp_068` で current best 候補になった条件で、今回も比較基準として十分強い。  
`cycle 20-29 mean` が imitation 直後平均を上回っていたことから、**final 単点ではなく後半 plateau で見たときに安定して強い**基準である。

### A `anchor075_ratio030`

seed ごと:
- seed 42: init `1841.00`, best `3150.75`, final `1337.50`, late mean `1820.63`
- seed 43: init `1682.25`, best `2204.75`, final `1748.25`, late mean `1895.68`
- seed 44: init `2581.25`, best `3073.25`, final `3066.00`, late mean `2514.03`

3-seed 平均:
- `final_score = 2050.58`
- `cycle20-29 mean = 2076.78`
- `drawdown = 759.00`
- `rank = 2.3408`
- `win = 0.2997`
- `deal_in = 0.4458`

読み:
- score は `0.10` よりかなり下
- ただし `drawdown` は最小
- rank / win / loss-event 系も悪くない

つまり、**安全側・安定側には寄るが、点数の高い plateau は落ちる** 条件である。

### B `anchor075_ratio050`

seed ごと:
- seed 42: init `1359.75`, best `2939.75`, final `1908.00`, late mean `1984.70`
- seed 43: init `704.25`, best `2194.25`, final `730.50`, late mean `1184.88`
- seed 44: init `2115.25`, best `3868.75`, final `3868.75`, late mean `3154.85`

3-seed 平均:
- `final_score = 2169.08`
- `cycle20-29 mean = 2108.14`
- `drawdown = 831.83`
- `rank = 2.3542`
- `win = 0.2975`
- `deal_in = 0.4493`

読み:
- `0.30` よりは score を戻す
- ただし `0.10` には届かない
- seed 間ばらつきが大きく、今回の中で最も読みづらい帯

一言で言うと、**当たる seed ではかなり伸びるが、平均では current best を超えない**。

### C `anchor075_ratio070`

seed ごと:
- seed 42: init `1926.50`, best `3043.25`, final `1526.00`, late mean `2080.80`
- seed 43: init `2458.25`, best `2751.75`, final `2188.75`, late mean `2117.95`
- seed 44: init `2275.25`, best `3168.00`, final `2450.50`, late mean `2182.10`

3-seed 平均:
- `final_score = 2055.08`
- `cycle20-29 mean = 2126.95`
- `drawdown = 932.58`
- `rank = 2.3542`
- `win = 0.3023`
- `deal_in = 0.4488`

読み:
- score はやはり `0.10` に負ける
- ただし `win_rate` は今回の中で最高
- `clip_fraction`, `ratio_std` も最小

これはかなり興味深く、**high ratio は崩壊ではなく score と他指標のズレ** を作っているように見える。

## 5. `teacher_agreement` の `None` について

今回 driver validation で一度問題になったが、これは実験失敗ではない。

高 `policy_ratio` 条件では一部 cycle で
- `teacher_agreement.num_best_set_samples = 0`
となり、
- `best_set_hit_rate_before/after = None`
になることがある。

具体的には:
- `ratio050`: 3 seeds のうち 1 本で複数 cycle に発生
- `ratio070`: 3 seeds のうち 2 本で `cycle 0` から発生

意味としては、
- baseline samples 自体はある
- ただし best-set 比較可能な baseline samples がその cycle では 0

ということ。

したがって、**high ratio では teacher best-set 系の診断 coverage が少し落ちる** と解釈するのが自然。

## 6. 何が分かったか

今回の実験から、かなり自信を持って言えることは次の 4 点である。

1. **`policy_ratio=0.10` は偶然ではなく、本当に強い帯**
   - `0.30 / 0.50 / 0.70` のどれも score 系で越えられなかった

2. **`policy_ratio` を上げると update はむしろ穏やかになる**
   - `clip_fraction` も `ratio_std` も単調に下がる
   - したがって、今回の悪化は「暴れすぎ」ではない

3. **高 ratio 条件でも signal は壊れていない**
   - `improve > 0`, `same < 0`, `worsen < 0` は維持
   - target/advantage 再定義が再び必要、という証拠はない

4. **distribution mismatch は real だが、actor mix を増やしすぎると baseline の良さを失う**
   - `0.0 -> 0.10` は効いた
   - しかし `0.10 -> 0.30+` は score を下げた

## 7. 実務的な結論

現時点の current best baseline 候補は引き続き:

- `gamma=0.75`
- `gae_lambda=0.3`
- `clip_epsilon=0.15`
- `policy_anchor.coef=0.75`
- `training.rule_mix.policy_ratio=0.10`
- `value_loss_coef=0.25`
- `shanten_scale=0.003`

である。

次に `policy_ratio` を見るなら、大きく上げる方向ではなく、むしろ

- `0.10 / 0.15 / 0.20`

のような **近傍探索** に戻るのが自然。

また、もし high ratio 側をさらに追うなら、次は `policy_ratio` そのものより

- `baseline_sample_weight`

を調整して、**高 actor mix でも baseline rail を残せるか** を見る方が筋が良い。
