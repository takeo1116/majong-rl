# Experiment Report: exp_070

作成日: 2026-03-23  
参照: `experiments/exp_070/runbook.md`, `experiments/exp_070/run_map.json`, `experiments/exp_068/report.md`

## 1. 要約

`exp_068` の current best baseline 候補

- `gamma=0.75`
- `gae_lambda=0.3`
- `clip_epsilon=0.15`
- `policy_anchor.coef=0.75`
- `training.rule_mix.policy_ratio=0.10`

の上に、防御特徴

- `opponent_current_shanten`
- `opponent_tenpai_flag`
- `danger_mask`

を段階的に追加して 1 seed (`42`) で pilot を行った。  
その後、最良だった `C context_plus_danger` について `seed=43,44` を追加で追試した。

結論はかなりはっきりしている。

1. **`danger_mask` が圧倒的に効いた**
2. **`context_only` は弱い**
3. **`context_plus_danger` が最良**
4. 新特徴量は PPO だけでなく、`cycle 0 eval_before` に見える imitation 直後の性能からも効いている
5. 今回の特徴量追加は、少なくとも FullObservation 下では **非常に大きい headroom** を掘り当てた
6. **`C context_plus_danger` は 3 seeds でも非常に強く、current best baseline 候補として扱ってよい**

特に `context_plus_danger` は

- `final avg_score = 6467.0`
- `cycle 20-29 mean = 5948.68`
- `drawdown = 191.0`

となり、参照条件 `REF` を大幅に上回った。

## 2. 実行条件

### 共通固定条件

- seed: `42`
- `gamma=0.75`
- `gae_lambda=0.3`
- `clip_epsilon=0.15`
- `policy_anchor.coef=0.75`
- `training.rule_mix.policy_ratio=0.10`
- `value_loss_coef=0.25`
- `reward.shaping.shanten_delta.scale=0.003`
- `policy_direct_hints.enabled=true`
- `policy_direct_hints.context_gate.enabled=true`

### 比較条件

- `REF` `anchor075_ratio010`
  - `exp_068` の B `seed=42` を参照
  - 特徴量追加なし
- `A` `context_only`
  - `opponent_current_shanten=true`
  - `opponent_tenpai_flag=true`
  - `danger_mask=false`
- `B` `danger_only`
  - `danger_mask=true`
  - `policy_direct_hints.sources += danger_mask_*`
- `C` `context_plus_danger`
  - `opponent_current_shanten=true`
  - `opponent_tenpai_flag=true`
  - `danger_mask=true`
  - `policy_direct_hints.sources += danger_mask_*`

### init の定義

今回の report では、各条件の `init` は

- **`cycle_0.eval_before.avg_score`**

を使う。これは「imitation phase 完了後、PPO cycle 0 開始前の評価」に対応する。

補足:

- `summary.phase_stats.imitation` には `eval_after` が入っていないため、従来の整理と整合するよう `cycle_0.eval_before` を imitation 直後相当の評価として扱う
- 新特徴量は imitation 自体にも効いているため、今回の条件差は PPO のみならず warmstart から反映されている

## 3. 結果一覧

| 条件 | init | best | best cycle | final | cycle 20-29 mean | final-init | late-init | drawdown | final rank | final win | final deal_in | clip frac | ratio std | best_set_after | improve adv | same adv | worsen adv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `REF` | 2095.0 | 3169.5 | 18 | 2187.0 | 2481.10 | +92.0 | +386.10 | 982.5 | 2.4100 | 0.2929 | 0.4566 | 0.1226 | 0.0967 | 0.9127 | +0.1132 | -0.0316 | -0.0321 |
| `A context_only` | 2930.0 | 3664.5 | 3 | 2623.75 | 2343.20 | -306.25 | -586.80 | 1040.75 | 2.3375 | 0.2994 | 0.4534 | 0.1370 | 0.1022 | 0.9132 | +0.0880 | -0.0288 | +0.0092 |
| `B danger_only` | 1849.0 | 5976.0 | 27 | 5509.5 | 5544.93 | +3660.5 | +3695.93 | 466.5 | 2.0825 | 0.3047 | 0.4149 | 0.1136 | 0.0938 | 0.9252 | +0.0973 | -0.0303 | -0.0012 |
| `C context_plus_danger` | 2579.25 | 6658.0 | 24 | 6467.0 | 5948.68 | +3887.75 | +3369.43 | 191.0 | 2.0850 | 0.3144 | 0.4061 | 0.1070 | 0.0894 | 0.9229 | +0.0976 | -0.0276 | -0.0243 |

## 4. 主結論

### 4.1 `danger_mask` が本命

`danger_mask` を入れた条件は、どちらも `REF` を大幅に上回った。

- `REF final = 2187.0`
- `B final = 5509.5`
- `C final = 6467.0`

また、後半 plateau を見る `cycle 20-29 mean` でも

- `REF = 2481.10`
- `B = 5544.93`
- `C = 5948.68`

となり、改善は単点のノイズではない。

したがって今回の pilot では、

**牌別 danger 情報を direct branch に入れることが決定的に効いた**

と見てよい。

### 4.2 `context_only` は弱い

`A context_only` は

- `final = 2623.75`
- `cycle 20-29 mean = 2343.20`

で、`REF final` は上回ったが、`late_mean` は `REF` を下回った。

さらに

- `worsen_adv = +0.0092`

で、今回の 4 条件中唯一 `worsen_adv` が正に転んでいる。

このため、`opponent_current_shanten` / `opponent_tenpai_flag` だけでは

- trunk 文脈としては少し効く
- しかし plateau を押し上げる主因ではない

と解釈するのが自然。

### 4.3 `context + danger` が最良

`C context_plus_danger` は `B danger_only` よりさらに良かった。

- `final`: `6467.0` vs `5509.5`
- `cycle 20-29 mean`: `5948.68` vs `5544.93`
- `drawdown`: `191.0` vs `466.5`
- `win_rate`: `0.3144` vs `0.3047`
- `deal_in_rate`: `0.4061` vs `0.4149`

つまり、

- **主役は danger_mask**
- ただし **相手危険度文脈を足すとさらに押し上がる**

という構図に見える。

今回の model への入れ方

- `opponent_current_shanten` / `opponent_tenpai_flag` → trunk
- `danger_mask` → direct branch

は、少なくとも FullObservation 下ではかなり正しかったと言える。

## 5. 条件ごとの読み

### REF `anchor075_ratio010`

`exp_068` で current best だった参照条件。

- `final = 2187.0`
- `cycle 20-29 mean = 2481.10`
- `drawdown = 982.5`

post-fix 後の baseline としては良いが、新特徴量あり条件と比べると明確に下。

### A `context_only`

- `init = 2930.0` と初期評価は悪くない
- `best = 3664.5` までは上がる
- しかし `final = 2623.75`
- `cycle 20-29 mean = 2343.20`
- `drawdown = 1040.75`

`REF` より少し良い点もあるが、plateau 保持の改善としては弱い。むしろ `final-init` と `late-init` は負で、PPO がうまく噛み合っていない。

### B `danger_only`

- `init = 1849.0`
- `best = 5976.0`
- `final = 5509.5`
- `cycle 20-29 mean = 5544.93`
- `drawdown = 466.5`

score, rank, win, loss-event 系すべてで大幅改善。今回の pilot で一番強いメッセージは、

**danger_mask だけでここまで伸びる**

ということ。

### C `context_plus_danger`

- `init = 2579.25`
- `best = 6658.0`
- `final = 6467.0`
- `cycle 20-29 mean = 5948.68`
- `drawdown = 191.0`

4 条件の中で最良。後半高原の高さと保持の両方で勝っている。

特に `drawdown = 191.0` はかなり小さく、今回の「peak 貼りつき」観点でも最も強い。

## 6. diagnostics の見え方

### 6.1 update はむしろ穏やか

- `REF`: `clip_fraction=0.1226`, `ratio_std=0.0967`
- `B`: `clip_fraction=0.1136`, `ratio_std=0.0938`
- `C`: `clip_fraction=0.1070`, `ratio_std=0.0894`

つまり、新特徴量を入れても PPO が荒れているわけではない。むしろ

- ratio の散りが減り
- clip も少し減り
- 高い plateau に乗っている

という良い方向。

### 6.2 teacher rail も改善

- `REF best_set_after = 0.9127`
- `B best_set_after = 0.9252`
- `C best_set_after = 0.9229`

`danger_mask` 系は、teacher rail を壊すどころかむしろ高く保っている。

### 6.3 `improve / same / worsen` も自然

- `B`: `improve +`, `same -`, `worsen ≈ 0-`
- `C`: `improve +`, `same -`, `worsen -`

少なくとも signal の向きが壊れた感じはない。

一方、`A` だけは

- `worsen_adv = +0.0092`

で少し怪しく、ここも「context_only は弱い」という読みと一致する。

## 7. imitation 側への影響

新特徴量は PPO だけでなく imitation warmstart にも効いている。

| 条件 | teacher_top1 | teacher_best_set | imitation policy_loss | imitation value_loss |
|---|---:|---:|---:|---:|
| `REF` | 0.4325 | 0.9431 | 0.2730 | 0.01293 |
| `A` | 0.4370 | 0.9425 | 0.2714 | 0.01263 |
| `B` | 0.4346 | 0.9522 | 0.2423 | 0.01299 |
| `C` | 0.4409 | 0.9525 | 0.2413 | 0.01259 |

特に `danger_mask` を入れた B/C では

- `teacher_best_set_hit_rate`
- `policy_loss`

が改善している。

したがって今回の改善は

- PPO がたまたまうまくいった

だけではなく、**表現そのものが imitation から学びやすくなっている**可能性が高い。

## 8. 解釈

今回の pilot が示していることはかなり明確。

1. **FullObservation 下では、防御由来の headroom は非常に大きい**
2. その headroom の本体は、相手手牌の構造そのものではなく、**牌別 danger 情報**として直接与えると最も効く
3. `opponent_current_shanten` / `opponent_tenpai_flag` は主役ではないが、`danger_mask` と併用するとさらに良い
4. したがって、以前の
   - 「相手手牌 shape を学ばせるべきか」
   よりも
   - **danger を直接学ばせる方が効率が良い**
   という仮説はかなり支持された

## 9. 注意点

- 本体比較は **1 seed pilot** だが、最良条件 `C context_plus_danger` については後続で `seed=43,44` を追試している
- ただし A/B の多 seed 確認はまだであり、feature ablation の頑健性までは未確定
- これは **FullObservation** 下の結果であり、将来の PartialObservation で同じ改善がそのまま出るとは限らない
- `deal_in_rate` は純粋なロン放銃率ではなく、相手得点を伴う失点イベント率として解釈する

## 10. 次アクション

追試後の現時点では、次はかなり素直にこれでよい。

1. **`C context_plus_danger` を new best baseline 候補として固定する**
2. 余力があれば **`B danger_only` も `seed=43,44` で追試**し、`context` の寄与を追加確認する
3. その後、`danger_mask` あり条件を土台にハイパラ再調整する

現時点の最有力候補は、

- `anchor=0.75`
- `policy_ratio=0.10`
- `danger_mask`
- `opponent_current_shanten`
- `opponent_tenpai_flag`

を有効にした `context_plus_danger` 条件である。

## 11. `C context_plus_danger` 追試 (`seed=43,44`)

追加 batch:

- `runs/20260323_stage1_full_flat_mlp_rule_only_anchor_ppo_baseline_batch_f67d83e2`

対象:

- `C context_plus_danger`
- `seed=43,44`

比較対象:

- `seed=42` の `C context_plus_danger`（本体 pilot）

### 11.1 seed ごとの結果

| seed | init | best | best cycle | final | cycle 20-29 mean | final-init | late-init | drawdown | final rank | final win | final deal_in | clip frac | ratio std | best_set_after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `42` | 2579.25 | 6658.0 | 24 | 6467.0 | 5948.68 | +3887.75 | +3369.43 | 191.0 | 2.0850 | 0.3144 | 0.4061 | 0.1070 | 0.0894 | 0.9229 |
| `43` | 1984.5 | 7250.0 | 25 | 7041.25 | 6700.30 | +5056.75 | +4715.80 | 208.75 | 2.0175 | 0.3281 | 0.3984 | 0.1182 | 0.0950 | 0.9224 |
| `44` | 1549.5 | 6048.0 | 29 | 6048.0 | 5572.60 | +4498.50 | +4023.10 | 0.0 | 2.1050 | 0.3058 | 0.4102 | 0.1098 | 0.0917 | 0.9125 |

### 11.2 3-seed 集計

| 指標 | mean | sd |
|---|---:|---:|
| `init` | 2037.75 | 516.94 |
| `best` | 6652.00 | 601.02 |
| `final` | 6518.75 | 498.64 |
| `cycle 20-29 mean` | 6073.86 | 574.18 |
| `drawdown` | 133.25 | 115.74 |
| `final rank` | 2.0692 | 0.0458 |
| `final win_rate` | 0.3161 | 0.0113 |
| `final deal_in_rate` | 0.4049 | 0.0060 |
| `clip_fraction` | 0.1117 | 0.0059 |
| `ratio_std` | 0.0920 | 0.0028 |
| `best_set_after` | 0.9193 | 0.0059 |

`shanten_diag` の 3-seed 平均:

- `improve_adv = +0.0916`
- `same_adv = -0.0259`
- `worsen_adv = -0.0233`

imitation 側の 3-seed 平均:

- `teacher_top1 = 0.4386`
- `teacher_best_set = 0.9469`
- `imitation policy_loss = 0.2591`
- `imitation value_loss = 0.01261`

### 11.3 解釈

この追試でかなり重要なのは、

1. **3 seeds すべてで `final` が 6k 前後に乗っている**
2. **`cycle 20-29 mean` も 3 seeds すべてで 5.5k を超えている**
3. **drawdown 平均が 133.25 と非常に小さい**

という点である。

したがって、`seed=42` の結果は単発の当たりではなく、
**`context_plus_danger` は高い plateau に安定して貼りつける条件**
と見てよい。

### 11.4 `exp_068` baseline との比較

`exp_068` の best baseline 候補 `anchor075_ratio010` 3-seed 集計は

- `final = 2348.58`
- `cycle 20-29 mean = 2494.93`
- `drawdown = 882.33`

だった。

それに対し、`C context_plus_danger` 3-seed 集計は

- `final = 6518.75` で **+4170.17**
- `cycle 20-29 mean = 6073.86` で **+3578.93**
- `drawdown = 133.25` で **-749.08**

であり、改善幅は非常に大きい。

このため、現時点では

**`context_plus_danger` を Stage 1 FullObservation の current best baseline とみなしてよい**

と判断できる。
