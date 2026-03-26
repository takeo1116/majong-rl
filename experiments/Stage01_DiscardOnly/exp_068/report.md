# Experiment Report: exp_068

作成日: 2026-03-22  
参照: `experiments/exp_068/runbook.md`, `experiments/exp_068/run_map.json`

## 1. 要約

`exp_067` で効いた単発ノブ

- `anchor=0.75`
- `clip=0.10`
- `policy_ratio=0.10`

を組み合わせ、corrected semantics 上で **final が imitation 直後を超えられるか** を 1 seed (`42`) で確認した。

結論はかなりはっきりしている。

1. **最良は B `anchor075_ratio010`**
2. **B だけが final で imitation 直後を上回った**
3. `clip=0.10` は rule-only では有効でも、`policy_ratio=0.10` と同時に入れると悪化した
4. corrected semantics 後の問題は「改善不能」ではなく「保持失敗」だったが、今回 **少なくとも 1 条件で保持に成功した**

今回の一番大きい前進は、

**「post-fix でも final > init は達成可能」**

だと分かったこと。

## 2. 実行条件

- seed: `42`
- 共通 baseline:
  - 新モデル (`policy_direct_hints + context_gate`)
  - `gamma=0.75`
  - `gae_lambda=0.3`
  - `value_loss_coef=0.25`
  - `shanten_scale=0.003`
- 比較条件:
  - `REF g075_gae030`
  - A `anchor075_clip010`
  - B `anchor075_ratio010`
  - C `clip010_ratio010`
  - D `anchor075_clip010_ratio010`

補足:

- 今回は仮想環境を有効化し忘れて実行したため、driver log に PyTorch の `TypedStorage is deprecated` warning が出ている
- ただし全条件 `completed` で、summary の整合性にも問題はなかったため、結果はそのまま採用する

## 3. 重要な前提

今回の 5 run は、**imitation 側の指標が完全に一致**していた。

- `teacher_top1_match_rate = 0.432466`
- `teacher_best_set_hit_rate = 0.943054`
- `policy_loss = 0.272973`
- `value_loss = 0.012926`

したがって、条件差は本質的に **PPO 側の差** と見てよい。

一方で `eval_before` の score は `1492.25` から `2715.25` までかなりばらついた。  
imitation 自体は同一なので、これは主に **100-match rotation eval の単発ノイズ** と考えるのが自然。

そのため今回の解釈では、

- `final - init`
- `best -> final drawdown`
- final の learner diagnostics

を重視し、`init` の絶対値だけで条件の優劣を断定しない。

## 4. 結果一覧

| 条件 | init score | best score | best cycle | final score | final-init | drawdown | final rank | final win | final deal_in | clip frac | ratio std | best_set_after | policy samp | baseline samp | improve adv | same adv | worsen adv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| REF `g075_gae030` | 1492.25 | 2808.00 | 14 | 1245.00 | -247.25 | 1563.00 | 2.4300 | 0.2775 | 0.4598 | 0.1950 | 0.1237 | 0.9016 | 0 | 88212 | +0.1122 | -0.0324 | -0.0215 |
| A `anchor075_clip010` | 2063.50 | 3172.75 | 13 | 1534.25 | -529.25 | 1638.50 | 2.4050 | 0.2873 | 0.4602 | 0.2324 | 0.0961 | 0.9128 | 0 | 88212 | +0.1110 | -0.0330 | -0.0139 |
| B `anchor075_ratio010` | 2029.75 | 3169.50 | 18 | 2187.00 | +157.25 | 982.50 | 2.4100 | 0.2929 | 0.4566 | 0.1226 | 0.0967 | 0.9127 | 8136 | 80232 | +0.1132 | -0.0316 | -0.0321 |
| C `clip010_ratio010` | 1845.00 | 2833.50 | 11 | 1576.50 | -268.50 | 1257.00 | 2.3850 | 0.2898 | 0.4620 | 0.2629 | 0.1087 | 0.9057 | 7940 | 79758 | +0.0993 | -0.0347 | +0.0301 |
| D `anchor075_clip010_ratio010` | 2715.25 | 3108.75 | 7 | 1554.00 | -1161.25 | 1554.75 | 2.3825 | 0.2872 | 0.4590 | 0.2046 | 0.0880 | 0.9160 | 7848 | 79996 | +0.1125 | -0.0320 | -0.0263 |

## 5. 主結論

### 5.1 B `anchor075_ratio010` が最良

B は今回の 5 条件の中で:

- **唯一 `final > init`**
- final `avg_score = 2187.0` で最良
- `drawdown = 982.5` で最小
- `clip_fraction = 0.1226`, `ratio_std = 0.0967` と update も穏やか
- `best_set_after = 0.9127` と teacher rail も高い
- `worsen_adv = -0.0321` と signal も自然

だった。

今回の sweep で一番強く言えるのは、

**post-fix の本命は `anchor=0.75` + `policy_ratio=0.10`**

であるということ。

### 5.2 `anchor=0.75` は効くが、`clip=0.10` を足すと mixed 条件では逆効果

単発探索 `exp_067` では:

- `anchor=0.75`
- `clip=0.10`

の両方が効いた。

しかし今回の組み合わせでは:

- A `anchor075_clip010` は final `1534.25`
- D `anchor075_clip010_ratio010` は final `1554.0`

で、どちらも B に大きく負けた。

特に D は:

- `best_set_after = 0.9160` で最高
- `ratio_std = 0.0880` で最小

にもかかわらず final は低い。

これは、

**「teacher に近い」「update が小さい」だけでは足りず、mixed 条件では `clip=0.10` が過度に保守的になっている**

可能性を示している。

言い換えると、

- `clip=0.10` は rule-only では効いた
- しかし `policy_ratio=0.10` を入れた状態では、**むしろ update を縛りすぎる**

という読みが自然。

### 5.3 `policy_ratio=0.10` は本物に見える

`policy_ratio=0.10` を含む条件は:

- B `anchor075_ratio010` が最良
- C `clip010_ratio010` も REF よりは上

だった。

また mixed 条件では実際に:

- policy samples `~7.8k - 8.1k`
- baseline samples `~80k`

が入っており、単なる設定ミスではない。

したがって、

**distribution mismatch は corrected semantics 後も依然として重要**

と見てよい。

## 6. 条件ごとの読み

### REF `g075_gae030`

- baseline の再実行
- `best = 2808.0` までは上がるが、final `1245.0`
- `drawdown = 1563.0`

corrected semantics 後の baseline 単体では、依然として **peak は作れるが保持できない**。

### A `anchor075_clip010`

- `anchor=0.75` と `clip=0.10` を rule-only で併用
- `best = 3172.75` と peak は高い
- しかし final `1534.25`
- `drawdown = 1638.5`

rule-only では「良いピーク」を作れるが、**保持にはつながっていない**。

### B `anchor075_ratio010`

- 今回の最良条件
- `best = 3169.5`, final `2187.0`
- `final - init = +157.25`
- `drawdown = 982.5`

`anchor=0.75` で drift を抑えつつ、`policy_ratio=0.10` で actor 自身の分布を少量混ぜるのが効いた、と解釈できる。

### C `clip010_ratio010`

- `clip=0.10` と `policy_ratio=0.10`
- final `1576.5`
- `worsen_adv = +0.0301`
- `value_error_mean = 0.0171` で今回最大

ここはかなり重要で、**強い anchor なしで mixed + tight clip にすると、signal の質が少し崩れる**気配がある。

単発探索では `clip=0.10` は効いたが、mixed 条件では **anchor を伴わないと不安定** なのかもしれない。

### D `anchor075_clip010_ratio010`

- 当たりノブ全部載せ
- `best = 3108.75`, final `1554.0`
- `best_set_after = 0.9160` で最高
- `ratio_std = 0.0880` で最小

数値だけ見ると「安定」していそうだが、結果は悪い。

ここから言えるのは、

**`anchor=0.75` と `policy_ratio=0.10` は効くが、そこへ `clip=0.10` を足すと current setup では過拘束になる**

ということ。

## 7. `improve / same / worsen` の見え方

今回の 5 条件すべてで、少なくとも大枠としては

- `improve_adv > 0`
- `same_adv < 0`

が維持された。

これは CQ-0210 / CQ-0211 修正後の signal が引き続き自然であることを示している。

ただし C だけは

- `worsen_adv = +0.0301`

となっており、ここは注意点。

したがって今回は、

- **signal 定義がまた壊れた**

というより、

- **mixed 条件での制約の掛け方によって、局所的に signal の質が悪くなる**

と読む方が自然。

## 8. `deal_in_rate` の扱い

この指標は名前に反して、**純粋なロン放銃率ではない**。

現在の evaluator 実装では、局終了時に

- 自分が失点し
- 他家の誰かが増点している

場合をカウントしているため、

- 他家への放銃
- 他家ツモでの失点
- 流局ノーテン罰符

が混ざる。

したがって本 report では、`deal_in_rate` を厳密な「放銃率」ではなく、

**相手得点を伴う失点イベント率**

として読む。

## 9. 解釈

今回の結果は、かなり前向きに読んでよい。

### 分かったこと

1. corrected semantics 後でも **final > init は達成可能**
2. そのため、今の問題は「target / advantage の再定義がまだ必要」よりも  
   **hyperparameter / training regime の再調整** に近い
3. 特に効いたのは
   - 強い anchor
   - 少量 actor mix
4. 一方で `clip=0.10` は、rule-only では有効でも mixed 条件では逆効果になりうる

### まだ残っている問題

- 1 seed のため `eval_before` のノイズが大きい
- `anchor=0.75 + policy_ratio=0.10` が本当に安定して勝つかは未確認
- `clip` の最適値が
  - rule-only では `0.10`
  - mixed では `0.15`
 なのかもしれず、ここは切り分けが必要

## 10. 次の実験候補

この section は **初回 1-seed 結果時点** の候補整理。  
第一候補だった B の 3-seed 再確認は、後述の **section 12 で実施済み**。

今回の結果だけを見ると、次の優先度はかなり明確。

### 第一候補

**B `anchor075_ratio010` を 3 seeds で再確認**

理由:

- 今回唯一 `final > init`
- 絶対値でも final 最良
- drawdown も最小
- signal / teacher rail / update すべてバランスが良い

### 第二候補

**B を中心に `policy_ratio` だけ軽く振る**

候補:

- `anchor=0.75`, `policy_ratio=0.05`
- `anchor=0.75`, `policy_ratio=0.10`
- `anchor=0.75`, `policy_ratio=0.15`

`clip` は `0.15` のままにして、まず actor mix の最適量を見た方が筋が良い。

### 今は後回しでよいもの

- `clip=0.10` を mixed 条件にさらに足す探索
- `gamma=0.90` の再検証
- shanten shaping の弱化

今回の evidence では、これらは優先度が低い。

## 11. 最終まとめ

`exp_068` の主結論は一文で言うとこうなる。

**post-fix の当たりは `anchor=0.75` と `policy_ratio=0.10` の組み合わせであり、`clip=0.10` は rule-only では効いても mixed 条件では足を引っ張る可能性が高い。**

そして何より重要なのは、

**corrected semantics 後でも PPO が final で imitation を超えうることが、今回初めて明確に確認できた**

という点である。

## 12. 追試: B `anchor075_ratio010` を 3 seeds で再確認

作成後、B 条件だけを追加で

- seed `43`
- seed `44`

で再実行し、初回の seed `42` と合わせて 3 seeds で見直した。  
runbook を切り直さず、同一条件の追加実行として扱う。

### 12.1 条件

- `gamma=0.75`
- `gae_lambda=0.3`
- `clip_epsilon=0.15`
- `policy_anchor.coef=0.75`
- `training.rule_mix.policy_ratio=0.10`
- `value_loss_coef=0.25`
- `shanten_scale=0.003`

### 12.2 3 seeds の結果

| seed | init score | best score | best cycle | final score | final-init | drawdown | final rank | final win | final deal_in | late-cycle mean (10-25) | missing eval cycles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 2029.75 | 3169.50 | 18 | 2187.00 | +157.25 | 982.50 | 2.4100 | 0.2929 | 0.4566 | 2681.97 | なし |
| 43 | 1827.00 | 2821.00 | 21 | 2513.25 | +686.25 | 307.75 | 2.3325 | 0.2960 | 0.4457 | 2125.47 | なし |
| 44 | 3183.00 | 3702.25 | 2 | 2345.50 | -837.50 | 1356.75 | 2.3250 | 0.2912 | 0.4524 | 2979.25 | cycle 25 |

### 12.3 集計

- `init_score`
  - mean `2346.58`
  - sd `597.20`
- `best_score`
  - mean `3230.92`
  - sd `362.38`
- `final_score`
  - mean `2348.58`
  - sd `133.21`
- `final - init`
  - mean `+2.00`
  - sd `631.68`
- `drawdown`
  - mean `882.33`
  - sd `434.07`
- `final_rank`
  - mean `2.3558`
- `final_win_rate`
  - mean `0.29335`
- `final_deal_in_rate`
  - mean `0.45157`

final learner diagnostics の平均:

- `clip_fraction = 0.1444`
- `ratio_std = 0.1052`
- `value_error_mean = 0.00061`
- `best_set_after = 0.9130`
- `improve_adv = +0.1014`
- `same_adv = -0.0295`
- `worsen_adv = -0.0193`

### 12.4 何が分かったか

#### 1. `final - init` はかなりノイジー

3 seeds で見ると、

- seed 42: `+157`
- seed 43: `+686`
- seed 44: `-838`

と大きく振れている。  
一方で `init_score` の標準偏差は `597` と非常に大きい。

つまり、以前から疑っていた通り、

**100-match eval の `init` はかなりノイジー**

であり、単発の `final - init` をそのまま真に受けるべきではない。

#### 2. それでも B 自体はかなり有望

重要なのは、`final_score` の標準偏差が `133` と小さいこと。  
つまり 3 seeds で:

- `2187.0`
- `2513.25`
- `2345.5`

に収束しており、**final の絶対値はかなり揃っている**。

これは B が

- 偶然 1 本だけ当たった

というより、

- **だいたい `2.2k - 2.5k` 付近に着地しやすい条件**

であることを示す。

#### 3. 「高い plateau に戻る」仮説はかなり支持される

各 seed の上位 cycle を見ると:

- seed 42
  - cycle `18`, `23`, `25`, `10`, `5`
- seed 43
  - cycle `21`, `19`, `20`, `29`, `12`
- seed 44
  - cycle `2`, `14`, `20`, `21`, `1`

で高得点が出ている。

特に

- seed 42 は後半に何度も `3.1k` 近辺へ戻る
- seed 43 は final 自体が上位 5 cycle に入る
- seed 44 は early peak が強いが、中後半にも `3.3k - 3.4k` 帯がある

ので、これは

**単峰で崩壊しているより、「高い plateau に滞在していて評価ノイズで上下して見える」**

という説明にかなり整合する。

`cycle 10-25` の平均 score も

- `2681.97`
- `2125.47`
- `2979.25`

で、seed 44 は final が低く見えても late-phase 全体では高い。

#### 4. signal は 3 seeds でも自然

3 seeds 平均でも

- `improve_adv > 0`
- `same_adv < 0`
- `worsen_adv < 0`

を維持している。  
したがって B の良さは、壊れた signal の偶然ではなく、**corrected semantics 上のまともな learning signal で出ている**と見てよい。

#### 5. final の rank / win / deal_in はまだ完全には良くない

平均で見ると:

- `final_rank = 2.3558`
- `final_win_rate = 0.29335`
- `final_deal_in_rate = 0.45157`

で、ここはまだ「圧倒的改善」とは言いにくい。

特に seed 44 では

- init `3183.0`
- final `2345.5`

と `final - init` は大きく負になっている。  
ただしこれは init 自体がかなり高く、しかも late-cycle mean は `2979.25` なので、**final 1 点だけで seed 44 を悲観しすぎるのも危険**。

### 12.5 補足

seed 44 では `cycle 25` の `eval` が `summary` 内で欠けていた。  
ただし

- top-level final eval
- cycle 29 eval
- learner diagnostics

は正常に揃っていたため、全体の解釈は可能。  
上の集計では `cycle 25` だけ late-phase 平均から除外している。

### 12.6 追試後の結論

初回時点では

- 「B だけが `final > init`」

という読みだったが、3 seeds で見ると、より正確な理解はこうなる。

1. **B は 3 seeds で見ても最も有望**
2. ただし `final - init` は評価ノイズが大きく、そのまま主指標にしにくい
3. むしろ
   - `final_score` の絶対値
   - `drawdown`
   - `cycle 10-25` の平均
   - final diagnostics の安定性
   で見るべき
4. その観点では、B は
   - final が揃う
   - signal が自然
   - teacher rail も高い
   - mixed PPO として一番バランスが良い

したがって、**post-fix の current best baseline 候補は引き続き B `anchor075_ratio010`** と判断してよい。

### 12.7 `cycle 20-29` 平均で見た imitation 比較

`final` 1 点は 100-match eval ノイズの影響を受けやすいので、B 条件については  
**後半 plateau の代表値**として `cycle 20-29` の `eval.avg_score` 平均でも比較した。

比較対象の imitation 直後 score は、各 run の top-level `phase_stats.eval_before.avg_score` を用いた。

| seed | imitation直後 | cycle 20-29 平均 | 差 |
|---|---:|---:|---:|
| 42 | 2029.75 | 2481.10 | +451.35 |
| 43 | 1827.00 | 2074.63 | +247.63 |
| 44 | 3183.00 | 2929.06 | -253.94 |

集計:

- imitation直後 mean: `2346.58`
- `cycle 20-29` mean: `2494.93`
- 差: **`+148.34`**

補足:

- seed 44 は `cycle 25` の eval が欠けているため、`cycle 20-29` のうち **9 点平均**
- seed 42 / 43 は **10 点平均**

この見方では、B は

- 3 seeds 中 2 本で明確に imitation を上回り
- 3 seeds 平均でも imitation を上回る

したがって、`final - init` の単点比較よりも  
**「B は後半 plateau では平均的に imitation より改善している」**  
と解釈する方が実態に近い。
