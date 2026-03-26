# Experiment Report: exp_061

作成日: 2026-03-18  
対象: bugfix 後の新モデルを前提に、`imitation 1000 x 3 chunks` を warmstart として、`rule-only / actor` と `policy_anchor on/off` の 2x2 を比較した。

- A: `rule_only_no_anchor`
- B: `rule_only_anchor`
- C: `actor_no_anchor`
- D: `actor_anchor`

参照:
- `experiments/exp_061/runbook.md`
- `experiments/exp_061/run_map.json`

補足:
- 実行時の `batch_summary.json` や `runs/` 配下のローカル成果物は、VCS に載せない前提のため本 report からは参照しない

## 1. 結論

一番自然な読みは、**主因は policy drift で、state distribution mismatch も二次的に効いている** である。

- `rule-only` で anchor を入れない A は、前半で大きく伸びるが後半で強く崩れた。
- 同じ `rule-only` でも anchor を入れた B は、30 cycle 後も高い性能を維持した。
- anchor なし同士で比べると、C（actor）は A（rule-only）より明確に良く、actor 分布を見せる効果も確認できた。
- ただし最終性能は **B > D > C >> A** で、最良は `rule_only_anchor` だった。

したがって、

1. drift はかなり本丸  
2. rule-only 分布だけでも anchor があれば十分戦える  
3. distribution mismatch も無視できないが、第一原因ではない  

という整理がもっともしっくり来る。

## 2. 実験条件

共通:
- 新モデル (`policy_direct_hints + context_gate`)
- `training.imitation_loss_mode=tie_aware_best_set`
- imitation `1000 matches x 3 chunks`
- PPO `200 matches x 30 cycles`
- `gamma=0.50`
- `gae_lambda=0.0`
- `reward.shaping.shanten_delta.scale=0.003`
- seeds `42,43,44`

差分:
- A: rule-only, anchor off
- B: rule-only, anchor on
- C: actor PPO, anchor off
- D: actor PPO, anchor on

## 3. 最終結果

| 条件 | final avg_rank | final avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A `rule_only_no_anchor` | `2.5042` | `-113.2` | `0.2678` | `0.4638` |
| B `rule_only_anchor` | `2.3583` | `1943.0` | `0.2914` | `0.4539` |
| C `actor_no_anchor` | `2.4100` | `1626.6` | `0.2762` | `0.4607` |
| D `actor_anchor` | `2.3692` | `1883.8` | `0.3007` | `0.4543` |

最終順位は **B > D > C >> A**。

- A だけが最終的に `avg_score` ほぼ 0 まで落ちた。
- B と D はどちらも高く維持。
- C も A よりかなり良いが、anchor 条件には届かない。

## 4. Warmstart の確認

imitation 指標は、A/B と C/D でそれぞれほぼ同じだった。

### A/B 共通の imitation
- `teacher_top1_match_rate ≈ 0.4334`
- `teacher_best_set_hit_rate ≈ 0.9436`
- `value_loss ≈ 0.00633`

### C/D 共通の imitation
- `teacher_top1_match_rate ≈ 0.4332`
- `teacher_best_set_hit_rate ≈ 0.9438`
- `value_loss ≈ 0.00665`

補足:
- `cycle0.eval_before` は anchor on/off でやや差があった。
- ただし imitation 指標はほぼ同じなので、今回の解釈は主に **30 cycle 後の着地と崩れ方** を重視する。

## 5. Peak と崩れ方

| 条件 | init avg_score | best cycle | best avg_score | final avg_score | best→final |
|---|---:|---:|---:|---:|---:|
| A | `60.8` | `3` | `2687.7` | `-113.2` | `-2800.8` |
| B | `1916.1` | `8` | `2803.6` | `1943.0` | `-860.6` |
| C | `1474.7` | `2` | `2537.7` | `1626.6` | `-911.1` |
| D | `2056.6` | `6` | `2682.3` | `1883.8` | `-798.5` |

ここがかなり重要だった。

### A: rule-only / no-anchor
- 立ち上がり自体は最も派手
- しかしその後の戻りが圧倒的に大きい
- `best→final = -2800.8`

これは、以前の「前半は積めるが後半で壊れる」という観測を、3 seeds で再確認した形になっている。

### B: rule-only / anchor
- peak は高い
- しかも final でもかなり保つ
- `best→final = -860.6`

anchor を入れるだけで、rule-only PPO の崩れ方がかなり変わった。

### C: actor / no-anchor
- A より明確に良い
- actor 分布を見せるだけでも、anchor なし rule-only より崩れにくい

### D: actor / anchor
- 安定性は高い
- ただし B を超えない

## 6. A/B 比較: anchor の効果

rule-only 条件の比較では、anchor の効果が非常に明確だった。

### final metrics
- avg_score: `-113.2 -> 1943.0`
- avg_rank: `2.5042 -> 2.3583`
- deal_in_rate: `0.4638 -> 0.4539`

### teacher 診断 (final cycle)
A:
- `action_match before/after ≈ 0.4318 -> 0.4324`
- `best_set_hit before/after ≈ 0.8404 -> 0.8349`

B:
- `action_match before/after ≈ 0.4055 -> 0.4070`
- `best_set_hit before/after ≈ 0.9103 -> 0.9096`

ポイント:
- A は後半で `best_set_hit` をかなり削っている
- B は `best_set_hit` を高く保ったまま回せている

これは、**rule-only PPO の問題は「改善信号が無い」より「imitation から離れすぎること」** と読むのが自然である。

## 7. A/C 比較: actor 分布の効果

anchor なし同士で比べると、C は A よりかなり良い。

### final metrics
- avg_score: `-113.2 -> 1626.6`
- avg_rank: `2.5042 -> 2.4100`
- deal_in_rate: `0.4638 -> 0.4607`

つまり、**actor の状態分布を見るだけでも改善する**。

この比較からは、distribution mismatch も確かに存在すると言える。

ただし、C だけで B には届かない。
そのため、

- distribution mismatch はある  
- しかし第一原因はそれだけではない  

という読みになる。

## 8. B/D 比較: actor + anchor は最良ではなかった

もし「actor 分布が本丸」であれば、D が最良になるはずだった。

実際には:
- B final avg_score: `1943.0`
- D final avg_score: `1883.8`

差は大きくはないが、**D が B を上回る形にはならなかった**。

この結果は、
- actor 分布は役に立つ
- でも anchor で drift を抑えた rule-only PPO が、少なくとも今回の条件では十分強い

ということを示している。

## 9. learner 診断

final cycle の代表値:

| 条件 | clip_fraction | ratio_std | value_error_mean |
|---|---:|---:|---:|
| A | `0.267` | `0.164` | `0.00348` |
| B | `0.195` | `0.127` | `0.00249` |
| C | `0.090` | `0.092` | `0.00443` |
| D | `0.079` | `0.082` | `0.00249` |

読み方:
- anchor は A→B, C→D の両方で `clip_fraction` / `ratio_std` を下げている
- actor 条件は rule-only 条件より全体に `ratio_std` が小さい

ただし、数値が一番穏やかな D が B を超えたわけではない。
なので、

- 単純な数値安定性だけでは説明しきれない  
- それでも drift 指標は anchor 効果と整合している  

という位置づけになる。

## 10. 解釈

今回の 2x2 で一番自然な暫定結論は次の通り。

### 1. 主因は policy drift
- A が大きく崩れ、B が大きく改善した
- rule-only 条件での anchor 効果が最も大きい

### 2. state distribution mismatch もある
- A より C がかなり良い
- actor データを見せる価値はある

### 3. ただし distribution mismatch は第一原因ではない
- D が B を明確に超えない
- anchor あり rule-only がすでにかなり強い

したがって今の優先順位は、

1. **drift をどう抑えるか**  
2. そのうえで **rule-only データの中で良い行動/悪い行動をどう切るか**  

になる。

## 11. 次にやること

今回の結果を踏まえると、次の候補はかなり絞られる。

1. `policy_anchor` を前提にして、rule-only PPO をしばらく主診断系にする  
2. その上で、**rule の打牌の中でも良いものだけを強く学ぶ** 方向を試す  
   - 例: `advantage > 0` の baseline sample のみ使う
   - 例: baseline sample を `relu(advantage)` で重み付けする
3. actor mix は補助線として残すが、第一候補にはしない

今の段階では、**「distribution を変える」より先に、「rule-only PPO の drift と weighting を詰める」** のが自然である。
