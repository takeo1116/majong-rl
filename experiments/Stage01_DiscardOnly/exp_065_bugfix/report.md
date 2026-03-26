# Experiment Report: exp_065

作成日: 2026-03-22  
対象: bugfix 後の新モデルを前提に、current best PPO 条件 `gamma=0.75, gae_lambda=0.3` を固定し、`value_loss_coef` を sweep した。

条件:
- A: `v010`
- B: `v025`
- C: `v050`

参照:
- `experiments/exp_065/runbook.md`
- `experiments/exp_065/run_map.json`

補足:
- 実行時の `batch_summary.json` や `runs/` 配下のローカル成果物は、VCS に載せない前提のため本 report からは参照しない
- B `v025` は `exp_064` の best 条件 `g075_gae030` をそのまま再利用している

## 1. 結論

今回の sweep から強く言えるのは次の 4 点である。

1. **`value_loss_coef=0.25` が最良**
   - final `avg_score = 2297.0` で 3 条件中トップ
   - `avg_rank` も最良

2. **`value_loss_coef=0.10` は改善しない**
   - `0.25` より score が明確に下がる
   - peak から final への戻り幅も大きい

3. **`value_loss_coef=0.50` はさらに悪い**
   - `0.25` より final `avg_score` が大きく悪化
   - 保持も改善しない

4. **critic 重みそのものは主因ではなさそう**
   - teacher rail は 3 条件でほぼ同じ
   - `clip_fraction` / `ratio_std` / `value_error_mean` も大差ない
   - それでも score 差が出るため、`value_loss_coef` は二次要因寄りと見るのが自然

したがって、current best PPO baseline に対しては

- **`value_loss_coef=0.25` を維持**

が妥当である。

## 2. 実験条件

共通:
- 新モデル (`policy_direct_hints + context_gate`)
- `training.imitation_loss_mode=tie_aware_best_set`
- imitation `1000 matches x 3 chunks`
- PPO `200 matches x 30 cycles`
- `policy_anchor.coef=0.5`
- `clip_epsilon=0.15`
- `gamma=0.75`
- `gae_lambda=0.3`
- `reward.shaping.shanten_delta.scale=0.003`
- `training.rule_mix.policy_ratio=0.0`
- seeds `42,43,44`

差分:
- A: `value_loss_coef=0.10`
- B: `value_loss_coef=0.25`
- C: `value_loss_coef=0.50`

補足:
- B `v025` は `exp_064` best 条件と完全に同一であり、既存 batch を再利用した
- したがって今回の差は、基本的に **PPO 中の `value_loss_coef` 差** と見てよい

## 3. 実行結果

| 条件 | success |
|---|---:|
| A `v010` | `3/3` |
| B `v025` | `3/3` |
| C `v050` | `3/3` |

注記:
- 3 条件とも正常完走
- B `v025` のみ再利用、A/C は新規実行

## 4. 最終結果

| 条件 | value_loss_coef | final avg_rank | final avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|---:|
| A `v010` | `0.10` | `2.3458` | `2002.0` | `0.2914` | `0.4540` |
| B `v025` | `0.25` | `2.3317` | `2297.0` | `0.2928` | `0.4497` |
| C `v050` | `0.50` | `2.3558` | `1838.9` | `0.2906` | `0.4552` |

所見:
- **B `v025` が総合最良**
- A `v010` は悪くないが、B には明確に届かない
- C `v050` は A よりさらに悪く、critic を強める方向は少なくとも今回の条件では外れ

## 5. Peak と保持

| 条件 | best cycle mean | best avg_score mean | final avg_score | best→final drawdown |
|---|---:|---:|---:|---:|
| A `v010` | `14` | `2739.8` | `2002.0` | `737.8` |
| B `v025` | `19` | `2599.3` | `2297.0` | `302.3` |
| C `v050` | `18` | `2475.6` | `1838.9` | `636.7` |

ここが今回かなり重要である。

- **B `v025` は peak 自体が極端に高いわけではない**
- しかし **peak を一番うまく保持できている**
- A/C は peak 後に 600〜700 点規模で戻す
- B だけは戻り幅が `302` とかなり小さい

したがって、今回の差は

**「どれだけ高く跳ねるか」より「どれだけ失速せずに保てるか」**

に表れている。

## 6. learner 診断

final learner 診断の代表値:

| 条件 | best_set_hit_after | action_match_after | clip_fraction | ratio_std | value_error_mean |
|---|---:|---:|---:|---:|---:|
| A `v010` | `0.9107` | `0.4064` | `0.1853` | `0.1219` | `0.00324` |
| B `v025` | `0.9107` | `0.4071` | `0.1815` | `0.1207` | `0.00335` |
| C `v050` | `0.9104` | `0.4065` | `0.1787` | `0.1198` | `0.00318` |

ここから読み取れること:

1. **teacher safety rail は 3 条件でほぼ同じ**
   - `best_set_hit_after` はすべて `0.91` 前後
   - したがって今回の差は、teacher rail の維持率そのものではない

2. **update の荒さにも大差がない**
   - `clip_fraction` / `ratio_std` は小差に留まる
   - `0.50` が特別に不安定という形ではない

3. **value error も大差がない**
   - `value_error_mean` は 3 条件ともほぼ同じ帯
   - 単純に critic が壊れた/改善した、という見え方ではない

このため、今回の結果は

**「critic 重みを変えれば PPO の本質問題が解ける」ではない**

と読むのが自然である。

## 7. shanten advantage の癖

代表 seed の final `shanten_diag.advantage.mean`:

| 条件 | same | improve | worsen |
|---|---:|---:|---:|
| A `v010` | `+0.0406` | `-0.1203` | `-0.0355` |
| B `v025` | `+0.0413` | `-0.1206` | `-0.0407` |
| C `v050` | `+0.0409` | `-0.1201` | `-0.0387` |

今回も以前からの違和感は残っている。

- `same` は一貫して正
- `improve` は一貫して強く負
- `worsen` は負だが、`improve` よりは悪くない

しかもこの形は、`value_loss_coef` を変えてもほぼ動いていない。

したがって、今回の sweep は

**「critic の重みでは、この advantage の順位づけの違和感はほとんど変わらない」**

という意味でも重要である。

## 8. 解釈

今回の結果を一番自然に言い換えると、こうなる。

### 1. `value_loss_coef=0.25` は現時点で十分妥当
- `0.10` に下げても改善しない
- `0.50` に上げると悪化する
- current best horizon 条件では、`0.25` 付近が一番バランスが良い

### 2. critic / advantage quality の問題は、critic の loss 重みだけでは切れない
- value error は大差ない
- teacher rail も大差ない
- それでも final score は変わる

### 3. 本題はまだ reward / target / distribution 側に残っている
- `improve < worsen` の違和感はそのまま
- したがって次に見るべきは
  - `policy_ratio`
  - あるいは advantage の作り方
の方が自然である

## 9. 暫定判断

現時点の暫定判断は次のとおりである。

- **採用**: `value_loss_coef=0.25`
- **保留**: `value_loss_coef=0.10`
  - 明確に悪いわけではないが、採用理由は弱い
- **見送り**: `value_loss_coef=0.50`

## 10. 次アクション

1. current best PPO baseline は
   - `gamma=0.75`
   - `gae_lambda=0.3`
   - `value_loss_coef=0.25`
   を採用候補として扱う
2. 次の切り分けは `policy_ratio` sweep を優先する
3. それでも説明しきれなければ、reward / advantage quality 側を見に行く
