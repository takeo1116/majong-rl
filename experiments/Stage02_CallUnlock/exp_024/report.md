# Experiment Report: exp_024

作成日: 2026-05-02  
Stage: `Stage02_CallUnlock`

参照:

- `experiments/Stage02_CallUnlock/exp_024/runbook.md`
- `experiments/Stage02_CallUnlock/exp_024/run_map.json`
- `experiments/Stage02_CallUnlock/exp_023/report.md`
- `experiments/Stage02_CallUnlock/exp_019/report.md`
- `experiments/Stage02_CallUnlock/exp_017/report.md`

## 1. 要約

`exp_024` は、以前 `exp_016`〜`exp_019` で不採用にした `tile_presence_flags`、通称 `yakuflags` を、`exp_023` で安定化した `separated policy-only PPO` 基準の上で再評価した実験である。

今回の新規条件は、過去に最も見込みがあった `on_wide` 相当である。

```text
feature_encoder.tile_presence_flags = true
model.value_hidden_dims = [256,128]
model.semantic_aux.tile_presence_flags_semantic_only = false
training.rule_mix_learner.ppo_mode = "separated"
```

結論:

- 3 seed すべて正常完了
- `exp_023` より順位系指標が小幅に改善
- `final avg_rank`: `2.167 -> 2.142`
- `tail10 avg_rank`: `2.199 -> 2.169`
- `best10 avg_rank`: `2.182 -> 2.130`
- PPO diagnostics は悪化せず、`ratio_max` / `clip_fraction` はむしろ改善
- yaku semantic は全体的に改善
  - `yaku micro F1`: `0.3931 -> 0.4087`
  - `yaku macro F1`: `0.0753 -> 0.0894`
  - `Tanyao mean_p`: `0.2683 -> 0.3023`
  - `Pinfu mean_p`: `0.2054 -> 0.2746`
- 一方で terminal accuracy と `win_called top3` は悪化
- 今回の差分は `tile_presence_flags=true` と `value_hidden_dims=[256,128]` の同時変更なので、特徴量単独効果とは断定しない

実務判断として、`exp_024 on_wide separated` は **Stage2a の新 baseline 候補に昇格**させてよい。

## 2. 背景

### 2.1 以前の yakuflags 判断

`CQ-0270` では、役の学習補助として以下の 6 つの self tile-presence flags を追加した。

| feature | 意味 |
|---|---|
| `self_has_honor` | 字牌を 1 枚以上持つ |
| `self_has_terminal` | 1/9 牌を 1 枚以上持つ |
| `self_has_simple` | 2-8 数牌を 1 枚以上持つ |
| `self_has_man` | 萬子を 1 枚以上持つ |
| `self_has_pin` | 筒子を 1 枚以上持つ |
| `self_has_sou` | 索子を 1 枚以上持つ |

狙いは、`Tanyao` のような「么九牌・字牌がない」条件を MLP に読ませやすくすることだった。

過去実験では次のように整理していた。

- `exp_016`: shared input に常時追加すると policy が悪化
- `exp_017`: `value_hidden_dims=[256,128]` に広げると `on_wide` はかなり回復し、`Tanyao` の弱い確率信号は改善
- `exp_018`: semantic-only routing は悪化
- `exp_019`: `on_wide` を 3 seed 比較しても practical baseline には勝てず、不採用

ただし、この判断は `exp_023` 以前、つまり mixed PPO 問題が残っていた時期の判断だった。

### 2.2 exp_023 による前提更新

`exp_023` では、baseline actor sample を PPO ratio 付き policy loss に混ぜることが long-run collapse の主因だったと判断した。

そこで `training.rule_mix_learner.ppo_mode="separated"` にして、PPO policy update を `actor_type="policy"` sample のみに限定したところ、3 seed で collapse が解消した。

このため、`yakuflags` は現行 separated PPO 環境で再検証する価値があると判断した。

## 3. 実験条件

### 3.1 reference: exp_023 separated baseline

`exp_023` の 3 seed 結果を再利用する。

条件:

- `feature_encoder.tile_presence_flags = false`
- `model.value_hidden_dims = [128,64]`
- `model.semantic_aux.tile_presence_flags_semantic_only = false`
- `training.rule_mix_learner.ppo_mode = "separated"`
- no-anchor
- `lr = 0.0001`
- `clip_epsilon = 0.15`
- 60 cycles

### 3.2 new: exp_024 on_wide separated

新規 3 seed 実行。

共通条件:

- config: `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- seeds: `42`, `43`, `44`
- `training.multi_cycle.num_cycles = 60`
- `training.multi_cycle.selfplay_matches_per_cycle = 200`
- `training.policy_anchor.enabled = false`
- `training.policy_anchor.coef = 0.0`
- `training.lr = 0.0001`
- `training.clip_epsilon = 0.15`
- `training.entropy_coef = 0.0`
- `training.value_loss_coef = 0.125`
- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix.save_baseline_actions = true`
- `training.rule_mix_learner.ppo_mode = "separated"`
- `training.rule_mix_learner.baseline_imitation_epochs = 0`
- `training.rule_mix_learner.policy_ppo_epochs = 1`
- `training.rule_mix_learner.allow_mixed_offpolicy_baseline = false`
- `model.semantic_aux.enabled = true`
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.terminal_loss_coef = 0.1`
- `training.semantic_aux.yaku_loss_coef = 0.05`
- `feature_encoder.tile_presence_flags = true`
- `model.value_hidden_dims = [256,128]`
- `model.semantic_aux.tile_presence_flags_semantic_only = false`

実行対応は `experiments/Stage02_CallUnlock/exp_024/run_map.json` を参照する。

## 4. 実行結果

3 seed すべて正常完了した。

| seed | label | status |
|---:|---|---|
| 42 | `Y_onwide_separated_seed42` | completed |
| 43 | `Y_onwide_separated_seed43` | completed |
| 44 | `Y_onwide_separated_seed44` | completed |

加えて、各 run の final checkpoint と `cycle_59/selfplay` shard に対して semantic eval を実行した。

- `experiments/Stage02_CallUnlock/exp_024/semantic_eval_seed42_final_cycle59/semantic_eval_final_cycle59_summary.md`
- `experiments/Stage02_CallUnlock/exp_024/semantic_eval_seed43_final_cycle59/semantic_eval_final_cycle59_summary.md`
- `experiments/Stage02_CallUnlock/exp_024/semantic_eval_seed44_final_cycle59/semantic_eval_final_cycle59_summary.md`

## 5. 主結果

avg_rank は低いほど良い。

### 5.1 exp023 reference

| seed | imitation | final | best | best cycle | best5 | best10 | tail5 | tail10 | tail20 | final win | final deal-in |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.365 | 2.155 | 1.975 | c58 | 2.115 | 2.143 | 2.129 | 2.153 | 2.168 | 0.2274 | 0.2031 |
| 43 | 2.450 | 2.265 | 2.135 | c57 | 2.248 | 2.264 | 2.250 | 2.272 | 2.272 | 0.2354 | 0.1913 |
| 44 | 2.240 | 2.080 | 2.010 | c25 | 2.104 | 2.138 | 2.150 | 2.171 | 2.159 | 0.2210 | 0.1840 |
| avg | 2.352 | 2.167 | 2.040 | - | 2.156 | 2.182 | 2.176 | 2.199 | 2.200 | 0.2279 | 0.1928 |

### 5.2 exp024 on_wide separated

| seed | imitation | final | best | best cycle | best5 | best10 | tail5 | tail10 | tail20 | final win | final deal-in |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.450 | 2.170 | 2.015 | c58 | 2.125 | 2.130 | 2.132 | 2.139 | 2.158 | 0.2013 | 0.1818 |
| 43 | 2.385 | 2.190 | 2.010 | c26 | 2.088 | 2.109 | 2.183 | 2.146 | 2.144 | 0.2270 | 0.1996 |
| 44 | 2.395 | 2.065 | 2.055 | c27 | 2.146 | 2.151 | 2.195 | 2.222 | 2.217 | 0.2366 | 0.1908 |
| avg | 2.410 | 2.142 | 2.027 | - | 2.120 | 2.130 | 2.170 | 2.169 | 2.173 | 0.2216 | 0.1908 |

### 5.3 3seed 平均比較

| condition | final | best | best5 | best10 | tail5 | tail10 | tail20 | final win | final deal-in |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp023 baseline | 2.167 | 2.040 | 2.156 | 2.182 | 2.176 | 2.199 | 2.200 | 0.2279 | 0.1928 |
| exp024 on_wide | 2.142 | 2.027 | 2.120 | 2.130 | 2.170 | 2.169 | 2.173 | 0.2216 | 0.1908 |
| diff | -0.025 | -0.013 | -0.036 | -0.052 | -0.006 | -0.030 | -0.027 | -0.0063 | -0.0020 |

読み:

- 順位系指標はほぼすべて `exp024` が改善
- 特に `best10` と `tail10` の改善が見やすい
- `final win_rate` は少し下がった
- `deal_in_rate` はわずかに改善
- 総合的には、policy performance は小〜中程度改善したと読む

## 6. cycle window

3 seed 平均。

| condition | c00-c09 | c10-c19 | c20-c29 | c30-c39 | c40-c49 | c50-c59 |
|---|---:|---:|---:|---:|---:|---:|
| exp023 baseline | 2.302 | 2.274 | 2.264 | 2.219 | 2.201 | 2.199 |
| exp024 on_wide | 2.339 | 2.284 | 2.230 | 2.172 | 2.177 | 2.169 |
| diff | +0.037 | +0.010 | -0.034 | -0.047 | -0.024 | -0.030 |

読み:

- early は `exp024` がやや悪い
- `c20` 以降は `exp024` が一貫して良い
- `c50-c59` でも改善しており、late collapse は見えない
- `imitation` は `exp024` の方が悪いが、PPO 後半で追い越している

## 7. PPO diagnostics

final cycle の 3 seed 平均。

| condition | entropy_last | clip_last | log_ratio_p01_last | ratio_max_last | max_prob_mean_last |
|---|---:|---:|---:|---:|---:|
| exp023 baseline | 0.2841 | 0.0897 | -0.4537 | 6.0594 | 0.8853 |
| exp024 on_wide | 0.2849 | 0.0846 | -0.4248 | 3.7602 | 0.8842 |

読み:

- entropy は維持
- `clip_fraction` はわずかに改善
- `log_ratio_p01` は少し穏やか
- `ratio_max` は改善
- `max_prob_mean` はほぼ同等

つまり、`tile_presence_flags + wide value trunk` は、今回の条件では policy を尖らせすぎたり、PPO ratio を荒らしたりしていない。

## 8. semantic eval

### 8.1 exp023 reference

| seed | terminal acc | yaku micro | yaku macro | exact | Tanyao p | Tanyao hit@0.2 | Tanyao top3 | Tanyao recall | Yakuhai p | Yakuhai recall | Pinfu p | Pinfu recall | win_called top3 | deal AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.6157 | 0.4104 | 0.0841 | 0.1775 | 0.2692 | 0.8981 | 0.8535 | 0.0000 | 0.3755 | 0.0198 | 0.2237 | 0.0236 | 0.7919 | 0.5332 |
| 43 | 0.6273 | 0.3322 | 0.0578 | 0.1333 | 0.1843 | 0.3732 | 0.8693 | 0.0000 | 0.4049 | 0.0055 | 0.1250 | 0.0000 | 0.9177 | 0.5320 |
| 44 | 0.6193 | 0.4368 | 0.0841 | 0.1604 | 0.3515 | 0.9389 | 0.9158 | 0.0046 | 0.1897 | 0.0000 | 0.2676 | 0.0795 | 0.2531 | 0.5195 |
| avg | 0.6208 | 0.3931 | 0.0753 | 0.1571 | 0.2683 | 0.7367 | 0.8795 | 0.0015 | 0.3234 | 0.0084 | 0.2054 | 0.0344 | 0.6542 | 0.5282 |

### 8.2 exp024 on_wide

| seed | terminal acc | yaku micro | yaku macro | exact | Tanyao p | Tanyao hit@0.2 | Tanyao top3 | Tanyao recall | Yakuhai p | Yakuhai recall | Pinfu p | Pinfu recall | win_called top3 | deal AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.5756 | 0.4511 | 0.0767 | 0.1683 | 0.2130 | 0.4381 | 0.1583 | 0.0000 | 0.2038 | 0.0000 | 0.3565 | 0.1075 | 0.2604 | 0.5464 |
| 43 | 0.6225 | 0.4182 | 0.0965 | 0.1983 | 0.3313 | 0.9275 | 0.9327 | 0.0496 | 0.1550 | 0.0019 | 0.2424 | 0.0172 | 0.2489 | 0.5424 |
| 44 | 0.6198 | 0.3569 | 0.0949 | 0.1558 | 0.3625 | 0.9614 | 0.9803 | 0.0447 | 0.4381 | 0.3267 | 0.2249 | 0.1532 | 0.8797 | 0.5369 |
| avg | 0.6060 | 0.4087 | 0.0894 | 0.1741 | 0.3023 | 0.7757 | 0.6904 | 0.0314 | 0.2656 | 0.1095 | 0.2746 | 0.0926 | 0.4630 | 0.5419 |

### 8.3 semantic 差分

| metric | exp023 | exp024 | diff |
|---|---:|---:|---:|
| terminal accuracy | 0.6208 | 0.6060 | -0.0148 |
| yaku micro F1 | 0.3931 | 0.4087 | +0.0156 |
| yaku macro F1 | 0.0753 | 0.0894 | +0.0140 |
| yaku exact match | 0.1571 | 0.1741 | +0.0171 |
| Tanyao mean_p | 0.2683 | 0.3023 | +0.0339 |
| Tanyao hit@0.2 | 0.7367 | 0.7757 | +0.0389 |
| Tanyao top3 | 0.8795 | 0.6904 | -0.1891 |
| Tanyao recall | 0.0015 | 0.0314 | +0.0299 |
| Yakuhai mean_p | 0.3234 | 0.2656 | -0.0577 |
| Yakuhai recall | 0.0084 | 0.1095 | +0.1011 |
| Pinfu mean_p | 0.2054 | 0.2746 | +0.0692 |
| Pinfu recall | 0.0344 | 0.0926 | +0.0583 |
| win_called top3 | 0.6542 | 0.4630 | -0.1912 |
| deal_in ROC-AUC | 0.5282 | 0.5419 | +0.0137 |

読み:

- yaku 全体は改善
- `Tanyao` は `mean_p`, `hit@0.2`, `recall` が改善
- `Pinfu` はかなり改善
- `Yakuhai` は `mean_p` は下がったが、recall は上がった
- terminal はやや悪化し、特に `win_called top3` は悪化
- `deal_in` risk はわずかに改善

`Tanyao top3` の悪化は気になるが、`mean_p` と `recall` は改善しているため、単純に Tanyao signal が弱くなったというより、役間の ranking / calibration が変わったと見るのが自然である。

## 9. 解釈

### 9.1 過去の yakuflags 不採用判断は更新される

`exp_019` では、`yakuflags on_wide` は 3 seed で practical baseline を上回らず、不採用だった。

しかし `exp_024` では、同じ on_wide 系を `separated policy-only PPO` の安定条件上で再評価したところ、順位系指標と yaku semantic 指標の両方で改善した。

したがって、以前の

```text
yakuflags は現在の実験環境では性能向上に寄与しない
```

という判断は、現行環境では撤回してよい。

より正確には、次の判断に更新する。

```text
mixed PPO 問題が残っていた旧環境では yakuflags は採用できなかった。
separated policy-only PPO 環境では、on_wide 条件は Stage2a baseline を小幅に改善する。
```

### 9.2 改善は小さいが、多面的に整合している

`final`, `tail10`, `best10` が改善し、PPO diagnostics は悪化していない。

さらに semantic eval でも yaku 指標が改善している。

このため、今回の改善は単なる評価ノイズというより、条件差に意味がある可能性が高い。

ただし差分は大きくはない。`exp_023` 自体がかなり強い baseline なので、ここからの改善幅としては妥当な範囲である。

### 9.3 feature 単独効果ではない

今回の差分は次の 2 つを同時に含む。

- `tile_presence_flags=true`
- `model.value_hidden_dims=[256,128]`

したがって、改善を `tile_presence_flags` 単独の効果とは断定しない。

ただし過去の `exp_017` で `on_narrow` が弱く `on_wide` が最有望だったため、今回の実験は「採用候補として勝ち筋のある yakuflags 条件」を検証するものだった。

実務上は、この組み合わせを新 baseline 候補として扱ってよい。

### 9.4 terminal 側には副作用がある

terminal accuracy と `win_called top3` は悪化した。

これは、役ヒント特徴量が yaku/value 側には効く一方で、terminal class の一部、特に `win_called` の ranking には悪影響を与えている可能性を示す。

ただし practical performance は改善しており、PPO diagnostics も安定しているため、この副作用だけで採用を見送るほどではない。

## 10. 実務判断

今回の判断:

```text
exp024 on_wide separated を Stage2a 新 baseline 候補に昇格する。
```

採用寄りの理由:

- performance が 3 seed 平均で改善
- late tail でも悪化しない
- PPO diagnostics が悪化しない
- yaku semantic 指標が改善
- 以前の yakuflags 不採用判断の前提だった mixed PPO 問題が消えた状態で改善した

保留点:

- feature 単独効果ではなく wide 化との複合効果
- terminal accuracy / win_called top3 は悪化
- improvement 幅は小〜中程度

## 11. 次アクション

自然な次アクションは 2 つある。

### 11.1 exp024 条件を基準化する

次の実験では、`exp024 on_wide separated` を新 reference として扱う。

新 baseline 候補:

```text
separated policy-only PPO
no-anchor
lr=1e-4
clip=0.15
tile_presence_flags=true
value_hidden_dims=[256,128]
tile_presence_flags_semantic_only=false
```

### 11.2 policy_ratio sweep を再検討する

`exp_023` の後に候補として挙げていた `policy_ratio` sweep は、`exp_024` 条件を新基準として行う方がよい。

候補:

- `policy_ratio=0.50` reference: exp024
- `policy_ratio=0.75`
- `policy_ratio=1.00`

問い:

- baseline agent を卓に混ぜる比率を減らしても安定するか
- policy sample を増やすことでさらに伸びるか
- rule-based opponent は環境形成役としてどの程度必要か

## 12. 結論

`exp_024` の結論:

1. `yakuflags on_wide` は、現行 separated PPO 環境では `exp_023` baseline を小幅に上回った
2. `final`, `tail10`, `best10` はすべて改善した
3. PPO diagnostics は悪化せず、ratio / clip はむしろ少し安定した
4. yaku semantic 指標も改善した
5. terminal accuracy / win_called top3 には副作用がある
6. feature 単独効果とは断定しないが、実務上は `tile_presence_flags=true + value_hidden_dims=[256,128]` を新 baseline 候補にしてよい
7. 次はこの条件を基準に、`policy_ratio` など selfplay 混合比の最適化へ進むのが自然である
