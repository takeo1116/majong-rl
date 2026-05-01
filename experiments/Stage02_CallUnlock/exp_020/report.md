# Experiment Report: exp_020

作成日: 2026-04-30  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_020/runbook.md`
- `experiments/Stage02_CallUnlock/exp_020/run_map.json`
- `experiments/Stage02_CallUnlock/exp_015/report.md`
- `docs/CHANGE_QUEUE.md`

## 1. 要約

`exp_020` は、`CQ-0274`〜`CQ-0279` で Stage2a の RL 更新まわりの不備を修正した後、現在の practical baseline である

- `A2_semaux_light_vhalf_tenpaifix_prnorm`

を 3 seed で取り直した実験である。

結論は次の通り。

- 3 run はすべて正常完了した
- PPO diagnostics は壊れていない
- 全 seed で imitation より明確に良い best checkpoint に到達した
- best cycle は `cycle20 / cycle20 / cycle21` に集中した
- 一方で、30 cycle final まで性能を保持することには失敗した
- late drift は 3 seed でかなり再現性がある
- 現状は「RL が学べない」ではなく、「良い領域に入った後に保持できない」と整理するのが自然である

したがって、`CQ-0274`〜`CQ-0279` の修正により RL 信号はかなり健全化した可能性が高いが、まだ学習安定化は未完である。

次の焦点は、checkpoint selection ではなく、**late drift の原因を潰して最高点付近を保持できるようにすること**である。特に `teacher anchor` (`policy_anchor.reference=imitation_fixed`, `coef=0.75`) の影響は次に検証する価値が高い。

## 2. 背景

20 日ぶりの再点検で、Stage2a の RL 更新には複数の実装・運用 semantics 上の問題が見つかった。

今回までに修正済みの主な CQ は以下。

- `CQ-0274`: pending reward の上書きを廃止し、全 pending sample に reward を累積
- `CQ-0275`: PPO return / advantage を discard / call branch の元順に scatter
- `CQ-0276`: `reward_config` を Stage2a selfplay / eval / parallel 経路に伝播
- `CQ-0277`: terminal player-round weight を discard / call cross-branch で計算
- `CQ-0278`: Stage2a selfplay の torch RNG を match seed で固定し、`selfplay.temperature` を実際に反映
- `CQ-0279`: Stage2a shard semantics を v3 に上げ、旧 v2 shard を learner 側で fail-fast

このため、過去の `exp_015` / `exp_019` の結果は参考にはなるが、現行コードの RL 更新性能は fresh shard で取り直す必要があった。

## 3. 実験条件

全 seed 共通:

- config: `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- condition: `A2_semaux_light_vhalf_tenpaifix_prnorm_fixedrl`
- `training.multi_cycle.num_cycles = 30`
- `training.multi_cycle.selfplay_matches_per_cycle = 200`
- `training.policy_anchor.coef = 0.75`
- `training.policy_anchor.reference = imitation_fixed`
- `training.value_loss_coef = 0.125`
- `model.value_hidden_dims = [128, 64]`
- `model.semantic_aux.enabled = true`
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.enabled = true`
- `training.semantic_aux.terminal_loss_coef = 0.1`
- `training.semantic_aux.yaku_loss_coef = 0.05`
- `feature_encoder.tile_presence_flags = false`
- `model.semantic_aux.tile_presence_flags_semantic_only = false`
- `selfplay.temperature = 1.0`
- expected shard semantics: `sample_semantics_version = 3`

実行 seed:

| label | seed | run_dir |
|---|---:|---|
| `A2_semaux_light_vhalf_tenpaifix_prnorm_fixedrl_seed42` | 42 | `runs/20260430_stage2a_exp020_A2_semaux_light_vhalf_tenpaifix_prnorm_fixedrl_seed42_0a26a46e` |
| `A2_semaux_light_vhalf_tenpaifix_prnorm_fixedrl_seed43` | 43 | `runs/20260430_stage2a_exp020_A2_semaux_light_vhalf_tenpaifix_prnorm_fixedrl_seed43_6fde9bf9` |
| `A2_semaux_light_vhalf_tenpaifix_prnorm_fixedrl_seed44` | 44 | `runs/20260430_stage2a_exp020_A2_semaux_light_vhalf_tenpaifix_prnorm_fixedrl_seed44_dac201a9` |

## 4. 主結果

### 4.1 seed 別結果

| seed | imitation | final | final差分 | best | tail5 | tail10 | final win/deal | best win/deal |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.365 | 2.440 | +0.075 | c20 2.225 | 2.455 | 2.392 | 0.2387 / 0.1935 | 0.2597 / 0.1890 |
| 43 | 2.450 | 2.295 | -0.155 | c20 2.155 | 2.294 | 2.287 | 0.2529 / 0.1861 | 0.2660 / 0.1874 |
| 44 | 2.240 | 2.345 | +0.105 | c21 2.185 | 2.428 | 2.385 | 0.2454 / 0.1886 | 0.2596 / 0.1849 |

読み:

- best checkpoint は全 seed で imitation より良い
- final は seed43 のみ imitation より良い
- seed42 / seed44 は best から final までに大きく悪化
- seed43 も best からは悪化しているが、final でも imitation より良い水準を保った

### 4.2 3 seed 平均

| 指標 | avg_rank | win_rate | deal_in_rate |
|---|---:|---:|---:|
| imitation | 2.352 | 0.2531 | 0.1881 |
| final | 2.360 | 0.2457 | 0.1894 |
| best cycle | 2.188 | 0.2618 | 0.1871 |
| tail5 | 2.392 | 0.2513 | 0.1893 |
| tail10 | 2.355 | 0.2544 | 0.1877 |

読み:

- best cycle 平均は非常に良い
- final 平均は imitation とほぼ同等まで戻る
- tail5 は imitation より悪い
- tail10 は imitation とほぼ同等

つまり、RL 更新は「強い点に到達する能力」はあるが、「30 cycle まで保持する能力」はまだ弱い。

## 5. cycle 形状

best cycle は以下に集中した。

| seed | best cycle | best avg_rank |
|---:|---:|---:|
| 42 | c20 | 2.225 |
| 43 | c20 | 2.155 |
| 44 | c21 | 2.185 |

5-cycle window で見ると、3seed 平均の最良 window は `cycle20-24` だった。

| window | 3seed avg_rank |
|---|---:|
| c18-c22 | 2.326 |
| c19-c23 | 2.335 |
| c20-c24 | 2.317 |
| c25-c29 | 2.392 |

読み:

- `cycle20` 付近でかなり再現性高く良い領域に入る
- その後 `cycle25-29` では明確に悪化する
- late drift は単発の偶然ではなく、少なくとも今回の 3 seed では構造的に見える

## 6. PPO diagnostics

| seed | ratio_mean avg/last/max | clip_fraction avg/last/max | anchor_kl_discard avg/last/max |
|---:|---:|---:|---:|
| 42 | 1.0048 / 1.0007 / 1.0256 | 0.1101 / 0.0964 / 0.1672 | 0.00662 / 0.00975 / 0.01075 |
| 43 | 1.0023 / 1.0049 / 1.0172 | 0.1131 / 0.1053 / 0.1312 | 0.00594 / 0.00640 / 0.00739 |
| 44 | 1.0019 / 1.0062 / 1.0164 | 0.0842 / 0.0791 / 0.1260 | 0.00413 / 0.00448 / 0.00473 |

読み:

- `ratio_mean` は健全
- `clip_fraction` も高すぎる状態ではない
- `anchor_kl_discard` は seed42 で後半にやや上がるが、破滅的ではない
- 少なくとも exp_004 初期の mixed PPO 崩壊のような挙動ではない

したがって late drift は PPO 数値の爆発ではなく、良い policy 近辺で更新方向が安定しない現象と見るのが自然である。

## 7. 解釈

### 7.1 RL 更新は有効化されている

`CQ-0274`〜`CQ-0279` 修正後、全 seed で imitation より良い best checkpoint に到達した。

これは重要で、以前の「RL がそもそも安定して改善できているか分からない」という状態からは前進している。

特に best 平均 `2.188` は、現在の Stage02a 条件としてかなり強い。

### 7.2 ただし final 採用はまだ危険

30 cycle final は平均 `2.360` で、imitation 平均 `2.352` とほぼ同等である。

これは、学習が進んだ結果として最終 policy が強くなっているというより、途中で強い点を通過した後に drift して戻っている形に近い。

したがって現時点で「最後の checkpoint を採用する」運用は危険である。

ただし、これは checkpoint selection だけで解決すべき問題ではない。学習が本当に安定していれば、良い領域に入った後もそこに留まれるはずである。

### 7.3 teacher anchor は疑う価値がある

今回の条件では、`policy_anchor.reference = imitation_fixed`、`coef = 0.75` が常に有効である。

これは初期には policy を壊さないために有効だったが、RL が一度 imitation より良い方策に到達した後には、次のような綱引きを生む可能性がある。

- PPO は selfplay 分布上で改善方向に押す
- imitation-fixed anchor は元の teacher policy 側に引き戻す
- その結果、良い中間点に留まれず、cycle ごとに揺れる

過去実験では anchor `0.5 / 0.75 / 1.0` は試しているが、現在の `A2 + fixedRL + v3 shard` 条件での no-anchor は未実施である。

初期の mixed PPO では anchor が安定化に重要だったが、当時は reward/GAE/RNG/shard semantics に未修正の問題が残っていた。したがって、過去の不安定性だけで no-anchor を棄却するのは早い。

## 8. 次の実験候補

最有力は、現行 A2 fixedRL 条件での teacher anchor ablation である。

比較候補:

1. `anchor075`
   - exp_020 相当
   - `policy_anchor.enabled = true`
   - `policy_anchor.coef = 0.75`
2. `noanchor`
   - `policy_anchor.enabled = false`
   - または `policy_anchor.coef = 0.0`

見るべき点:

- best cycle が維持されるか
- late drift が弱まるか
- `clip_fraction` / `ratio_mean` が壊れるか
- win_rate / deal_in_rate のバランス
- semantic aux が壊れないか

もし no-anchor で late drift が改善するなら、teacher anchor は初期安定化には有用だが、後半の保持を阻害している可能性が高くなる。

逆に no-anchor が崩壊するなら、次は anchor を外すのではなく、以下を検討する。

- anchor coefficient decay
- lagged/self anchor
- cycle-based LR decay
- clip epsilon decay

## 9. 結論

`exp_020` の結論は次の通り。

- 修正後 Stage2a RL は、3 seed すべてで imitation より強い checkpoint に到達した
- best cycle は `c20` 付近に集中し、再現性が高い
- しかし final / tail では性能保持に失敗している
- late drift は再現性のある問題として扱うべき
- 現時点の課題は「RL が学べない」ではなく、「良い policy を保持できない」こと
- 次は teacher anchor の有無を疑う実験が最も自然である
