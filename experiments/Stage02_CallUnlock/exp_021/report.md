# Experiment Report: exp_021

作成日: 2026-05-01  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_021/runbook.md`
- `experiments/Stage02_CallUnlock/exp_021/run_map.json`
- `experiments/Stage02_CallUnlock/exp_020/report.md`
- `docs/CHANGE_QUEUE.md`

## 1. 要約

`exp_021` は、`exp_020` で確認された late drift の原因候補として `teacher anchor` を疑い、no-anchor 系の PPO 更新を比較した実験である。

結論は次の通り。

- 全 12 run のうち、再利用 4 run、完了済み 2 run、新規 6 run を使って A/B/C/D を比較した
- `C_noanchor_lr1e4_clip010_seed42` の初回 continuation だけ eval worker crash で失敗したが、再実行は成功した
- 3 seed 平均では `B: no-anchor, lr=1e-4, clip=0.15` が最も良い
- B は final / best / best5 / tail5 / tail10 のすべてで anchor あり A を上回った
- B は best 一点から tail5 への drift はあるが、tail 平均でも imitation より改善している
- `C: clip=0.10` は B を超えず、clip を狭める明確な利点は見えなかった
- `D: lr=5e-5` は `training.lr` が imitation warmstart にも効いており、PPO lr だけを下げた純粋比較ではない
- D は結果も B を超えていないため、現時点では深掘り優先度は低い

暫定採用候補は以下。

```text
policy_anchor.enabled = false
training.lr = 1e-4
training.clip_epsilon = 0.15
```

ただし no-anchor は PPO diagnostics 上、anchor ありより更新が強い。特に `clip_fraction` が高く、entropy が低下する。したがって次は「B の改善を維持しつつ、更新の荒さをどう抑えるか」が焦点である。

## 2. 背景

`exp_020` では、`CQ-0274`〜`CQ-0279` 修正後の Stage2a baseline を 3 seed で取り直した。

`exp_020` の主な結論:

- RL は imitation より良い checkpoint に到達できる
- best cycle は `cycle20` 付近に集中した
- しかし 30 cycle final では性能を保持できず、late drift が見られた

当時の baseline は `policy_anchor.reference=imitation_fixed`, `coef=0.75` を使っていた。

そこで `exp_021` では、現行 fixedRL 条件で no-anchor を試し、さらに更新強度の調整として `clip_epsilon=0.10` と `lr=5e-5` を比較した。

## 3. 実験条件

共通条件:

- config: `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- `training.multi_cycle.num_cycles = 30`
- `training.multi_cycle.selfplay_matches_per_cycle = 200`
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

比較条件:

| Group | Anchor | lr | clip | 備考 |
|---|---|---:|---:|---|
| A | `enabled=true`, `coef=0.75` | 1e-4 | 0.15 | exp_020 を再利用 |
| B | disabled | 1e-4 | 0.15 | seed42 は no-anchor probe を再利用 |
| C | disabled | 1e-4 | 0.10 | clip を狭める |
| D | disabled | 5e-5 | 0.15 | lr を下げる。ただし imitation lr も下がっている |

run:

| Group | seed42 | seed43 | seed44 |
|---|---|---|---|
| A | `runs/20260430_stage2a_exp020_A2_semaux_light_vhalf_tenpaifix_prnorm_fixedrl_seed42_0a26a46e` | `runs/20260430_stage2a_exp020_A2_semaux_light_vhalf_tenpaifix_prnorm_fixedrl_seed43_6fde9bf9` | `runs/20260430_stage2a_exp020_A2_semaux_light_vhalf_tenpaifix_prnorm_fixedrl_seed44_dac201a9` |
| B | `runs/20260430_stage2a_anchor_probe_noanchor_seed42_af41f851` | `runs/20260430_stage2a_exp021_B_noanchor_lr1e4_clip015_seed43_28f081bd` | `runs/20260430_stage2a_exp021_B_noanchor_lr1e4_clip015_seed44_a4311ba8` |
| C | `runs/20260430_stage2a_exp021_C_noanchor_lr1e4_clip010_seed42_05f59603` | `runs/20260430_stage2a_exp021_C_noanchor_lr1e4_clip010_seed43_273200c3` | `runs/20260430_stage2a_exp021_C_noanchor_lr1e4_clip010_seed44_278851e8` |
| D | `runs/20260430_stage2a_exp021_D_noanchor_lr5e5_clip015_seed42_b072a037` | `runs/20260501_stage2a_exp021_D_noanchor_lr5e5_clip015_seed43_1376cd1f` | `runs/20260501_stage2a_exp021_D_noanchor_lr5e5_clip015_seed44_93da7b08` |

## 4. 主結果

avg_rank は低いほど良い。

### 4.1 3 seed 平均

| Group | imitation | final | best | best5 | tail5 | tail10 |
|---|---:|---:|---:|---:|---:|---:|
| A anchor075 lr1e-4 clip0.15 | 2.352 | 2.360 | 2.188 | 2.311 | 2.392 | 2.355 |
| B no-anchor lr1e-4 clip0.15 | 2.352 | 2.327 | 2.148 | 2.246 | 2.288 | 2.303 |
| C no-anchor lr1e-4 clip0.10 | 2.352 | 2.355 | 2.165 | 2.257 | 2.313 | 2.303 |
| D no-anchor lr5e-5 clip0.15 | 2.330 | 2.363 | 2.190 | 2.253 | 2.361 | 2.338 |

読み:

- B は A より全主要指標で良い
- C は B に近いが、B を明確に上回らない
- D は B を上回らない
- D は imitation baseline 自体が A/B/C と異なっており、PPO lr だけの効果としては読めない

### 4.2 seed 別結果

| Group | seed | imitation | final | final差分 | best | best差分 | tail5 | tail5差分 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 42 | 2.365 | 2.440 | +0.075 | 2.225 | -0.140 | 2.455 | +0.090 |
| A | 43 | 2.450 | 2.295 | -0.155 | 2.155 | -0.295 | 2.294 | -0.156 |
| A | 44 | 2.240 | 2.345 | +0.105 | 2.185 | -0.055 | 2.428 | +0.188 |
| B | 42 | 2.365 | 2.390 | +0.025 | 2.190 | -0.175 | 2.337 | -0.028 |
| B | 43 | 2.450 | 2.290 | -0.160 | 2.145 | -0.305 | 2.292 | -0.158 |
| B | 44 | 2.240 | 2.300 | +0.060 | 2.110 | -0.130 | 2.234 | -0.006 |
| C | 42 | 2.365 | 2.485 | +0.120 | 2.240 | -0.125 | 2.369 | +0.004 |
| C | 43 | 2.450 | 2.355 | -0.095 | 2.185 | -0.265 | 2.344 | -0.106 |
| C | 44 | 2.240 | 2.225 | -0.015 | 2.070 | -0.170 | 2.227 | -0.013 |
| D | 42 | 2.330 | 2.440 | +0.110 | 2.210 | -0.120 | 2.460 | +0.130 |
| D | 43 | 2.340 | 2.395 | +0.055 | 2.230 | -0.110 | 2.394 | +0.054 |
| D | 44 | 2.320 | 2.255 | -0.065 | 2.130 | -0.190 | 2.228 | -0.092 |

読み:

- B は 3 seed すべてで tail5 が imitation より良い
- A は seed42/44 で tail5 が imitation より悪化
- C は seed42 が弱く、B より安定して良いとは言えない
- D は seed44 だけ良いが、seed42/43 は tail が悪い

## 5. B の詳細

B は平均的に PPO で改善していると読める。

3 seed 平均:

| 指標 | avg_rank | imitation差分 |
|---|---:|---:|
| imitation | 2.352 | 0.000 |
| final | 2.327 | -0.025 |
| best | 2.148 | -0.203 |
| best5 | 2.246 | -0.106 |
| tail5 | 2.288 | -0.064 |
| tail10 | 2.303 | -0.048 |

cycle window:

| Window | avg_rank |
|---|---:|
| c00-c04 | 2.357 |
| c05-c09 | 2.325 |
| c10-c14 | 2.341 |
| c15-c19 | 2.299 |
| c20-c24 | 2.319 |
| c25-c29 | 2.288 |

読み:

- B は best だけでなく tail5 / tail10 でも imitation より良い
- best 一点から tail5 への drift はある
- ただし late window 平均はむしろ良く、exp_020 A のような明確な late drift とは違う
- B は「一瞬だけ良い」ではなく、後半も改善状態をある程度維持している

B の drift:

| seed | best | tail5 | best→tail5 drift | best5→tail5 drift |
|---:|---:|---:|---:|---:|
| 42 | 2.190 | 2.337 | +0.147 | +0.029 |
| 43 | 2.145 | 2.292 | +0.147 | +0.073 |
| 44 | 2.110 | 2.234 | +0.124 | +0.023 |

読み:

- best 一点からは `+0.12〜0.15` 程度戻る
- best5 から tail5 への戻りは小さめ
- 「最高点を完全保持できる」状態ではないが、「late collapse」ではない

## 6. PPO diagnostics

| Group | ratio_avg | ratio_last | ratio_std_avg | clip_avg | clip_last | clip_max | entropy_avg | entropy_last |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1.003 | 1.004 | 0.101 | 0.102 | 0.094 | 0.141 | 0.548 | 0.545 |
| B | 0.967 | 0.947 | 0.235 | 0.208 | 0.238 | 0.249 | 0.370 | 0.233 |
| C | 0.978 | 0.962 | 0.177 | 0.239 | 0.250 | 0.278 | 0.397 | 0.273 |
| D | 0.968 | 0.966 | 0.178 | 0.202 | 0.212 | 0.224 | 0.442 | 0.311 |

読み:

- no-anchor は A より `ratio_mean` が 1 未満に寄る
- no-anchor は A より `clip_fraction` が約 2 倍
- no-anchor は entropy が大きく下がる
- B は性能が最も良い一方、entropy 低下も強い
- C は clip を狭めたため、`clip_fraction` はむしろ高くなっている
- D は entropy は比較的残るが、性能面で B を超えない

解釈:

- no-anchor は imitation-fixed anchor から解放され、PPO 更新がより強く方策を動かしている
- その結果、A より性能は伸びるが、更新の荒さは増える
- 現時点ではこの荒さは性能改善につながっている
- ただし長期化すると entropy 低下・clip 高止まりが drift の原因になる可能性は残る

## 7. D の caveat

D は「PPO lr を 5e-5 に下げる」意図だったが、実装上は `training.lr=5e-5` が imitation warmstart にも効いている。

確認箇所:

- `python/mahjong_rl/stage2a_learner.py`
  - `self._lr = tc.get("lr", 3e-4)`
  - `self._optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)`
- `python/mahjong_rl/runner.py`
  - Stage2a multi-chunk imitation では `lc = self._as_dict(); lc["training"]["algorithm"] = "imitation"` で learner を作る
  - `training.imitation_optimizer` による上書きはこの経路では使われていない

そのため D は次の条件になっていた可能性が高い。

```text
imitation lr = 5e-5
PPO lr = 5e-5
```

実際、D の imitation eval は A/B/C と異なる。

| seed | A/B/C imitation | D imitation |
|---:|---:|---:|
| 42 | 2.365 | 2.330 |
| 43 | 2.450 | 2.340 |
| 44 | 2.240 | 2.320 |

したがって D は PPO lr だけの効果としては読めない。

ただし、出た結果として D は B を超えていないため、現時点で D を厳密にやり直す優先度は低い。

## 8. eval worker crash

`C_noanchor_lr1e4_clip010_seed42` の初回 continuation で以下が発生した。

```text
Stage2a eval worker errors:
eval_worker_9: non-zero exit code -11
```

該当 run:

- failed: `runs/20260430_stage2a_exp021_C_noanchor_lr1e4_clip010_seed42_f921d04f`
- success retry: `runs/20260430_stage2a_exp021_C_noanchor_lr1e4_clip010_seed42_05f59603`

失敗箇所は `cycle_12` の learner 完了後の eval subprocess であり、学習本体の Python 例外ではなかった。

暫定 driver は batch-level continue-on-error に変更した。1 run が失敗しても `run_map.json` に failed として記録し、次の条件へ進む。

恒久対応として `CQ-0280` を追加済み。

`CQ-0280` の方針:

- Stage2a multi-cycle eval worker failure 時に、同じ checkpoint/model state で eval だけ retry
- selfplay / learner は再実行しない
- retry worker 数・retry 回数を config 化
- retry 成功時は run 継続

## 9. 結論

`exp_021` の主結論は以下。

1. no-anchor は現行 fixedRL 条件で有望
2. 特に `B: no-anchor, lr=1e-4, clip=0.15` が最も良い
3. `clip=0.10` は採用理由が弱い
4. `lr=5e-5` は imitation lr も下げてしまっており、比較として汚れている
5. ただし D は結果も B を超えていないため、一旦スルーでよい
6. B は平均的に PPO で改善している
7. B にも best からの戻りはあるが、tail 平均でも imitation より良い
8. no-anchor は PPO diagnostics が荒くなるため、次は改善保持と更新安定化の両立を見る

暫定 practical setting:

```text
policy_anchor.enabled = false
training.lr = 1e-4
training.clip_epsilon = 0.15
feature_encoder.tile_presence_flags = false
model.semantic_aux.tile_presence_flags_semantic_only = false
```

## 10. 次の候補

優先度が高いもの:

1. B 設定を次の baseline として扱い、追加比較を組む
2. `CQ-0280` を実装し、eval worker crash を run 内 retry で吸収する
3. PPO diagnostics に advantage sign distribution を追加する
4. no-anchor の entropy 低下が問題か確認する

次に試す価値がある実験:

- `B + entropy_coef` の軽い追加
- `B + PPO-only lr decay`
- `B + cycle-based clip/lr schedule`
- `B + lagged anchor` または `anchor decay`

一方で、現時点で優先度が低いもの:

- D の単純な再実行
- `clip=0.10` の追加 seed
- yakuflags 系の再検証
