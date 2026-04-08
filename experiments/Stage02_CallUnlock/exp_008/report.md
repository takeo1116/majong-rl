# Experiment Report: exp_008

作成日: 2026-03-31  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_008/runbook.md`
- `experiments/Stage02_CallUnlock/exp_008/run_map.json`
- `experiments/Stage02_CallUnlock/exp_007/report.md`
- `experiments/Stage02_CallUnlock/exp_006/report.md`

## 1. 要約

`exp_008` では、`exp_007` で最も有望だった `R3` 条件をそのまま使い、
**50 cycle mixed PPO を 3 seeds で回したときに improvement が積み上がるか**
を確認した。

結論は次の通り。

- **3 seeds とも stable**
- imitation 直後より final が少し良い seed はある
- ただし **50 cycle まで延ばしたことによる明確な improvement accumulation は弱い**
- best cycle はむしろ早い段階に来ることが多く、後半で大きく伸びる傾向は見えない

したがって、

- `R3` は **長時間でも壊れない mixed 条件**としては十分有望
- しかし **cycle 数を増やすだけでは optional を含めた改善量はあまり増えない**

と整理するのが自然である。

## 2. 実験目的

`exp_007` では、S1 mixed baseline から 1 ノブだけを動かした比較を行い、
`policy_anchor.coef=0.75` の `R3` が最も有望だった。

20 cycle 時点では、

- stable に完走
- imitation 直後より final が改善
- discard 側では branch-swap 上も改善の気配

が見えていた。

そこで `exp_008` では、

1. `R3` を 50 cycle まで延ばしても stable か
2. improvement が長時間で積み上がるか
3. その傾向が 3 seeds でも再現するか

を確認した。

## 3. 実行条件

基準条件:

- `R3_lower_anchor_075`
- mixed PPO
- A `core_minimal`
- `policy_ratio=0.50`
- `baseline_sample_weight=0.25`
- `policy_anchor.coef=0.75`
- `lr=1e-4`
- `clip_epsilon=0.15`
- `max_grad_norm=0.50`

今回の変更点:

- `training.multi_cycle.num_cycles = 50`
- seed を `42 / 43 / 44` の 3 本に拡張

実行管理:

- `scripts/local/stage2/exp_008_driver.py`
- `experiments/Stage02_CallUnlock/exp_008/run_map.json`

## 4. 対象 run

- `R3_seed42_mc50`
- `R3_seed43_mc50`
- `R3_seed44_mc50`

対応は `experiments/Stage02_CallUnlock/exp_008/run_map.json` を参照。

## 5. 主結果

### imitation 直後と final の比較

| Seed | imitation avg_rank | imitation win_rate | final avg_rank | final win_rate | delta avg_rank | delta win_rate |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 2.315 | 0.2394 | 2.315 | 0.2450 | 0.000 | +0.0056 |
| 43 | 2.395 | 0.2520 | 2.320 | 0.2597 | -0.075 | +0.0076 |
| 44 | 2.380 | 0.2449 | 2.345 | 0.2475 | -0.035 | +0.0026 |

平均:

- imitation: `avg_rank=2.3633`, `win_rate=0.2454`
- final: `avg_rank=2.3267`, `win_rate=0.2507`

### tail-5 average

| Seed | tail-5 avg_rank | tail-5 win_rate |
|---|---:|---:|
| 42 | 2.373 | 0.2463 |
| 43 | 2.368 | 0.2432 |
| 44 | 2.402 | 0.2423 |

tail-5 平均の全 seed 平均:

- `avg_rank=2.3810`
- `win_rate=0.2439`

### final PPO diagnostics

| Seed | ratio_mean | clip_fraction | anchor_kl_discard |
|---|---:|---:|---:|
| 42 | 1.0047 | 0.2280 | 0.0221 |
| 43 | 1.0189 | 0.2404 | 0.0250 |
| 44 | 1.0080 | 0.2273 | 0.0247 |

## 6. 読み取り

### 6.1 安定性はかなり良い

3 seeds とも 50 cycle を最後まで完走し、最終 diagnostics も健全だった。

- `ratio_mean` は全 seed で `1.0` 近傍
- `clip_fraction` は `0.23〜0.24`
- `anchor_kl_discard` は `0.022〜0.025`

つまり、`R3` は
**20 cycle だけでなく 50 cycle でも stable な mixed 条件**
として扱ってよい。

### 6.2 final は imitation より少し良いが、改善量は小さい

全 seed で、

- `avg_rank` は横ばい〜やや改善
- `win_rate` は小幅改善

になっている。

したがって、
**完全に「何も学んでいない」わけではない**。

ただし改善量は大きくない。
特に seed42 は final `avg_rank` が imitation と同値で、
勝率だけが少し増えた形だった。

### 6.3 50 cycle にしたからといって、後半で積み上がる感じは弱い

今回一番大事なのはここである。

best cycle の `avg_rank` は

- seed42: cycle 9 で `2.195`
- seed43: cycle 19 で `2.155`
- seed44: cycle 12 で `2.205`

だった。

つまり、**最良値は比較的早い段階で出る**。
50 cycle まで延ばしても、後半でさらに明確に上積みされる傾向は見えない。

tail-5 平均でも、

- final 1 点よりむしろ少し弱い
- 3 seed 平均では imitation よりごく小さい差しかない

ため、**長時間 PPO を回すだけでは improvement accumulation が起きにくい**
と読むのが自然である。

### 6.4 今回の失敗は「不安定化」ではなく「改善信号の弱さ」

この実験は失敗ではない。
むしろ、

- `R3` が 50 cycle でも壊れない
- mixed PPO の長時間運用が可能

と確認できたのは大きい。

ただし、次に必要なのは
`num_cycles` をさらに伸ばすことではない。

今回の結果は、
**PPO の学習信号そのものが弱い**
ことを示している。

## 7. 結論

`exp_008` から得られる結論は次の通り。

1. `R3` は 50 cycle x 3 seeds でも stable
2. final は imitation より少し良いが、改善量は小さい
3. best cycle は早い段階に来ることが多く、50 cycle 後半での蓄積は弱い
4. したがって、**今後は cycle 数を増やすより、学習信号や表現を改善する方向へ進むべき**

## 8. 次アクション

今回の結果を踏まえると、次の方向はかなり明確である。

### 8.1 `R3` は stable mixed baseline として維持する

`R3` は今後の mixed 実験の土台として使ってよい。

### 8.2 次は semantic auxiliary trunk を優先する

現在の課題は、

- optional 側の改善が弱い
- value / policy が将来の役筋や終局形を十分に扱えていない可能性

である。

そのため、次は `CQ-0256` で整理した

- `terminal_head`
- `yaku_head`
- `value_head`

の 3-head semantic auxiliary trunk を入れ、
その summary を `discard` / `optional` policy に渡す方向が自然である。

参照:

- `docs/CHANGE_QUEUE.md`
- `reference/stage2/stage2a_semantic_aux_trunk_design.md`

### 8.3 当面は「長く回す」より「意味のある改善を出す」ことを優先する

今回の `exp_008` により、

- stable mixed PPO の regime は見つかった
- しかし long-run だけでは改善量は増えない

と分かった。

したがって次の焦点は、
**PPO を長く回すことではなく、optional まで含めて学習が効く表現を作ること**
である。
