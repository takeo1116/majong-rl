# Experiment Report: exp_006

作成日: 2026-03-30  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_006/runbook.md`
- `experiments/Stage02_CallUnlock/exp_006/run_map.json`
- `experiments/Stage02_CallUnlock/exp_005/report.md`
- `experiments/Stage02_CallUnlock/exp_004/report.md`

## 1. 要約

`exp_006` では、`exp_005` で mixed canonical baseline とみなした S1 run を対象に、
branch-swap eval を実施した。

比較したのは以下の 4 条件である。

- `II`: discard=imitation, optional=imitation
- `FI`: discard=final, optional=imitation
- `IF`: discard=imitation, optional=final
- `FF`: discard=final, optional=final

value branch は全条件で final に固定した。

結論としては、**4 条件の差はかなり小さく、500 match rotation eval でも branch ごとの明確な改善証拠は弱かった**。

- discard 改善だけが効いている、とは言いにくい
- optional 改善だけが効いている、ともまだ言いにくい
- 両 branch の相乗効果もはっきりは見えない

したがって現時点では、

- mixed PPO の **安定化には成功**している
- しかし 20 cycle PPO による **branch 単位の改善量はまだ小さい**

と整理するのが自然である。

## 2. 実験目的

`exp_005` で S1 条件が mixed PPO の最小有効条件と判明した。

S1 条件:

- `policy_ratio=0.50`
- `baseline_sample_weight=0.25`
- `policy_anchor.coef=1.0`
- `lr=1e-4`
- `clip_epsilon=0.15`
- `max_grad_norm=0.50`

ただし、S1 run の end-to-end eval が imitation 直後より良かったとしても、
それが

- discard branch の改善なのか
- optional branch の改善なのか
- あるいは両方なのか

は総合 eval だけでは分からない。

そこで `exp_006` では、**再学習なし**で checkpoint を branch 単位に合成し、
discard / optional の寄与を分解して確認した。

## 3. 実行条件

基準 run:

- S1（source run は `experiments/Stage02_CallUnlock/exp_005/run_map.json` の `S1_low_lr_only` を参照）

使用 checkpoint:

- imitation: `checkpoints/checkpoint_imitation.pt`
- final: `checkpoints/checkpoint_cycle_19.pt`

branch namespace:

- discard: `discard_trunk.*`, `discard_head.*`
- optional: `optional_trunk.*`, `candidate_encoder.*`, `optional_scorer.*`
- value: `value_trunk.*`, `value_head.*`

評価条件:

- `evaluation.mode = rotation`
- `evaluation.num_matches = 500`
- `evaluation.num_workers = 10`
- 実質 `2000 match` 相当

実行管理:

- `experiments/Stage02_CallUnlock/exp_006/run_map.json`
- `scripts/local/stage2/exp_006_driver.py`

補足:

- 初版 driver は `reuse-from` を使っていたが、runner の共通 checkpoint preload が Stage1 モデルを先に立てるため state_dict 衝突で失敗した
- 最終版 driver では `run_stage2a_eval_parallel()` を直接呼ぶ方式に変更した

## 4. 対象 run

- II / FI / IF / FF の対応は `experiments/Stage02_CallUnlock/exp_006/run_map.json` を参照

## 5. 主結果

| Condition | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| II | 2.4235 | 25608.8 | 0.24148 | 0.17439 |
| FI | 2.4220 | 25734.25 | 0.24070 | 0.17543 |
| IF | 2.4175 | 25673.9 | 0.24121 | 0.17373 |
| FF | 2.4250 | 25764.0 | 0.24173 | 0.17620 |

参考として、S1 本体の 50-match eval は以下だった。

- imitation 直後: `avg_rank=2.315`, `win_rate=0.2394`
- final: `avg_rank=2.325`, `win_rate=0.2568`

ただし branch-swap の 500-match 結果では、その差が branch ごとに再現される形では観測されなかった。

## 6. 読み取り

### 6.1 discard 単独改善の証拠は弱い

`FI` と `II` を比べると、

- `avg_rank`: `2.4235 -> 2.4220`
- `win_rate`: `0.24148 -> 0.24070`
- `deal_in_rate`: `0.17439 -> 0.17543`

rank はごくわずかに良いが、win_rate と deal-in はむしろ少し悪い。

したがって、**discard final に差し替えるだけで明確に強くなる**とは今回の結果からは言いにくい。

### 6.2 optional 改善の気配はあるが弱い

`IF` と `II` を比べると、

- `avg_rank`: `2.4235 -> 2.4175`
- `win_rate`: `0.24148 -> 0.24121`
- `deal_in_rate`: `0.17439 -> 0.17373`

`avg_rank` と `deal_in_rate` は 4 条件の中で最良だった。
ただし差はかなり小さく、win_rate はほぼ同水準である。

したがって、**optional final の方が少し良さそうな気配はあるが、明確改善と呼ぶには弱い**。

### 6.3 相乗効果は見えていない

`FF` が `FI` と `IF` を両方上回るなら相乗効果を疑いやすいが、今回はそうなっていない。

- `FF avg_rank = 2.4250` で 4 条件中最良ではない
- `FF deal_in_rate = 0.17620` で 4 条件中最良でもない

したがって、**両 branch の同時改善が強く効いている証拠も薄い**。

## 7. 結論

今回の `exp_006` から得られる結論は次の通り。

1. mixed PPO の **安定化**には成功している
2. しかし 20 cycle PPO による **branch 単位の改善量はまだ小さい**
3. discard / optional のどちらが主に伸びているかは、今回の 500-match branch-swap では明確に分離できなかった
4. optional にわずかな改善の気配はあるが、断定できるほどではない

つまり、現段階の Stage02a mixed PPO は

- 壊れない
- 少しは動いている可能性がある
- しかし **「branch ごとに明確改善している」と胸を張って言える段階ではまだない**

という状態である。

## 8. 解釈

この結果は、`exp_005` で確立した S1 条件が

- mixed PPO を stable にするには十分
- ただし PPO の効き自体はやや弱い

ことを示唆している。

したがって次の焦点は、**頂点性能探索ではなく、安定を壊さず PPO の改善量を少し強めること**になる。

具体的には、

- `lr=1e-4` は維持しつつ、少しだけ戻す
- あるいは anchor を少しだけ弱める

といった「安全側から少しだけ強める」方向の探索が自然である。

## 9. 次アクション

`exp_007` では、S1 を baseline にして、

- `lr=1.5e-4`
- `lr=2.0e-4`
- `policy_anchor.coef=0.75`

のような、**改善量を少し強める single-knob 実験**を行う。

目的は、

- mixed PPO の安定性を維持しつつ
- imitation 直後より final が少しでも明確に良くなる条件

を見つけることである。
