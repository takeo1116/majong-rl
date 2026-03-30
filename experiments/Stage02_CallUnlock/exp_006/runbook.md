# Experiment Runbook: exp_006

作成日: 2026-03-30  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_005/report.md`
- `experiments/Stage02_CallUnlock/exp_004/report.md`
- `python/mahjong_rl/models/stage2a_model.py`

## 1. 背景

`exp_005` により、Stage02a mixed PPO の最小有効条件は実質的に S1 であると分かった。

S1 条件:

- `policy_ratio=0.50`
- `baseline_sample_weight=0.25`
- `policy_anchor.coef=1.0`
- `lr=1e-4`
- `clip_epsilon=0.15`
- `max_grad_norm=0.50`

この条件では、mixed PPO は stable に回り、imitation 直後より final eval が良い兆候も見えた。

ただし、**全体性能が良くなったこと**と、
**discard branch / optional branch のどちらが実際に改善したか**はまだ別問題である。

したがって次の確認事項は、

- discard は本当に改善しているか
- optional は本当に改善しているか
- 片方だけ改善してもう片方は学習できていないのではないか

を分解して見ることになる。

## 2. 問い

S1 の final checkpoint において、

1. `discard` branch は imitation checkpoint より改善しているか
2. `optional` branch は imitation checkpoint より改善しているか
3. 両 branch の改善は独立か、相乗的か

## 3. 基本方針

この実験では**再学習は行わない**。

やることは以下のみ。

1. `checkpoint_imitation.pt` を読む
2. final checkpoint を読む
3. branch 単位で state_dict を組み替えた checkpoint を作る
4. その checkpoint を既存 eval にかける

つまり、これは

- 学習実験ではなく
- **checkpoint 合成 + eval 実験**

である。

## 4. 対象 run

基準 run:

- S1（source run は `experiments/Stage02_CallUnlock/exp_005/run_map.json` の `S1_low_lr_only` を参照）

使用 checkpoint:

- imitation: `checkpoints/checkpoint_imitation.pt`
- final: `checkpoints/checkpoint_cycle_19.pt` を第一候補とする  
  存在しない場合は最終 learner/checkpoint を使う

## 5. branch 定義

Stage2a model の parameter namespace は概ね以下に分かれている。

### discard branch

- `discard_trunk.*`
- `discard_head.*`

### optional branch

- `optional_trunk.*`
- `candidate_encoder.*`
- `optional_scorer.*`

### value branch

- `value_trunk.*`
- `value_head.*`

参照:

- `python/mahjong_rl/models/stage2a_model.py`

## 6. 比較条件

### I/I

- discard = imitation
- optional = imitation
- value = final で固定

意味:

- imitation checkpoint を基準にした control

### F/I

- discard = final
- optional = imitation
- value = final で固定

意味:

- **discard 改善の寄与**を見る

### I/F

- discard = imitation
- optional = final
- value = final で固定

意味:

- **optional 改善の寄与**を見る

### F/F

- discard = final
- optional = final
- value = final で固定

意味:

- 実際の final model

## 7. value branch の扱い

今回の主目的は policy branch の寄与分解なので、
**value branch は final に固定**する。

理由:

1. eval の関心は policy 改善にある
2. value を毎回切り替えると、差分要因が増える
3. Stage2a evaluator は policy 出力が主であり、value を見たいわけではない

したがって今回の比較は、

- discard policy
- optional policy

の 2 軸に集中する。

## 8. 評価方法

既存の Stage02a eval をそのまま使う。

想定:

- `evaluation.mode = "rotation"`
- `evaluation.num_matches = 500`
- `evaluation.num_workers = 10`

結果として見る指標:

- `avg_rank`
- `win_rate`
- `deal_in_rate`

## 9. 読み方

### ケース 1: `F/I > I/I`, `I/F ≈ I/I`

解釈:

- discard は改善している
- optional はまだほぼ学習できていない

### ケース 2: `I/F > I/I`, `F/I ≈ I/I`

解釈:

- optional は改善している
- discard はあまり改善していない

### ケース 3: `F/I > I/I` かつ `I/F > I/I`

解釈:

- 両 branch とも改善している

### ケース 4: `F/F > F/I` かつ `F/F > I/F`

解釈:

- 両 branch の相乗効果がある

### ケース 5: `F/I` は良いが `I/F` は悪い

解釈:

- optional branch はまだ改善余地が大きい

## 10. この実験の意義

この実験が重要なのは、次の判断に直結するためである。

### そのままルール拡張してよいケース

- discard / optional の両方が改善している
- もしくは optional が少なくとも悪化していない

### optional をもう少し詰めるべきケース

- discard は改善している
- optional は `I/F ≈ I/I`
  あるいは悪化する

この場合、全体としては前進していても、optional branch 自体はまだ弱いと判断できる。

## 11. 実装方針

この runbook では、実行用の driver を追加して実施する。

管理ファイル:

1. `scripts/local/stage2/exp_006_driver.py`
   - imitation / final から branch 単位で state_dict を合成する
   - 条件ごとに eval 専用 run_dir を新規作成する
   - 合成 checkpoint をその run_dir 配下に保存する
   - `run_stage2a_eval_parallel()` を直接呼んで I/I, F/I, I/F, F/F を順に評価する
2. `experiments/Stage02_CallUnlock/exp_006/run_map.json`
   - 各条件の合成 checkpoint と eval run を記録する
3. `experiments/Stage02_CallUnlock/exp_006/run_map.json`
   - 各条件の branch source と eval 指標を記録する

補足:

- `reuse-from` は使わない
- 理由は、現行 runner の共通 checkpoint preload が Stage1 モデル前提で先に走り、
  Stage2a checkpoint を eval-only reuse に使うと state_dict 衝突を起こすため
- 今回は driver が直接 Stage2a evaluator を呼ぶことでこの問題を回避する

実行コマンド:

```bash
./.venv/bin/python scripts/local/stage2/exp_006_driver.py
```

別の source run を使う場合:

```bash
./.venv/bin/python scripts/local/stage2/branch_swap_eval.py \
  --source-run <source-run-dir> \
  --config configs/stage2a_core_minimal_mixed_s1_baseline.yaml \
  --out-dir experiments/Stage02_CallUnlock/exp_006 \
  --experiment-prefix stage2a_exp006 \
  --eval-matches 500 \
  --eval-workers 10 \
  --eval-seed-start 400000
```

smoke 用:

```bash
EXP006_ONLY=II EXP006_NUM_MATCHES=1 EXP006_NUM_WORKERS=1 ./.venv/bin/python scripts/local/stage2/exp_006_driver.py
```

## 12. 成功判定

この実験自体の成功は、

- 4 条件の checkpoint 合成が正しくできる
- eval が完走する
- discard / optional の寄与について、少なくとも方向性のある結論が出る

ことである。

## 13. 次アクション判定

### discard / optional の両方が改善していた場合

- S1 mixed baseline を前提に、そのまま部分ルール拡張へ進む

### discard だけ改善して optional は弱い場合

- optional branch の改善を追加テーマとして切る
- ただし全体が十分前進しているなら、ルール拡張を先に進める選択肢もある

### optional が悪化している場合

- optional learner / imitation / candidate scoring を見直す
- 完全麻雀化の前に optional branch の sanity check を追加する
