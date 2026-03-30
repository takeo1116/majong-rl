# Experiment Report: exp_002

作成日: 2026-03-29  
参照:
- `experiments/Stage02_CallUnlock/exp_002/runbook.md`
- `experiments/Stage02_CallUnlock/exp_002/report.md`

## 1. 要約

`exp_002` は、Stage02a の feature 比較を再開する前に、**PPO で壊れない条件を決める**ための実験として実施した。

今回は A `core_minimal` を固定し、次の 2 条件を比較した。

1. `mixed`
2. `separated`

現時点での結論は明確である。

1. **`mixed` は不採用**
2. **`separated` は採用候補**
3. Stage02a の暫定 baseline PPO 条件としては、まず `separated` を採用するのが妥当

`mixed` は完走こそしたが、後半 cycle で learner 指標が大きく崩れ、eval も悪化した。  
一方 `separated` は 20 cycle を通して learner / eval ともに安定しており、今回の問いに対して十分な差が出た。

## 2. 実験の目的

`exp_001` では、当初 A/B/C の feature 比較を行う予定だったが、実際には PPO scaffold 側の不安定性が先に露出した。

そのため `exp_002` では、feature 条件は A `core_minimal` に固定し、

- `policy_ratio=0.25`
- `baseline_sample_weight=0.5`
- `policy_anchor(reference="imitation_fixed", coef=0.5)`
- `num_cycles=20`

を共通条件として、

- `ppo_mode="mixed"`
- `ppo_mode="separated"`

のどちらが Stage02a long run の暫定 baseline に向くかを検証した。

## 3. 実行条件

共通:

- A `core_minimal`
- `selfplay.imitation_matches = 1000`
- `training.imitation_epochs = 8`
- `training.batch_size = 256`
- `training.lr = 0.0003`
- `training.multi_chunk_imitation.num_chunks = 3`
- `training.multi_cycle.num_cycles = 20`
- `training.multi_cycle.selfplay_matches_per_cycle = 200`
- `training.rule_mix.policy_ratio = 0.25`
- `training.rule_mix_learner.baseline_sample_weight = 0.5`
- `training.policy_anchor.reference = "imitation_fixed"`
- `training.policy_anchor.coef = 0.5`
- `selfplay.num_workers = 10`
- `selfplay.worker_num_threads = 1`
- `evaluation.num_workers = 10`
- `evaluation.worker_num_threads = 1`

比較対象:

### Run A1 `mixed`

- run label: `A1_mixed` （数値は本 report に転記済み）
- `training.rule_mix_learner.ppo_mode = "mixed"`

### Run A2 `separated`

- run label: `A2_separated` （数値は本 report に転記済み）
- `training.rule_mix_learner.ppo_mode = "separated"`

## 4. imitation の確認

PPO 比較の前提として、imitation throughput が改善後の mainline で安定しているかも確認した。

### mixed

chunk timing:

- chunk 0: `data_generation=70.092s`, `learner=146.075s`, `diagnostics=0.880s`, `total=283.643s`
- chunk 1: `72.327s`, `114.016s`, `0.803s`, `254.384s`
- chunk 2: `75.529s`, `109.292s`, `1.134s`, `253.517s`

### separated

chunk timing:

- chunk 0: `67.098s`, `113.991s`, `0.851s`, `247.540s`
- chunk 1: `69.158s`, `110.007s`, `0.806s`, `244.176s`
- chunk 2: `69.695s`, `111.986s`, `0.974s`, `247.974s`

観測:

- いずれの条件でも imitation chunk は現実的な時間で完走した
- `exp_001` 初期 long run に比べて、imitation throughput は大きく改善した
- 今回の本題は PPO 比較なので、imitation 差は二次的だが、少なくとも throughput 側が再び主問題にはなっていない

## 5. mixed の結果

### 5.1 学習の挙動

`mixed` は cycle 前半は一見動くが、後半で明確に不安定化した。

例:

- cycle 00 learner loss: `0.0149`
- cycle 07 learner loss: `4.3097`
- cycle 10 learner loss: `7.4811`
- cycle 15 learner loss: `578.4401`
- cycle 19 learner loss: `7480.6082`

最終 learner summary:

- `policy_loss = 7480.6082`
- `value_loss = 246995.5084`
- `ratio_mean = 413438.4352`
- `ratio_std = 100142915.0858`
- `clip_fraction = 0.4898`
- `anchor_kl_discard = 5.6866`
- `anchor_kl_optional = 0.0057`

### 5.2 eval の推移

- cycle 00: `avg_rank = 2.505`, `win_rate = 0.231`
- final: `avg_rank = 3.45`, `win_rate = 0.048`

eval は後半に向けて崩れ続けた。

### 5.3 解釈

- `baseline_sample_weight`
- advantage 全体一括正規化
- unsafe mixed guard

を入れた後でも、`mixed` は long run ではまだ不安定だった。

特に

- `ratio_mean`
- `clip_fraction`
- `anchor_kl_discard`

の崩れ方から、前回と同様に **discard branch 主導の drift** が残っていると考えるのが自然である。

したがって `mixed` は、Stage02a の暫定 baseline PPO 条件としてはまだ採用しにくい。

## 6. separated の結果

### 6.1 学習の挙動

`separated` は 20 cycle を通して learner 指標が安定していた。

例:

- cycle 00 learner loss: `0.0053`
- cycle 05 learner loss: `0.0069`
- cycle 10 learner loss: `0.0070`
- cycle 15 learner loss: `0.0075`
- cycle 19 learner loss: `0.0081`

最終 learner summary:

- `policy_loss = 0.0081`
- `value_loss = 237430.8150`
- `ratio_mean = 1.0035`
- `ratio_std = 0.4713`
- `clip_fraction = 0.2462`
- `anchor_kl_discard = 0.0688`
- `anchor_kl_optional = 0.0055`

### 6.2 eval の推移

- cycle 00: `avg_rank = 2.55`, `win_rate = 0.230`
- final: `avg_rank = 2.555`, `win_rate = 0.231`

eval は後半で悪化せず、少なくとも今回の規模では十分安定している。

### 6.3 解釈

`separated` に切り替えると、

- learner loss
- PPO ratio
- anchor KL
- eval rank / win rate

のすべてが `mixed` より大幅に安定した。

これは、今回の不安定性が

- feature set
- imitation
- selfplay throughput

ではなく、主に **mixed PPO の更新条件** にあることをかなり強く示している。

## 7. mixed vs separated 比較

### final summary 比較

| 項目 | mixed | separated | 解釈 |
| --- | ---: | ---: | --- |
| `policy_loss` | `7480.6082` | `0.0081` | separated が圧倒的に安定 |
| `ratio_mean` | `413438.4352` | `1.0035` | mixed は完全に drift |
| `clip_fraction` | `0.4898` | `0.2462` | separated の方が自然 |
| `anchor_kl_discard` | `5.6866` | `0.0688` | discard drift は separated で大幅改善 |
| `anchor_kl_optional` | `0.0057` | `0.0055` | optional は元々大差なし |
| `avg_rank` | `3.45` | `2.555` | separated が大幅に良い |
| `win_rate` | `0.048` | `0.231` | separated が大幅に良い |

### 結論

今回の 2 条件比較では、

- `mixed`: 不採用
- `separated`: 採用候補

と判断してよい。

## 8. 今回わかったこと

1. Stage02a の throughput 改善後でも、`mixed` は long run でまだ不安定
2. `separated` にすると、今回の A `core_minimal` 条件では安定して回る
3. 問題の本体は feature 差ではなく PPO 条件差にあった
4. 少なくとも現時点では、Stage02a の暫定 baseline PPO 条件は `separated` が自然

## 9. Stage1 との関係

今回の結果は、「Stage1 では `mixed` が回っていたのに、なぜ Stage02 では `separated` が勝つのか」を整理する材料にもなった。

まず、Stage1 の current best 付近である `exp_070` は `mixed` を採用していた。一方で Stage1 でも、`exp_051` / `exp_052` では baseline imitation と policy PPO を分ける `separated` 的な 2 段学習を試していた。

つまり Stage1 でも両方を試しており、最終的には discard-only 条件では `mixed` が成立していた、という理解が正しい。

ただし Stage02 では事情が違う。

1. Stage02 は optional decision と鳴き後局面を含むため、discard branch 自体の分布が Stage1 より難しい
2. Stage1 の安定 `mixed` は `policy_ratio=0.10` の軽い混合だったが、Stage02 ではそれより複雑な分布を相手にする
3. 今回の `mixed` run でも崩れていたのは主に `anchor_kl_discard` で、optional より discard drift が支配的だった

したがって、今回の結果は「`mixed` が一般に悪い」のではなく、**少なくとも現時点の Stage02 scaffold では `mixed` より `separated` の方が安定**という意味で解釈するのが適切である。

## 10. 採否

### 採用

- `training.rule_mix_learner.ppo_mode = "separated"`

### 不採用

- `training.rule_mix_learner.ppo_mode = "mixed"`  
  理由: 20 cycle でも discard drift を十分に抑えられず、eval が崩れる

## 11. 次アクション

1. `exp_002` の結論として、Stage02a の暫定 baseline PPO 条件を `separated` に決める
2. `exp_001` で止まっていた A/B/C feature 比較を、`separated` 条件で再開する
3. もし将来 `mixed` を再挑戦するなら、それは baseline 条件の比較とは別実験に分離する

## 12. 現時点の判断

`exp_002` の目的は十分に達成できた。

- `mixed` はまだ不安定
- `separated` は安定

したがって、Stage02a の baseline 比較実験を再開する土台としては、**`separated` を採用する**のが妥当である。
