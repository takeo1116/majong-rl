# Experiment Report: exp_010

作成日: 2026-04-01  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_010/runbook.md`
- `experiments/Stage02_CallUnlock/exp_010/run_map.json`
- `experiments/Stage02_CallUnlock/exp_009_bugfix/report.md`
- `experiments/Stage02_CallUnlock/exp_009_bugfix/bug_report.md`
- `experiments/Stage02_CallUnlock/exp_010/semantic_eval_a1_cycle19/semantic_eval_imitation_cycle19_summary.md`
- `experiments/Stage02_CallUnlock/exp_010/semantic_eval_a1_cycle19/semantic_eval_final_cycle19_summary.md`

## 1. 要約

`exp_010` は、open-hand bugfix (`CQ-0259`〜`CQ-0261`) 後に
`exp_009_bugfix` の semantic auxiliary 比較を取り直す再実行である。

結論は次の通り。

- **bugfix 自体は効いている**
  - imitation data / selfplay shard の両方で `win_called` が復活した
- ただし、**bugfix 後は `A1_semaux_default` の優位がかなり弱まった**
- `A1` は best cycle では最良だったが、final / tail-5 で明確優位とは言いにくい
- semantic diagnostics では、**`win_called` support は復活したが、semantic head 自体はまだ `win_called` を全く学べていない**
- `terminal_head` は `ron_bystander` 偏重、`yaku_head` は `Riichi` 偏重が強い

したがって、

- `exp_009_bugfix` で見えていた `A1` の優位は、open-hand bug の交絡をかなり含んでいた可能性が高い
- 一方で semantic auxiliary の方向性が完全に否定されたわけではなく、**teacher supply は直ったが semantic head の学習がまだ弱い**

と整理するのが自然である。

## 2. 実験目的

`exp_009_bugfix` では、semantic auxiliary trunk の初回比較として

- `C0_r3_control`
- `A1_semaux_default`
- `A2_semaux_light`

を比較し、1 seed では `A1` が本命という暫定結論だった。

ただしその後、

- rule-based teacher の open-hand shanten / acceptance
- evaluator baseline seat
- encoder の shanten 系 hint

に bug があり、imitation teacher data では `win_called=0` だったことが判明した。

そこで `exp_010` では、

1. bugfix 後に `win_called` support が本当に復活しているか
2. bugfix 後の正しい teacher / feature semantics 上でも `A1` が本命か
3. semantic head が open-hand を含む terminal / yaku を学べているか

を確認した。

## 3. 条件

共通固定:

- A `core_minimal`
- `full observation`
- mixed PPO
- `policy_ratio=0.50`
- `baseline_sample_weight=0.25`
- `policy_anchor.coef=0.75`
- `lr=1e-4`
- `clip_epsilon=0.15`
- `max_grad_norm=0.50`
- `num_cycles=20`
- `imitation_eval.enabled=true`
- `eval_each_cycle=true`
- seed `42`

差分:

### C0: bugfix 後 R3 control

- `model.semantic_aux.enabled=false`
- `training.semantic_aux.enabled=false`

### A1: bugfix 後 semantic aux default

- `model.semantic_aux.enabled=true`
- `training.semantic_aux.enabled=true`
- `policy_projection_dim=16`
- `terminal_loss_coef=0.2`
- `yaku_loss_coef=0.1`

### A2: bugfix 後 semantic aux light

- `model.semantic_aux.enabled=true`
- `training.semantic_aux.enabled=true`
- `policy_projection_dim=16`
- `terminal_loss_coef=0.1`
- `yaku_loss_coef=0.05`

## 4. 主結果

### 4.1 imitation 直後と final

#### C0

- imitation: `avg_rank=2.470`, `win_rate=0.2340`
- final: `avg_rank=2.600`, `win_rate=0.2303`, `deal_in_rate=0.1934`

#### A1

- imitation: `avg_rank=2.550`, `win_rate=0.2206`
- final: `avg_rank=2.545`, `win_rate=0.2220`, `deal_in_rate=0.1975`

#### A2

- imitation: `avg_rank=2.515`, `win_rate=0.2345`
- final: `avg_rank=2.485`, `win_rate=0.2217`, `deal_in_rate=0.1916`

### 4.2 PPO 安定性

#### C0

- `ratio_mean=1.0339`
- `clip_fraction=0.2375`
- `anchor_kl_discard=0.0267`

#### A1

- `ratio_mean=1.0752`
- `clip_fraction=0.2747`
- `anchor_kl_discard=0.0421`

#### A2

- `ratio_mean=1.2378`
- `clip_fraction=0.2520`
- `anchor_kl_discard=0.0981`

### 4.3 best cycle

- `C0`: cycle 16, `avg_rank=2.325`, `win_rate=0.2394`, `deal_in_rate=0.1843`
- `A1`: cycle 15, `avg_rank=2.300`, `win_rate=0.2528`, `deal_in_rate=0.1882`
- `A2`: cycle 4, `avg_rank=2.395`, `win_rate=0.2487`, `deal_in_rate=0.1749`

### 4.4 tail-5 average

- `C0`: `avg_rank=2.487`, `win_rate=0.2358`, `deal_in_rate=0.1900`
- `A1`: `avg_rank=2.493`, `win_rate=0.2321`, `deal_in_rate=0.1970`
- `A2`: `avg_rank=2.505`, `win_rate=0.2376`, `deal_in_rate=0.1906`

## 5. bugfix 後の `win_called` 確認

今回一番大きい確認はここである。

### imitation data

全条件共通で:

- rows: `1,783,037`
- `overall_win_called = 167,377`
- `call_win_called = 37,062`

### cycle_19 selfplay

- `C0`: `overall_win_called = 11,206`, `call_win_called = 2,540`
- `A1`: `overall_win_called = 11,702`, `call_win_called = 2,575`
- `A2`: `overall_win_called = 11,615`, `call_win_called = 2,537`

したがって、`exp_009_bugfix` 以前の

- imitation teacher data で `win_called=0`
- semantic diagnostics でも `win_called support=0`

という状態は、bugfix 後には明確に解消している。

## 6. semantic diagnostics

`A1_semaux_default_bugfix` について、

- `checkpoint_imitation.pt`
- `checkpoint_learner.pt`

を同じ `cycle_19/selfplay` shard 上で評価した。

対象:
- `experiments/Stage02_CallUnlock/exp_010/semantic_eval_a1_cycle19/`

### 6.1 terminal_head

#### imitation checkpoint

- accuracy: `0.4224`
- `win_called` recall: `0.0000` with support `11702`
- `win_menzen` recall: `0.0809`
- `ron_bystander` recall: `0.9197`

#### final checkpoint

- accuracy: `0.4386`
- `win_called` recall: `0.0000`
- `win_menzen` recall: `0.0047`
- `ron_bystander` recall: `0.9999`

読み:

- `win_called` support は復活しているのに、**terminal_head は `win_called` を全く当てられていない**
- PPO 後も `win_called` recall は改善していない
- overall accuracy は少し上がるが、中身は **`ron_bystander` 偏重**が強まっている

### 6.2 yaku_head

#### imitation checkpoint

- micro F1: `0.4486`
- macro F1: `0.0837`
- exact match: `0.3100`

主な per-yaku recall:

- `Riichi`: `0.5685`
- `Yakuhai`: `0.5701`
- その他はほぼ `0`

#### final checkpoint

- micro F1: `0.3355`
- macro F1: `0.0721`
- exact match: `0.1913`

主な per-yaku recall:

- `Riichi`: `0.7978`
- `Yakuhai`: `0.0855`
- その他はほぼ `0`

読み:

- yaku_head は **広く役を学んでいるというより、少数役に偏っている**
- final では特に **`Riichi` への偏り**が強い
- PPO 後に yaku 指標全体としてはむしろ悪化している

## 7. 読み取り

### 7.1 bugfix の効果は本物

今回もっとも大事なのは、teacher / selfplay shard に `win_called` が復活したことだ。

これは、

- open-hand 和了そのものは engine 上可能
- rule-based teacher の open-hand 進行不全が実際に Optional 学習の交絡要因だった

ことを強く支持する。

### 7.2 bugfix 後は `A1` の優位がかなり弱まった

`exp_009_bugfix` では 1 seed 上 `A1` が本命だったが、bugfix 後の `exp_010` では

- final だけ見ると `A2` が最良
- best cycle では `A1` が最良
- tail-5 では大差がなく、むしろ `C0/A2` が少し良い

という結果になった。

つまり、

- **以前の `A1` 優位は bug の交絡をかなり含んでいた可能性が高い**

と読むのが自然である。

### 7.3 semantic aux は完全否定ではないが、まだ弱い

`A1` は best cycle では最良なので、semantic auxiliary の方向性が完全に無意味とは言えない。

ただし今回の diagnostics では、

- `win_called` を全く学べていない
- `terminal_head` は `ron_bystander` に潰れている
- `yaku_head` は `Riichi` 偏重

が見えた。

したがって今の問題は、

- teacher supply は直った
- しかし **semantic head / loss / data balance がまだ十分ではない**

と整理するのが正しい。

### 7.4 control も意味が変わった

`C0` は bugfix 後に imitation から final で明確に悪化した。

これは、teacher / feature semantics が変わったことで、

- 以前の `C0` の振る舞い
- 以前の `A1` との比較差

の両方が再定義されたことを意味する。

よって、今後の判断は `exp_009_bugfix` ではなく **`exp_010` を新しい基準線**として置くべきである。

## 8. 結論

`exp_010` から得られる結論は次の通り。

1. open-hand bugfix は有効であり、`win_called` support は imitation / selfplay で復活した
2. bugfix 後は `A1_semaux_default` の優位がかなり弱まり、1 seed では「本命」とは言い切れなくなった
3. semantic head 自体はまだ弱く、特に `win_called` と open-hand 役筋をほとんど学べていない
4. したがって、次の主課題は multi-seed 化ではなく、**semantic head が open-hand terminal / yaku を学べる設計にすること**である

## 9. 次アクション

1. semantic aux の診断結果を前提に、`terminal_head` / `yaku_head` の学習設計を見直す
2. 特に
   - `win_called` class imbalance
   - `terminal_head` の collapse
   - `yaku_head` の `Riichi` 偏重
   を次の論点として切る
3. `exp_010` を新しい基準線として、次の実験を設計する
