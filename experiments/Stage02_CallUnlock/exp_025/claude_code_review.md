# Stage2a 残ボトルネック レビュー (exp_025)

作成日: 2026-05-03
作成者: Claude Code (Anthropic) / model: Claude Opus 4.7 (1M context)
対象: `experiments/Stage02_CallUnlock/exp_025` の周辺、Stage02b ルール拡張
へ進む前に simplified rule 環境で残っている性能上限ボトルネックの有無
参照ソース:

- `python/mahjong_rl/stage2a_learner.py`
- `python/mahjong_rl/stage2_selfplay_worker.py`
- `python/mahjong_rl/stage2a_parallel.py`
- `python/mahjong_rl/runner.py`
- `python/mahjong_rl/models/stage2a_model.py`
- `python/mahjong_rl/stage2a_evaluator.py`
- `python/mahjong_rl/semantic_eval.py`
- `python/mahjong_rl/outcome_vocab.py`
- `python/mahjong_rl/experiment.py`
- `src/core/environment_state.h` (C++ default)
- `src/rl/reward_policy.cpp`
- `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- `configs/stage1_full_flat_mlp_rule_only_anchor_ppo_baseline.yaml` (比較対象)
- `configs/default_stage1.yaml`
- `experiments/Stage02_CallUnlock/exp_022/claude_code_review.md`
- `experiments/Stage02_CallUnlock/exp_022/report.md`
- `experiments/Stage02_CallUnlock/exp_023/report.md`
- `experiments/Stage02_CallUnlock/exp_024/report.md`
- `experiments/Stage02_CallUnlock/exp_025/runbook.md`

---

## 0. 結論先出し (1 行)

> 最有力ボトルネックは **`reward.point_delta_scale` が Stage2a だけ default 1.0
> のまま (Stage1 は 0.0001)** であり、value/semantic head の学習効率を著しく
> 下げている可能性が高い。これを Stage1 と揃えてから 1seed probe を 1 本通せば、
> 「現環境でまだ伸びるか / もう Stage02b に進むべきか」の見通しは大きく
> はっきりする。

---

## 1. High-impact findings

確信度を `(ほぼ確信 / 有力 / 中)` で明示する。`exp_022` の mixed PPO 問題ほど
大きくはないが、性能上限を下げている可能性のある順に並べる。

### 1.1 (ほぼ確信) **Stage2a の `point_delta_scale` が default `1.0` のまま** — 報酬・value・semantic 全部が「raw 麻雀点 (8000 や 12000)」スケールで学習している

#### 該当箇所

- 設定 default が無い側:  
  [configs/stage2a_core_minimal_mixed_s1_baseline.yaml](configs/stage2a_core_minimal_mixed_s1_baseline.yaml)
  には `reward:` ブロックが存在しない
- config 受け側:
  [python/mahjong_rl/experiment.py:19](python/mahjong_rl/experiment.py#L19) — `reward: dict = field(default_factory=dict)`
- worker 側:
  [python/mahjong_rl/stage2_selfplay_worker.py:25-36](python/mahjong_rl/stage2_selfplay_worker.py#L25-L36) — `build_reward_policy_config({}) → None`
- env 側:
  [python/mahjong_rl/env/stage2_env.py:52-53](python/mahjong_rl/env/stage2_env.py#L52-L53) — `if reward_config: self._env.reward_policy_config = reward_config` (None なら C++ default 使用)
- C++ default:
  [src/core/environment_state.h:27](src/core/environment_state.h#L27) — `float point_delta_scale = 1.0f`
- Stage1 側 (比較):
  [configs/default_stage1.yaml:18](configs/default_stage1.yaml#L18) /
  [configs/stage1_full_flat_mlp_rule_only_anchor_ppo_baseline.yaml:45](configs/stage1_full_flat_mlp_rule_only_anchor_ppo_baseline.yaml#L45)
  — `point_delta_scale: 0.0001`

#### 影響

Stage2a の reward は raw mahjong points のままになっている (例: ron 8000 → reward
≈ 8000)。Stage1 は `0.0001` で正規化して reward ≈ 0.8。

これにより:

1. **value 目標が非常に大きい**。`returns = advantage + value` で value targets
   は ±1000〜±10000 オーダー。
2. **value_loss MSE が初期で 10^6 オーダー**。`value_loss_coef=0.125` を
   かけても 10^5 オーダー残る。
3. **Adam の second moment が value-trunk parameters では桁違いに大きく蓄積**
   される。これは value-trunk 経由のあらゆる勾配 (value 自身 + semantic head)
   の effective learning rate を下げる方向に働く。
4. **semantic head は value_trunk を共有している** ([stage2a_model.py:189-194](python/mahjong_rl/models/stage2a_model.py#L189-L194))。
   `terminal_loss_coef=0.1`, `yaku_loss_coef=0.05` は元々控えめだが、その上に
   Adam の second moment 効果で実効 LR が下がる。

これは観測されている semantic eval の症状と整合する:

- terminal accuracy 0.6157 だが minor class recall 0.0 が多発
- yaku は hit@0.2=0.89 と高い (= ranking は正しい) が recall 0.0〜0.02
- → モデルは「正しい順位」は学べているが「自信を持って 0.5 を超える logit」を
  出せていない。logit magnitude 不足の典型 = semantic head の有効 LR 不足の症状

policy_loss は advantage 正規化で常に O(1) なので、policy_trunk (value_trunk
独立) は影響を受けにくいが、entropy の蓄積を見ると最終的に max_prob_mean ≈ 0.88
で頭打ちしているのは、value 経由で semantic summary が伝わるところで情報量が
落ちているため、と説明可能。

#### 修正案

Stage2a config に Stage1 と同じ reward block を追加するだけで済む:

```yaml
reward:
  type: "point_delta"
  point_delta_scale: 0.0001
```

本変更だけで:

- value_loss が 8 桁下がる
- value-trunk Adam second moment が小さくなる → semantic head の実効 LR が
  上がる → semantic eval の recall / calibration が改善するはず
- policy 自体は advantage 正規化で挙動不変が原則だが、value calibration の
  改善で advantage 推定が良くなり、結果的に policy 性能も上がる可能性が高い

---

### 1.2 (有力) **`gae_lambda=0.0` + `gamma=0.5` が極端に短期かつ高 bias**

#### 該当箇所

- config default:
  [configs/stage2a_core_minimal_mixed_s1_baseline.yaml:56-57](configs/stage2a_core_minimal_mixed_s1_baseline.yaml#L56-L57)
  — `gamma: 0.50, gae_lambda: 0.0`
- 計算側:
  [python/mahjong_rl/stage2a_learner.py:1226-1258](python/mahjong_rl/stage2a_learner.py#L1226-L1258) — `_compute_returns_advantages`

#### 影響

`gae_lambda=0.0` のため advantage = TD-error 1 発:

```text
delta_t = r_t + 0.5 * V_{t+1} - V_t
adv_t   = delta_t  (lambda=0 なので bootstrap 無し)
```

reward backfill (CQ-0274) で per-decision r は「直近 same-player decision 間の
点数変動」で、ほとんど 0、終局付近だけ非ゼロ。`gamma=0.5` の効果と合わせると、
非終局 sample の advantage は 0.5*V_{t+1} − V_t という value 差分だけになる。

**1.1 で value が miscalibrated だと、advantage 推定そのものがノイズに支配
される**。1.1 の修正後でも、`gae_lambda=0.0` という選択は credit assignment を
極めて短期に絞り、value 関数だけが credit を運ぶ構造に依存する。

なお lambda を上げる場合、value scale の問題 (1.1) を先に解いた方が安全。
gae の bootstrap が長くなるほど value bias の影響が拡散するため。

#### 修正案

1.1 と組み合わせて:

```yaml
training:
  gamma: 0.99
  gae_lambda: 0.95
```

ただし `gamma=0.5` は意図的に短期化していた可能性もあるので、まずは
`gae_lambda=0.5`(中庸) だけ動かして 1seed probe するのが安全。

---

### 1.3 (中〜有力) **`policy_ppo_epochs=1` で「1 cycle で同じサンプルを 1 度しか見ない」運用**

#### 該当箇所

- config default:
  [configs/stage2a_core_minimal_mixed_s1_baseline.yaml:54](configs/stage2a_core_minimal_mixed_s1_baseline.yaml#L54)
  — `epochs: 1`
- 上書き:
  [configs/stage2a_core_minimal_mixed_s1_baseline.yaml:90](configs/stage2a_core_minimal_mixed_s1_baseline.yaml#L90)
  — `rule_mix_learner.policy_ppo_epochs: 1`
- 適用箇所:
  [python/mahjong_rl/runner.py:2050-2053](python/mahjong_rl/runner.py#L2050-L2053)
  / [python/mahjong_rl/stage2a_learner.py:961-994](python/mahjong_rl/stage2a_learner.py#L961-L994)

#### 影響

PPO の典型的な epochs は 4〜10。`epochs=1` は exp_022 の collapse 経験から
「off-policy drift を最小化する」目的で選ばれている可能性が高いが、その分:

- 1 cycle あたりの実 update は ~40 minibatch (sample 数 ~10000 / batch 256)
- 60 cycle 全体でも ~2400 minibatch update
- これは比較的少なく、大きく伸ばす余地が残る可能性がある

PPO ratio 系の collapse は CQ-0282 で baseline sample 除外により大幅に
収まったので、target_kl early-stop と組み合わせれば epochs=2〜4 に上げても
安全な可能性がある。

ただし `policy_ppo_epochs=1` → `2` で約 2 倍時間がかかるため、まずは
diagnostics を見ながら 1seed probe する。

#### 修正案

1.1 / 1.2 を入れた後で、target_kl early-stop (PPO 標準の安全網、CQ-0282
レビューでも提案済み) と一緒に `policy_ppo_epochs=2` か `3` を試す。

---

### 1.4 (中) **value_trunk と semantic_trunk が完全共有**

#### 該当箇所

- model 構造:
  [python/mahjong_rl/models/stage2a_model.py:188-196](python/mahjong_rl/models/stage2a_model.py#L188-L196)
  — `terminal_head = nn.Linear(prev_v, ...)`, `yaku_head = nn.Linear(prev_v, ...)`,
  `semantic_proj = nn.Linear(prev_v, sa_proj)` — すべて `prev_v` (value_trunk
  出口) を共有

#### 影響

意図的設計 (semantic supervision で value trunk を regularize する狙い)
だが、1.1 と組み合わさって semantic head が学習リソースを取れない。

semantic eval 結果:

- yaku Tanyao recall 0.0 / hit@0.2 = 0.90
- yaku Yakuhai recall 0.02 / hit@0.2 = 0.97
- terminal deal_in recall 0.0

これらは semantic head が「順位は分かっているが logit が小さい」状態。
ranking は正しいので、root cause はモデルの容量や特徴ではなく、**学習信号の
強さ**である可能性が高い (= 1.1 の reward scale + semantic_loss_coef の
コンビネーション)。

#### 修正案

1.1 で reward scale を直してから observation。それでも semantic 改善が
鈍ければ、`semantic_aux.terminal_loss_coef` を 0.1 → 0.3、`yaku_loss_coef`
を 0.05 → 0.15 に上げて再確認。

---

### 1.5 (中) **`policy_ratio=0.5` で「policy 席が 0」の match が ~6.25% 発生** — 政策サンプルが 0 件のまま捨てられる

#### 該当箇所

- 席割り当て:
  [python/mahjong_rl/stage2_selfplay_worker.py:368-375](python/mahjong_rl/stage2_selfplay_worker.py#L368-L375)
  — `_assign_seats(seed)`, `[rng.random() < 0.5 for _ in range(4)]`

#### 影響

`policy_ratio=0.5` で 4 席を独立に確率 0.5 で割り当てるため:

- 0 policy seats: (0.5)^4 = 6.25%
- 1 policy seat:  4 × (0.5)^4 = 25%
- 2 policy seats: 6 × (0.5)^4 = 37.5%
- 3 policy seats: 4 × (0.5)^4 = 25%
- 4 policy seats: (0.5)^4 = 6.25%

「0 policy seats」の match は 1 cycle 200 match 中 ~12.5 match。これは
selfplay は走るが PPO 更新には全く寄与しない (separated mode で baseline
sample は除外される)。実効 selfplay match 数が ~7% 削られる。

#### 修正案

軽微な改善。`_assign_seats` で「最低 1 席は policy にする」ガードを入れる
だけで取れる。あるいは config で `policy_ratio_min_seats=1` を導入。
ただし優先度は 1.1〜1.3 より低い。

---

### 1.6 (中) **`round_over` が GAE reset に使われていない** — round 跨ぎで credit が漏れる

#### 該当箇所

- selfplay 側で `round_over` flag を立てるだけ:
  [python/mahjong_rl/stage2_selfplay_worker.py:323-326](python/mahjong_rl/stage2_selfplay_worker.py#L323-L326)
- learner GAE は `terminated` だけ見る:
  [python/mahjong_rl/stage2a_learner.py:1244-1248](python/mahjong_rl/stage2a_learner.py#L1244-L1248)
  — `if t == g - 1 or grp_term[t]: ... last_gae = 0.0`
- `terminated` は match end 時にのみ set される:
  [python/mahjong_rl/stage2_selfplay_worker.py:395-396](python/mahjong_rl/stage2_selfplay_worker.py#L395-L396)

#### 影響

東 1 局終了 → 東 2 局開始の境界で GAE が reset されない。具体的には:

- 同 player の連続 decision 列で:
  - decision_t (東 1 終了直前) → reward = 東 1 終局点数 (e.g., +8000)
  - decision_{t+1} (東 2 開始) → reward = 0
- GAE 計算: `last_gae` が decision_t から decision_{t+1} に持ち越される
- 結果: 東 1 の reward が 0.5*last_gae で東 2 の advantage に微小に漏れる

`gamma=0.5, lambda=0` だと bootstrap 自体は 0 なので、漏れの伝播は1ステップ
で消える。具体的には:
- adv_t = r_t + 0.5*V_{t+1} - V_t
- adv_{t+1} = r_{t+1} + 0.5*V_{t+2} - V_{t+1}

bootstrap は value で吸収されるので、value 関数が round 境界を理解できる
限り、bias は深刻ではない。

ただし 1.2 で `gae_lambda > 0` にする際は `round_over` を境界として reset
すべき。それまでは優先度低。

#### 修正案

1.2 を適用するときに同時に修正。`round_over=True` でも `last_gae=0` reset
する形で十分。

---

### 1.7 (低) `forward_optional` が `candidate_encoder` を 2 回呼ぶ — 純粋な perf

#### 該当箇所

- [python/mahjong_rl/models/stage2a_model.py:444](python/mahjong_rl/models/stage2a_model.py#L444)
  と [python/mahjong_rl/models/stage2a_model.py:460](python/mahjong_rl/models/stage2a_model.py#L460)
  — semantic_aux 有効時、`cand_enc_pre` と `cand_enc` の 2 回計算 (同じ入力)

性能には効かない。修正は機械的だが優先度低。

---

## 2. Experiment recommendations

優先順位とコスト感を付ける。1seed probe で兆候が見えたら 3seed validation
に移る、という前提。

### Priority 1: 1seed probe — `point_delta_scale=0.0001` に揃える (1.1)

最低コストで最大効果が期待できる軸。**Stage02b 判断の前にやらないと損**。

#### 条件
- 全条件は exp_024 と同じ
- `reward.point_delta_scale = 0.0001` だけ追加 (Stage1 と同じ値)
- seed42 のみ、60 cycle

#### 期待される観測
- `learner_metrics.value_loss` が大きく下がる (10^6 → 10^-2 級)
- semantic eval の recall が改善 (deal_in / win_called / Tanyao / Yakuhai
  の recall が 0 → 数 % 〜数十 %)
- avg_rank も小幅 (0.02〜0.05) 改善する可能性が高い

#### 判断基準
- semantic recall が改善 + final/best avg_rank が exp_024 同等以上
  → 採用、3seed 化
- semantic は改善するが avg_rank が動かない → 採用、ただし「現アーキでは
  policy ボトルネックは別所」と見て 1.2 / 1.3 に進む
- どちらも改善しない → 想定外。報告

### Priority 2: 1seed probe — `gae_lambda=0.5` (1.2)

P1 採用後の続き。

#### 条件
- P1 (`point_delta_scale=0.0001`) 採用後の config を baseline
- `gae_lambda: 0.5` (gamma は 0.5 据え置き)
- seed42 のみ

#### 期待される観測
- advantage の variance が下がる
- early/middle cycle (cycle10〜30) の `tail10` 改善

#### 判断基準
- final/best が改善 → 3seed 化
- 変わらない or 悪化 → `gae_lambda` 据え置きで P3 / P4 に進む

### Priority 3: 1seed probe — `policy_ppo_epochs=2` + target_kl early-stop

PPO 標準的な「epoch を増やしつつ KL 暴走を防ぐ」セット。

#### 条件
- P1 採用後の config baseline
- `policy_ppo_epochs: 2`
- **要実装**: PPO target_kl early-stop (CQ として切る、`exp_022` レビューでも
  提案済み)

#### 期待される観測
- 1 cycle あたりの実 update が ~2 倍 → 学習速度が上がる
- target_kl で KL >0.02 ぐらいで早期停止 → ratio 暴走しない
- `clip_fraction`, `log_ratio_p99` が許容範囲に収まり続ける

#### 判断基準
- final/best が exp_024 から有意に改善 (final < 2.10) → 3seed 化
- 改善せず + diagnostics 悪化 → `epochs=1` に戻す

### Priority 4: 軽微な diagnostics 拡張 (実装コスト小)

#### 4.1 `value_loss` 系の追加 diagnostics
P1 を入れるとほぼ自動で観測できるが、明示的に:
- `value_pred_mean` / `value_pred_std`
- `return_mean` / `return_std`
- `explained_variance = 1 - Var(return - value_pred) / Var(return)`

これは `exp_022` レビューで CQ-0284 として提案済み。P1 と同じ流れで切れる。

#### 4.2 advantage と value の scale assertion
学習開始時に `|return| > 100` のような不自然な scale を検出してログに警告。
将来の Stage2a config 移行ミスを防ぐガードとして。

### Priority 5: そのうち、ただし P1〜P3 後 (やる場合)

- `_assign_seats` の "最低 1 席 policy" ガード (1.5)
- `round_over` GAE reset (1.6)
- `forward_optional` の `candidate_encoder` 重複呼び出し削減 (1.7)
- semantic_loss_coef の引き上げ (1.4)

---

## 3. Do not prioritize

今は見送ってよいこと:

### 3.1 model architecture 拡張
`value_hidden_dims=[256,128]` で十分大きい。さらに広げる前に 1.1 〜 1.3 を
やる。

### 3.2 anchor 系の再導入
exp_023 / 024 が安定している以上、いま anchor を入れる動機は薄い。
collapse 兆候が再発したら考える。

### 3.3 selfplay temperature schedule
P100 (exp_025) で entropy 低下傾向はあるが、現時点では深刻ではない。
1.1〜1.3 が効くかどうか先に見るべき。

### 3.4 baseline_imitation の再開
`baseline_imitation_epochs=0` のまま。BC を入れると baseline-like な
更新が混ざってまた分布が歪むリスクがあるので、separated 一本でいまは詰める。

### 3.5 Stage02b ルール拡張をいま開始
1.1 を試す前に進むのは「最も効きそうな修正をスキップする」ことになる。
P1 は 1seed probe だけなら 1 run の追加にすぎないので、これだけは
やってから判断する。

---

## 4. Stage02b へ進む条件

### 必要条件 (これを満たさないなら simplified rule で詰める方が先)

1. P1 (`point_delta_scale=0.0001`) が 1seed で `value_loss` を桁で下げる
   ことを確認すること
2. P1 を入れて semantic eval の代表的 recall (`deal_in_recall`, `win_called_recall`)
   が両方とも `> 0.05` (現状 0.0) を達成すること
3. P1 を入れた 3seed run で `final avg_rank` が `≤ 2.13` (exp_024 同等以上)

### 進んでもよい (= もう Stage02a で詰めない) 条件

- P1〜P3 の各 1seed probe で:
  - avg_rank が exp_024 と±0.02 以内 (= 改善しない)
  - 各 diagnostics に明確な悪化要素は無い
  - semantic recall は改善する (= reward scale の問題は解けたが性能上限は
    現アーキの限界)
  - この場合「現 simplified rule + 現アーキの性能上限はここ」と判断して
    Stage02b に進んで OK

### まだ詰める (= Stage02b は保留) 条件

- P1 を入れて avg_rank が 0.05 以上改善 (final ≤ 2.10) → ここを 3seed で
  validate してから次へ
- P2 / P3 のいずれかでさらに 0.03+ 改善する余地がある場合

---

## まとめ (1 行)

> Stage2a は CQ-0282 で大きな構造バグは取れたが、**`reward.point_delta_scale`
> が Stage1 と揃っていない (1.0 のまま)** という config-only の不整合が残って
> いる可能性が極めて高く、これが value 校正と semantic head 学習を抑え込んで
> いる。次は 1seed で reward scale を Stage1 に揃える 1 本だけ走らせれば、
> 「Stage02a でまだ伸びるか / 進むか」の見通しがほぼ確実につく。

---

## 署名

このレビューは Anthropic の CLI コーディング・アシスタント Claude Code が、
ユーザー (takeo1116) の依頼を受けて、リポジトリ内のソースコードと
実験結果のみを直接参照して独立に作成したものです。

- 作成: Claude Code (Anthropic)
- model: Claude Opus 4.7 (1M context)
- 作成日: 2026-05-03
- 対話セッション: ローカルの Claude Code CLI

(本レビューに先立って `experiments/Stage02_CallUnlock/exp_022/claude_code_review.md`
で指摘した「mixed PPO で baseline actor sample を PPO ratio に混ぜている」
問題は CQ-0282 として実装され、`exp_023` で 3 seed × 60 cycle の
tail10 が `2.985 → 2.199` に改善することで validated された。
本レビューはその次の段階で、`point_delta_scale` の Stage1/Stage2a 不整合という
構造の小さなギャップを指摘したものである。)
