# Stage2a CallUnlock ルール拡張前 最終レビュー (exp_032)

作成日: 2026-05-09
作成者: Claude Code (Anthropic) / model: Claude Opus 4.7 (1M context)
対象: Stage02_CallUnlock 全体。CQ-0282 〜 CQ-0288 までの修正を経て、
     ルール拡張へ進む直前の段階で残っている不具合・設計上の懸念・性能
     ボトルネックを独立に洗い出す。
参照ソース:

- `python/mahjong_rl/stage2a_learner.py`
- `python/mahjong_rl/models/stage2a_model.py`
- `python/mahjong_rl/stage2_selfplay_worker.py`
- `python/mahjong_rl/stage2a_parallel.py`
- `python/mahjong_rl/stage2a_evaluator.py`
- `python/mahjong_rl/runner.py`
- `python/mahjong_rl/call_shard.py`
- `python/mahjong_rl/outcome_vocab.py`
- `scripts/local/stage2/semantic_head_eval.py`
- `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- `tests/python/` 配下の Stage2a 関連
- `docs/CHANGE_QUEUE.md`
- `experiments/Stage02_CallUnlock/exp_026..032/report.md`
- 過去 review: `exp_022/claude_code_review.md`,
  `exp_025/claude_code_review.md`, `exp_027/claude_code_review.md`

---

## 0. 結論先出し (3 行)

> 1. **ルール拡張を止めるほどの Critical bug は見つからなかった**。
>    CQ-0282/0283/0285/0288 で過去の structural bug は順当に潰されている。
> 2. ただし **High severity の懸念が 3 件** ある。いずれもルール拡張で
>    悪化する向きの問題なので、拡張前に対処を検討する価値がある:
>    (i) multi-cycle で Stage2aLearner が毎 cycle 再生成され、Adam の
>    optimizer state (m1/m2) が毎 cycle 0 リセットされる。
>    (ii) `target_kl + skip_minibatch_on_exceed=True + policy_ppo_epochs=1`
>    の組合せが「early stop した瞬間に同 epoch の残り全データを捨てる」
>    挙動になっており、policy_lr 高め条件で 1 cycle あたりの実 update
>    量が大きく減りうる。
>    (iii) `lr_groups` は config 全体に効くため、imitation warmstart phase
>    も同じ `value_semantic: 1e-2` で更新される。意図と一致しているか
>    要確認。
> 3. ルール拡張に伴い必ず触ることになる **hard-coded 前提が 3 箇所**
>    (`RESPONSE_CONTEXT_DIM=3`, `NUM_YAKU=14`, selfplay の `5000` step
>    cap)。今のうちに「将来こう変える」というガードまたはコメントを
>    残しておくと拡張が楽になる。

---

## 1. 重大度順 findings

確信度 / 優先度を `Critical / High / Medium / Low` で示す。
重要なのは下記の通り。

### Critical
**該当なし**。  
構造的 bug (mixed PPO の baseline ratio 混入、reward scale、terminal scale、
`semantic_proj` dead weight) はいずれも閉じている。

### High

#### H-1. multi-cycle で Stage2aLearner が毎 cycle 再生成され、Adam state がリセットされる

ファイル: [python/mahjong_rl/runner.py:2046-2053](python/mahjong_rl/runner.py#L2046-L2053)
(separated path)、[python/mahjong_rl/runner.py:2058-2061](python/mahjong_rl/runner.py#L2058-L2061)
(mixed/simple path)、初期化は [python/mahjong_rl/stage2a_learner.py:97](python/mahjong_rl/stage2a_learner.py#L97)
など。

```python
for ci in range(num_cycles):
    ...
    ppo_learner = Stage2aLearner(             # ← cycle 毎に新規
        config=ppo_lc, model=s2_model, run_dir=cyc_dir)
    _setup_anchor(ppo_learner)
    train_metrics = ppo_learner.train(
        sp_dir, num_epochs=rml_ppo_epochs,
        filter_actor_type="policy")
```

何が問題か:
- `Stage2aLearner.__init__` で `torch.optim.Adam(model.parameters(), lr=...)`
  を作るので、毎 cycle Adam の m1 (1次) / m2 (2次) momentum buffer が **0
  リセットされる**
- model 重みは `s2_model` を使い回すので continuity あり、しかし
  optimizer 内部状態は cycle ごとに「初期 step」状態に戻る
- Adam は最初の数十 step は bias correction が荒く、effective lr が大きく
  振れる。`policy_ppo_epochs=1` で 1 cycle あたり ~50 minibatch しか
  更新しないため、Adam が落ち着く前に毎回リセットされている

なぜ性能・実験妥当性に影響するか:
- 特に `lr_groups.value_semantic=1e-2` のような高 lr 条件では、
  cycle 序盤の Adam bias correction (`1 - β2^t`) で実効 lr がさらに
  振れる。exp_032 系で seed 安定性が弱かったのはこれと相関しうる
- exp_028 〜 exp_032 はすべて同じ pattern で比較しているので相対比較は
  保たれているが、絶対性能の上限は restored Adam 比で控えめになっている
  可能性

修正案 (拡張前推奨):
- `_run_stage2a_multi_cycle` で `Stage2aLearner` を cycle ループの **外**
  で 1 回だけ作り、`learner.train(...)` を毎 cycle 呼ぶ形に変更
- mode 切り替え (separated baseline_imitation → policy_ppo) は引数や
  内部 state で切り替えるか、それぞれ separate persistent learner を持つ
- もしくは Adam state を `state_dict()` で保存・復元する明示 API を入れる

追加すべきテスト:
- 同 model 同 seed で「learner 1 個 × 3 epoch (= 3 cycles 相当)」と
  「learner 3 個 × 1 epoch ずつ」を比較し、parameter trajectory が
  異なることを assert (= 現挙動の確認)
- 修正後は parameter trajectory が等価 (Adam state が保持されている) を
  assert

#### H-2. `target_kl skip=True + policy_ppo_epochs=1` がデータ破棄になりうる

ファイル: [python/mahjong_rl/stage2a_learner.py:2142-2153](python/mahjong_rl/stage2a_learner.py#L2142-L2153)
(discard branch) / [python/mahjong_rl/stage2a_learner.py:2338-2353](python/mahjong_rl/stage2a_learner.py#L2338-L2353)
(call branch)

```python
if self._tk_enabled and approx_kl > self._tk_threshold:
    tk_stop_count = 1
    if self._tk_skip_on_exceed:
        ...
        break  # ← 当該 epoch の残り minibatch を全部捨てる
```

何が問題か:
- target_kl が minibatch k で発火すると、minibatch k と k+1, k+2, …,
  N-1 すべてが該当 branch / 該当 epoch で更新されない (skip パス)
- `policy_ppo_epochs=1` のとき、それは「この cycle で残りデータを使わない」
  と等価
- 例えば policy_lr=5e-4 で entropy がギリギリのときに minibatch 5/50 で
  threshold 超過 → 残り 45 minibatch 分のデータが捨てられる
- `skip_on_exceed=False` でも step 1 回挟んで break するので 1 minibatch
  しか進まない

なぜ性能・実験妥当性に影響するか:
- 1 cycle あたりの実効 update 量が条件依存で大きく揺れる。target_kl
  の発火頻度は「その cycle がたまたま KL 高めの minibatch に当たった」
  という運の要素が混じる
- 1 seed の `final_avg_rank` を見ているとき、ある seed が早期に target_kl
  を踏むほど更新数が減り、性能が seed-沿いにばらつく可能性
- `policy_ppo_epochs >= 2` であれば「1 epoch 早期停止 + 次 epoch で full
  pass」という挽回ができるが、現 default は 1 epoch なので挽回経路がない

修正案:
- 短期: `policy_ppo_epochs=2` 以上に上げて target_kl と組合せる
  (target_kl が 1 epoch で破綻しても 2 epoch 目で残データを舐められる)
- 長期: target_kl 超過時に「残り minibatch の forward だけ続け、
  optimizer step だけ skip」する mode を追加
  (= "soft target_kl"、updates なしでも diagnostics は完備)
- 中期: target_kl の発火頻度を `target_kl_skipped_minibatches /
  total_minibatches` として明示的に summary に出し、データ破棄量が
  大きい cycle を可視化

追加すべきテスト:
- `policy_ppo_epochs=2` で target_kl 発火 → 1 epoch 早期停止 →
  2 epoch 目は最初から実行されること
- 小 batch shard で target_kl を minibatch 2 で発火させ、minibatch 3+ が
  skip されることを step 数で assert (既存 test の延長)

#### H-3. `lr_groups` config が imitation warmstart phase にも適用される

ファイル: [python/mahjong_rl/runner.py:1682-1687](python/mahjong_rl/runner.py#L1682-L1687)

```python
learner_config = self._as_dict()
learner_config["training"]["algorithm"] = "imitation"
...
learner = Stage2aLearner(
    config=learner_config,
    model=s2_model,
    run_dir=run_dir,
)
```

何が問題か:
- `Stage2aLearner.__init__` は `training.lr_groups` を見て optimizer を
  分割する (CQ-0286)。algorithm が imitation でも分岐は同じ
- 結果、imitation warmstart phase (1000 match × 8 epoch のような大きな
  BC) も `policy=5e-4 / value_semantic=1e-2` で更新される
- 特に semantic_aux 有効時、imitation の terminal_loss * 0.1 + yaku_loss
  * 0.05 が value_semantic group (lr=1e-2) を 100x 強い更新で動かす

なぜ性能・実験妥当性に影響するか:
- imitation warmstart で value/terminal/yaku heads が過剰更新された
  状態を起点に PPO が始まるため、PPO 初期 cycle の挙動が config 依存に
  なる (exp 間で imitation も違うので比較が複雑)
- exp_028 / exp_032 で「lr_groups 上げると seed 安定性が弱くなる」傾向が
  見えていたが、PPO だけでなく imitation の不安定化も寄与している可能性
- 意図的にそうしているなら問題ないが、user の最近の議論は「PPO 側の
  lr 律速」が中心で、imitation は対象外と読めるため、要確認事項

修正案:
- `lr_groups.apply_to: ["ppo"]` のような scope 指定を追加し、imitation は
  単一 group (`training.lr`) に固定できるようにする
- もしくは imitation warmstart の `Stage2aLearner` 構築時に明示的に
  `lr_groups.enabled=False` で上書きする (ad-hoc)
- 最低限、現挙動を実装メモか docstring に明示し、user が config を読んで
  気づけるようにする

追加すべきテスト:
- imitation 学習中に optimizer.param_groups の lr が `lr_groups.value_semantic`
  を反映していることを assert (= 現挙動の確認 / ガードのリグレッション
  防止)
- 修正後は scope 指定で imitation は single group になることを assert

---

### Medium

#### M-1. selfplay の `for _ in range(5000):` ハードキャップが silent

ファイル: [python/mahjong_rl/stage2_selfplay_worker.py:161](python/mahjong_rl/stage2_selfplay_worker.py#L161)

何が問題か:
- 5000 を超えると `for` が抜けて、pending sample は flush されるが
  match の状態は残る (engine 側の round が中断状態)
- 現状の simplified rule では超えないが、ルール拡張 (riichi、副露追加、
  途中流局条件追加) で 1 局あたり decision 数が増えると到達しうる
- 到達時に warning も出ず、shard 上は普通の sample として保存される

修正案:
- 上限到達を detect したら logger.warning + 該当 match を sample から
  除外 (`run_id_match_seed` を破棄リストに記録)
- 上限を `selfplay.max_steps_per_match` で config 化

追加すべきテスト:
- 合成 env で stuck → 上限到達 → warning が出ることを capture

#### M-2. `forward_optional` で `candidate_encoder` が semantic_aux 有効時に 2 回計算される

ファイル: [python/mahjong_rl/models/stage2a_model.py:478-494](python/mahjong_rl/models/stage2a_model.py#L478-L494)

```python
if self._semantic_aux_enabled:
    cand_enc_pre = self.candidate_encoder(cand_features)   # 1回目: summary 用
    opt_summary_pre = self._make_optional_summary(cand_enc_pre, ...)
    h_v = self._compute_value_hidden(...)
    semantic = self._compute_semantic(h_v)
    opt_input = torch.cat([policy_features, response_context,
                            semantic["semantic_summary"]], dim=-1)
else:
    opt_input = torch.cat([policy_features, response_context], dim=-1)
h_c = self.optional_trunk(opt_input)

cand_enc = self.candidate_encoder(cand_features)   # 2回目: scoring 用
```

何が問題か:
- semantic_aux 有効時、同 `cand_features` で `candidate_encoder` を 2 回 forward
- 結果は同じ (parameter 同じ) だが計算が無駄
- exp_027 review でも指摘済み

修正案:
- `cand_enc = cand_enc_pre if self._semantic_aux_enabled else self.candidate_encoder(cand_features)` で再利用

追加すべきテスト:
- 修正後 forward 結果が変わらないこと (regression)

#### M-3. `RESPONSE_CONTEXT_DIM = 3` が hard-coded で、ルール拡張で要見直し

ファイル: [python/mahjong_rl/models/stage2a_model.py:16](python/mahjong_rl/models/stage2a_model.py#L16)

```python
RESPONSE_CONTEXT_DIM = 3  # tile_type/34 + rel_seat/4 + menzen_flag
```

何が問題か:
- response_context の構成 (tile_type, rel_seat, menzen) は call decision の
  context を 3 dim で要約
- ルール拡張で (riichi 中フラグ, 既副露メルド数, 場の風など) を追加したい
  ケースが出る → 拡張時に dim 変更が必要
- 旧 checkpoint との互換も壊れる (yaku_head と同じ問題)

修正案 (拡張準備として):
- `RESPONSE_CONTEXT_DIM` を関数 `_response_context_dim()` 経由にして
  config 由来で計算可能に
- もしくは Stage2bModel として明示的に分離 (Stage2a を不変に保つ)

優先度: ルール拡張時に確実に直すため、いま深い修正は不要。docstring
で「拡張時は make_response_context と RESPONSE_CONTEXT_DIM を同時に変える
こと」と明記しておく。

#### M-4. `NUM_YAKU` / `NUM_TERMINAL_CLASSES` が hard-coded で、yaku 拡張時に旧 checkpoint と互換不能

ファイル: [python/mahjong_rl/outcome_vocab.py:13,34](python/mahjong_rl/outcome_vocab.py#L13)

何が問題か:
- 役を増やすと `terminal_head` / `yaku_head` の Linear shape が変わる
- 旧 checkpoint の `yaku_head.weight` (shape `[14, prev_v]`) は新 model
  (例: `[16, prev_v]`) と一致しない → strict load で fail-fast
- CQ-0288 の `load_stage2a_state_dict` helper は `semantic_proj.*` のみ
  drop。yaku 拡張は別軸の compat 問題

修正案 (拡張時):
- `load_stage2a_state_dict` を一般化し、yaku_head の dim 拡張時には
  「旧 weight を初期化 weight の上 14 行に copy + 残りはランダム初期化」
  のような migration helper を入れる
- もしくは yaku 拡張時は旧 checkpoint からの resume を捨て、imitation
  からやり直す方針を明確化

優先度: 拡張時にやれば十分だが、いま「どう migrate するか」を CHANGE_QUEUE
案に書いておくと拡張がスムーズ。

#### M-5. policy_projection_dim 削除の deprecation が silent

ファイル: [python/mahjong_rl/models/stage2a_model.py:218-222](python/mahjong_rl/models/stage2a_model.py#L218-L222)

CQ-0288 で `policy_projection_dim` は無視 (ignored/deprecated) になった
が、user に警告が出ない。古い config をそのまま使い続けても crash しない
ので、`semantic_summary` の dim が縮んだことに気づきにくい。

修正案:
- `__init__` で `if "policy_projection_dim" in sa: logger.warning(...)`

追加すべきテスト:
- deprecation warning が capture されること (`pytest.warns`)

---

### Low

#### L-1. `_ppo_discard_epoch` の `enumerate` 由来 `batch_idx_in_epoch` が未使用

[python/mahjong_rl/stage2a_learner.py:1975](python/mahjong_rl/stage2a_learner.py#L1975) で
`enumerate(range(...))` しているが、ループ本体では `gn_measured` を
`gn_should_measure` に渡しており、`batch_idx_in_epoch` は未使用 (`_ppo_call_epoch`
は `enumerate` を使わない)。挙動上は問題なし、cleanup 案件。

#### L-2. target_kl skip 経路で policy_loss / value_loss / entropy が `*_losses` に append されない

[python/mahjong_rl/stage2a_learner.py:2144-2153](python/mahjong_rl/stage2a_learner.py#L2144-L2153)

skip=True で break する経路では `all_ratios / all_log_ratios / all_max_probs / all_batch_w`
は append されるが、`policy_losses / value_losses / entropies` は append
されない。結果、metrics 上 `policy_loss` の母集団は「実 step したもの」
だが、`ratio_*` 系の母集団は「skip 含む全 forward」となり、母集団が
微妙にずれる。

挙動を変えなくても、docstring か diag schema コメントで「ratio diagnostics
は forward 完了 minibatch、loss diagnostics は実 step minibatch」と
明示しておくと、後の解析者の混乱を減らせる。

#### L-3. eval seed_offset = `ci * 400` がマジックナンバー

[python/mahjong_rl/runner.py:2083](python/mahjong_rl/runner.py#L2083)

```python
eval_metrics = self._run_eval_stage2a(
    run_dir, encoder, seed_offset=ci * 400)
```

`400` は eval matches/seat=50 × 4 seats × 2 倍マージン相当。今のところ
collision していないが、将来 `eval_matches` を増やすと衝突しうる。  
`max(400, eval_total_matches * 2)` のように derive するか、定数として
constants ファイルに上げると将来的に安全。

#### L-4. `_compute_terminal_weights` の Python ループ

[python/mahjong_rl/stage2a_learner.py:99-117](python/mahjong_rl/stage2a_learner.py#L99-L117)

batch あたり O(N) Python ループで Counter / dict 走査。現在の batch=256 では
ms オーダーで問題ないが、将来の batch 拡張で hot-path 化したら numpy 化
の余地あり。性能改善目的のみで急ぎではない。

#### L-5. `Stage2aEvaluator` は temperature config を読まない (argmax 固定)

[python/mahjong_rl/stage2a_evaluator.py:184,218](python/mahjong_rl/stage2a_evaluator.py#L184)

```python
action, _ = select_discard_argmax(out.discard_logits[0], self._mask_buf[0])
...
idx, _ = select_optional_argmax(out.optional_scores[0], cand_mask[0])
```

evaluation はすべて argmax 固定。これは PPO の「決定的評価」として標準的。  
selfplay は temperature 1.0 sampling、eval は argmax で異なる。  
意図的だが、`evaluation.temperature` config を将来追加するなら明示
ガードを入れること。

---

## 2. ルール拡張前に必須で直すもの / 後回しでよいもの

### 必須で対処したい (推奨優先順)

| # | 内容 | 推奨対応 |
|---|---|---|
| **H-1** | multi-cycle で Adam state がリセット | learner を cycle ループ外で 1 回だけ作り、optimizer state を保持する CQ を切る |
| **H-3** | `lr_groups` が imitation にも効いている | `lr_groups.apply_to=["ppo"]` などで scope 化、または挙動を docstring 明記 |
| **M-3** | `RESPONSE_CONTEXT_DIM=3` hard-coded | 拡張時に必ず触るので「ここを変えると yaku_head dim も合わせる」と明記 |
| **M-4** | `NUM_YAKU` 拡張時の checkpoint 互換 | 拡張前に migration helper 設計を CHANGE_QUEUE 案に記載 |

### 後回しでよい (拡張後でも追加できる)

- **H-2** (target_kl + skip + epochs=1): すでに目立った害が出ていないなら、
  拡張後の実験で「`policy_ppo_epochs=2` が効くか」を 1seed probe で確認
  してから判断
- **M-1** (selfplay 5000 step cap): 拡張で 5000 を超えうるか実験的に
  確認してから対処
- **M-2** (`candidate_encoder` 二重計算): 純 perf
- **M-5** (`policy_projection_dim` deprecation): warning 1 行追加で済む
- **L-1〜L-5**: いずれも cleanup / docstring 整備の範囲

---

## 3. 既存実験結果の解釈を変えうる問題

> **重大度: 注意** (実験を破棄するほどではないが、解釈は要修正)

### 3.1 H-1 (Adam state リセット) の影響

- exp_028 〜 exp_032 はすべて同じ pattern (cycle 毎 reset) で比較しているので、
  **相対比較は引き続き有効** (どの条件が良い / 悪いの結論は変わらない)
- しかし「絶対性能の上限」は restored Adam 比で控えめに出ている可能性
- 例: `P5x_VS100x seed42 final 2.105` のような best 値は、optimizer state
  保持版なら +0.05 程度動く可能性 (要確認)

### 3.2 H-3 (lr_groups が imitation にも効く) の影響

- imitation 後の checkpoint がすでに value/semantic head 過剰学習状態に
  なっている可能性
- exp_028 (TERM50x) や exp_032 系の lr_groups 実験は「imitation も含めた
  total 効果」を見ていて、PPO 単独効果ではない
- 解釈変更: 「lr_groups の効果」という表現を「lr_groups + 強化 imitation の
  効果」に読み替える方が正確

### 3.3 H-2 (target_kl 早期停止) の影響

- target_kl を有効化した実験 (exp_032 followup 系) で、cycle ごとの
  実 update 数が条件によって大きく違っている可能性
- `target_kl_skipped_minibatches` / `target_kl_checked_minibatches` の比率
  を cycle ごとに確認すれば、データ破棄量が見える
- exp_032 report で改めて確認することを推奨

---

## 4. 検証してほしい point (拡張前)

実装を変えなくても確認できるもの:

1. exp_032 系 run の `summary.json` で、各 cycle の
   `learner_metrics.ppo_diag.target_kl_skipped_minibatches /
   target_kl_checked_minibatches` の比率を集計
   - **比率が 30% 超え cycle が複数 ある** なら、target_kl がデータ破棄の
     大要因になっている (H-2 が顕在化)
2. 同 summary.json の `learner_metrics.optimizer_lr_groups` を cycle 0/30/59
   で比較 (lr が固定されているか)
3. imitation phase の learner_metrics に `optimizer_lr_groups.value_semantic`
   が `1e-2` で出ていれば H-3 が確定
4. exp_028 を 1 cycle だけ「optimizer 持ち越し」で再実行する小実験 (= H-1
   のサンプリング)

---

## 5. 重大な実装バグ / 設計の見落としの有無

> **直ちに修正が必要な実装バグは無し**

過去レビューで指摘した以下は全て fix 済み:

- mixed PPO で baseline action を PPO ratio に混ぜる問題 (CQ-0282 で
  separated default 化)
- `point_delta_scale=1.0` で reward が raw 点数スケール (CQ-0283 で 0.0001
  に正規化)
- terminal loss の player-round 重複補正が group 数だけ scale 膨張する問題
  (CQ-0285 で `sum / weight_sum`)
- `semantic_proj` が dead weight (CQ-0288 で削除)
- gradient norm diagnostics 不在 (CQ-0284 で追加)
- optimizer lr 全体一括で imitation/PPO/policy/value の更新を同調しか
  できない問題 (CQ-0286 で lr_groups 追加)
- target_kl 不在による policy lr 高め時の暴走 (CQ-0287 で追加)

残っているのはいずれも **「次に効率的な学習を回すための整備事項」** で、
正しさそのものを脅かすものではない。

---

## まとめ (1 行)

> Stage2a は Critical bug 無しの状態でルール拡張に進めるレベルにあるが、
> **multi-cycle Adam reset (H-1)**, **target_kl + epochs=1 のデータ破棄
> (H-2)**, **imitation も lr_groups 適用 (H-3)** の 3 件は、拡張前に
> 「対処するか挙動として明示するか」を決めるのが望ましい。
> hard-coded 前提 (`RESPONSE_CONTEXT_DIM`, `NUM_YAKU`, selfplay step cap)
> は拡張時に必ず触るので、対応方針を CHANGE_QUEUE 案として今のうちに
> 残しておくと拡張作業がスムーズになる。

---

## 署名

このレビューは Anthropic の CLI コーディング・アシスタント Claude Code が、
ユーザー (takeo1116) の依頼を受けて、リポジトリ内のソースコードと
実験結果のみを直接参照して独立に作成したものです。

- 作成: Claude Code (Anthropic)
- model: Claude Opus 4.7 (1M context)
- 作成日: 2026-05-09
- 対話セッション: ローカルの Claude Code CLI

(過去レビュー履歴:
`exp_022/claude_code_review.md` で mixed PPO baseline ratio 問題を指摘
→ CQ-0282 で fix → exp_023 で validated。
`exp_025/claude_code_review.md` で `point_delta_scale=1.0` 問題を指摘
→ CQ-0283 で fix → exp_026 で validated。
`exp_027/claude_code_review.md` で CQ-0285 後の性能悪化原因として
「terminal-driven value_trunk shaping」と `semantic_proj` の dead weight を指摘
→ CQ-0287 (target_kl) / CQ-0286 (lr_groups) / CQ-0288 (semantic_proj 削除)
で順次対応。
本レビューはルール拡張前の 4 回目で、Critical 級の構造問題は出尽くした
状態を確認しつつ、High 級の運用懸念 3 件を提示するもの。)
