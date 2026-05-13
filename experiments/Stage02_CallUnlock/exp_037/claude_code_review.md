# Stage02b Optional Unlock 性能低下 独立レビュー (exp_037)

作成日: 2026-05-11
作成者: Claude Code (Anthropic) / model: Claude Opus 4.7 (1M context)
対象: CQ-0290〜CQ-0295 までを経た optional action 開放後の性能低下原因究明。
     `exp_036` (optional-all 60cycle) と `exp_037` (family ablation 30cycle)
     が `exp_034` (optional-off baseline) より明確に劣る現象について、
     コード/データから独立に判断する。
参照ソース:

- `python/mahjong_rl/env/stage2_env.py`
- `python/mahjong_rl/env/response_candidate.py`
- `python/mahjong_rl/stage2_selfplay_worker.py`
- `python/mahjong_rl/stage2a_evaluator.py`
- `python/mahjong_rl/stage2a_parallel.py`
- `python/mahjong_rl/stage2a_learner.py`
- `python/mahjong_rl/models/stage2a_model.py`
- `python/mahjong_rl/legal_mask.py`
- `python/mahjong_rl/candidate_encoding.py`
- `python/mahjong_rl/call_shard.py`
- `python/mahjong_rl/optional_family_audit.py`
- `python/mahjong_rl/runner.py`
- `experiments/Stage02_CallUnlock/exp_034/report.md`
- `experiments/Stage02_CallUnlock/exp_036/report.md`
- `experiments/Stage02_CallUnlock/exp_036/optional_family_audit_final_20k/optional_family_audit_exp036_final_20k_summary.md`
- `experiments/Stage02_CallUnlock/exp_037/report.md`
- `runs/20260510_20260510_stage2b_optional_all_probe_seed42_f5e9ee5a/config.yaml`
- `runs/20260510_20260510_stage2b_optional_all_probe_seed42_f5e9ee5a/summary.json`
- `docs/CHANGE_QUEUE.md` (CQ-0280, CQ-0290〜CQ-0295)

---

## 0. 結論先出し (3 行)

> 1. **致命的な実装バグは見つからなかった**。Critical 級の挙動不整合は無く、optional unlock の forward / reward attribution / target_kl gating はすべて期待どおり動いている。
> 2. **性能低下の主要因は「optional_riichi 有効時の wide discard mask が auto-riichi の強制を解いた」こと**。exp_034 baseline では mask が立直牌のみに縮退しており「テンパイ即立直」が engine 側で強制されていたが、optional-all ではポリシーがディフェンス牌を選びうるため、政策が auto-riichi の振る舞いを再学習するまで複数 cycle を要する (exp_036 cycle0 riichi=1000 → cycle59 riichi=1503)。
> 3. **副次的に Tsumo/Ron/Ankan/Kakan/Kyuushu の 5 family は決定的すぎて学習信号がほぼゼロ** (audit で全 6 binary family が 100% agreement / entropy ≈ 0)。これらは imitation/PPO 計算枠を消費しながら勾配にはほぼ寄与せず、advantage normalization の denominator を緩く下げ、discard 側の有効勾配を希釈している。
> 4. (おまけ) `make_ankan_optional_candidates` に **潜在バグ** がある (`//4` 欠落)。現在は engine 側が偶然 tile_type を直接渡してくれて事なきを得ているが、convention 変更で即クラッシュする。

優先度の高い対処は CQ 候補 A〜D に整理して §3 で示す。

---

## 1. Findings

### 1.1 High 級

#### Finding H1 — 6 種 binary optional family が「決定論的すぎて」勾配バジェットを浪費している

##### 根拠 (audit)

`experiments/Stage02_CallUnlock/exp_036/optional_family_audit_final_20k/optional_family_audit_exp036_final_20k_summary.md` 20k sample 抜粋:

| family | samples | teacher positive | policy agreement | entropy mean | max_prob mean | teacher_action_prob mean |
|---|---:|---:|---:|---:|---:|---:|
| riichi  | 246 | 1.000 | **1.000** | ~0     | 1.000 | 1.000 |
| tsumo   |  83 | 1.000 | **1.000** | ~0     | 1.000 | 1.000 |
| ron     | 248 | 1.000 | **1.000** | ~0     | 1.000 | 1.000 |
| ankan   |  69 | 0.000 | **1.000** | ~0     | 1.000 | 1.000 |
| kakan   | 164 | 0.000 | **1.000** | 0.005  | 0.999 | 0.999 |
| kyuushu |   4 | 0.000 | **1.000** | 0.181  | 0.954 | 0.954 |

集計 cycle 59 では 6 family すべてで policy が teacher と完全一致しており、エントロピーは実質ゼロ。

##### 問題

`python/mahjong_rl/stage2a_learner.py:1207-1242` の PPO ループは family に依らず call branch を一律に処理する。`_ppo_call_epoch` ([stage2a_learner.py:2275-2503](python/mahjong_rl/stage2a_learner.py#L2275-L2503)) は applied minibatch ごとに:

- `optional_trunk` forward
- `optional_scorer` forward (B, C, ...)
- `value_head` forward (`out.values["round_delta"]`)
- semantic-aux (terminal/yaku) forward — share する value_trunk hidden 経由
- `backward()` + `optimizer.step()`

を実行する。binary 6 family は **政策が teacher と完全一致** しているため:

- `ratio ≈ 1.0` → `policy_loss` の surrogate min は ≈ -|advantage|
- 但し log_ratio ≈ 0、`advantage_normalize` 後の advantage は他の sample に支配される
- `value_loss`: 直前 discard sample と同じ engine state → value target も同程度 → value_loss も小さい
- `entropy ≈ 0` (saturated)、`entropy_coef = 0.0` (config) のため寄与なし

つまりこれらの sample は **PPO の有効勾配にはほぼ寄与しないにも関わらず**:

- 行ごとの forward/backward 演算コスト (約 5,200 sample/cycle、1 epoch あたり ~20 minibatch)
- `advantage_normalize` の `mean` / `std` の denominator
- imitation 8 epoch ([config.yaml:93](runs/20260510_20260510_stage2b_optional_all_probe_seed42_f5e9ee5a/config.yaml) `imitation_epochs: 8`) × 5,200 sample 分の crossentropy

を消費する。

##### 影響度

cycle ごとの family 分布 (exp_036 cycle 59 summary.json から):

```text
discard: 95,590   response: 25,484
riichi: 1,503   tsumo:  498   ron: 1,584
ankan:    344   kakan: 1,252  kyuushu: 25
optional_decision_count: 5,206
```

総 sample (~152k) に対して 6 family 合計 5,206 = 約 3.4%。3.4% は単独では小さく見えるが、それらが **0 情報** なので advantage normalization と value loss の希釈効果が積み重なる。

### 1.2 High 級 / 仕様変更による副作用

#### Finding H2 — Trajectory に no-op transition が挿入されることで GAE と advantage normalization が微妙にずれる

`python/mahjong_rl/env/stage2_env.py:237-256` の `_SELF_ACTION_OPTIONAL_SKIP_KEYS` 経路では、TSUMO / ANKAN / KAKAN / KYUUSHU の Skip 選択時に **engine step を発行せず** `_auto_advance` を再走する。worker は `step_counter += 1` ([stage2_selfplay_worker.py:434, 499, 568](python/mahjong_rl/stage2_selfplay_worker.py)) を実行するが、engine 側 reward は 0 のまま。

`_compute_returns_advantages` ([stage2a_learner.py:1588-1620](python/mahjong_rl/stage2a_learner.py#L1588-L1620)) は `(episode_id, player_id)` 単位で step_id 順に GAE を計算する。Skip optional sample は:

- `reward = 0`
- `value(t)` と `value(t+1)` (次の DISCARD) が同じ engine state を見ているため近い値
- `delta ≈ 0 + γ V(t+1) − V(t) ≈ 0`

`gae_lambda = 0.0`, `gamma = 0.5` ([config.yaml:91-92](runs/20260510_20260510_stage2b_optional_all_probe_seed42_f5e9ee5a/config.yaml)) の下では GAE は単純 TD-error に縮退する。Skip optional sample の advantage は ≈ 0 になるだけなので、**advantage 値そのものは大きく歪まない**。

しかし `_compute_ppo_branch_targets` ([stage2a_learner.py:1554-1558](python/mahjong_rl/stage2a_learner.py#L1554-L1558)) の **全体 advantage normalization**:

```python
all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)
```

の `mean` / `std` には ~800 個の ≈ 0 sample が混ざる。これは:

- `mean` を 0 方向に弱く引く
- `std` を狭める (≈ 0 sample が多いほど分散が小さくなる)

結果として **discard family の advantage が事実上スケーリングされる**。`gae_lambda=0.0` 設定では絶対値の差はわずか (1〜数%) だが、PPO の clip 判定 (`abs(ratio-1) > clip_epsilon`) と policy gradient の符号付き寄与には影響する。これは Critical bug ではないが、optional-on と optional-off で **「同じ discard sample が違う scale で学習される」** ことになる微妙な原因。

#### Finding H3 — `optional_riichi_enabled=True` の wide discard mask が auto-riichi を解除し、政策の初期 trajectory 品質を著しく落とす **(最も支配的な性能低下要因)**

##### 仕様の変化

- `optional_riichi=False` (exp_034 既存):
  - `make_discard_mask_from_legal_actions(actions, include_all_discards=False)` ([legal_mask.py:21-57](python/mahjong_rl/legal_mask.py#L21-L57)) は **立直可能局面で立直牌だけを mask に含める**。
  - 結果として政策は「テンパイで discard する」=「立直を宣言する」を engine レベルで強制される。
  - 政策が学習する必要すらない。

- `optional_riichi=True` (exp_036/037):
  - `get_legal_mask` ([stage2_env.py:260-273](python/mahjong_rl/env/stage2_env.py#L260-L273)) は `include_all_discards=True` で **全 discard tile_type** を含める。
  - 政策がテンパイ局面でディフェンス牌 (= 立直しない選択) を選びうる。
  - 政策がディフェンス牌を選ぶと `_maybe_open_riichi_optional` ([stage2_env.py:316-352](python/mahjong_rl/env/stage2_env.py#L316-L352)) は発火しない (riichi 形でない tile_type には riichi action が無い)。
  - 結果: 立直チャンスを bypass する。riichi opportunity diagnostics (CQ-0294 で導入) はこれを `riichi_bypassed_by_non_riichi_discard_count` でカウントする。

##### 影響 (exp_036 summary.json cycle ベース実測)

```text
cycle  0: num_rounds 2193, riichi 1000 (1000/2193 ≈ 0.456 per round)
cycle 59: num_rounds 2130, riichi 1503 (1503/2130 ≈ 0.706 per round)
```

つまり **政策が auto-riichi 相当の振る舞いを獲得するのに 60 cycle 近く要している**。exp_034 では engine 強制で cycle 0 から最大値を取れていたはず。

立直しないとリーチ報酬 (供託・一発・裏ドラ・ツモ和了点数) を取り損なうため、**初期 cycle の trajectory 品質が劣化**する。劣化した trajectory で imitation/PPO が回るため、回復は鈍い。

##### evaluator 側の連動

`Stage2aEvaluator._policy_discard` ([stage2a_evaluator.py:240-258](python/mahjong_rl/stage2a_evaluator.py#L240-L258)) も同じ wide mask を使う。eval 時にも政策は立直を bypass しうる → eval 成績が cycle 後半まで上がりにくい。baseline seat は CQ-0294 で `get_teacher_discard_mask_from_snapshot` ([stage2_env.py:288-305](python/mahjong_rl/env/stage2_env.py#L288-L305)) を使う = riichi-only mask = 旧 auto-riichi 相当の挙動なので、ここはフェアだが、`policy_seat` のみハンディキャップを受ける形になっている。

##### exp_036 と exp_037 の挙動差

exp_036 は **CQ-0294 前** のため、teacher_top1_index も wide mask 上で計算されていた = `rule_based teacher` 自体も「立直しない選択」を提案しうる状態だった。これは imitation 段階でも auto-riichi を強制しないことを意味する。

exp_037 は CQ-0294 後で baseline/teacher は riichi-only mask に分離されているが、それでも policy mask 自体は wide のまま。教師サンプル空間と政策行動空間にズレがある状態で PPO が回る。

これが exp_036 と exp_037 のどちらも exp_034 を超えない理由として **最有力**。

### 1.3 Medium 級

#### Finding M1 — `RON_OPTIONAL` 発火時、Chi/Pon/Daiminkan の選択肢が消える

`python/mahjong_rl/env/stage2_env.py:561-579` で `optional_ron_enabled=True` かつ Ron が legal の場合、`RESPONSE` ではなく `RON_OPTIONAL` を発火する。`make_ron_optional_candidates` ([response_candidate.py:188-223](python/mahjong_rl/env/response_candidate.py#L188-L223)) は `[Ron, Skip]` の 2 候補のみを返すため、**同時に Chi/Pon/Daiminkan が legal だった場合、それらは政策から見えなくなる**。

```python
# response_candidate.py L194-200 のコメント
# ResponsePhase で Ron が合法なときに使う。Ron 候補のみ + Skip という
# binary optional に絞る (Pon/Chi/Daiminkan が同時に合法な稀なケースは
# Ron を選ぶか Skip を選ぶかに簡略化。バッチ 3 以降で再検討)。
```

実害は限定的 (Ron > Pon はほぼ常に正しい) だが、**action space の縮退** であり exp_034 baseline との不公平要素。

#### Finding M2 — exp_036 は CQ-0294 *前*なので、ベースラインとの比較条件が後の実装と異なる

exp_036 は 2026-05-10 実施。CQ-0294 (teacher/baseline mask 分離 + riichi_discard_mask + optional summary 拡張) は 2026-05-11 実装 (`docs/CHANGE_QUEUE.md` の CQ-0294 ログ参照)。

exp_036 のときの worker:

- `baseline.select_discard(hand_ids, mask, meld_count=mc)` の `mask` は wide mask
- → baseline 対戦相手 (selfplay の他 3 席) も auto-riichi しない
- → 全 4 席が「立直しがちな政策」ではなく「立直しない可能性が高い政策」に学習

これは exp_037 (post-CQ-0294) で fix されたが、exp_037 も exp_034 を超えていない。つまり **exp_036 と exp_037 のギャップは H2/H3 で説明できるが、exp_037 と exp_034 の残差は別の要因 (主に H3 の policy 側 wide mask) が支配的**。

#### Finding M3 — Imitation epoch が決定論的 family の label を 8 回繰り返し学習させている

`config.yaml:93` で `imitation_epochs: 8`。`_imitation_call_epoch` ([stage2a_learner.py:1005-1070](python/mahjong_rl/stage2a_learner.py#L1005-L1070)) は family を区別せず call branch 全体を CE で学習する。

audit が示すとおり、binary 6 family は **1 epoch でほぼ 100% 飽和**するため、残り 7 epoch は完全に冗長。蓄積誤差は無いが、**imitation 後の policy が `value_trunk` の他成分を学習する余地を奪う** 可能性がある。

8 epoch 強い CE 圧力が optional family の Skip/Action に集中 → `value_trunk` の hidden 表現がそれらに過適合 → discard branch / response branch の value 推定品質が相対的に低くなる、というシナリオ。

検証可能: CQ-0295 の learner family-level diagnostics (`ppo_diag["decision_family"][fam]["approx_kl_mean"]` 等) を cycle 別に追えば、imitation 後の cycle 1 PPO で family 別 KL がどの程度動くかで分かる。

#### Finding M4 — `selected_candidate_index` と `teacher_top1_index` の意味分離

binary optional family では:

- `selected_candidate_index = idx` (政策が実際に選んだ index)
- `teacher_top1_index = 0 or 1` (worker でハードコード)

PPO は `selected_candidate_index` を action として使い ([stage2a_learner.py:2305-2320](python/mahjong_rl/stage2a_learner.py#L2305-L2320))、imitation は `teacher_top1_index` を target として使う ([stage2a_learner.py:1042-1048](python/mahjong_rl/stage2a_learner.py#L1042-L1048))。

`policy_ratio=1.0` で政策がほぼ teacher と一致している現状では両者は事実上等しい。だが将来 teacher を非自明な (例: 高度な Tsumo skip 戦略) ものに置き換えるときに **両者の意味の食い違いが PPO と imitation を逆方向に引く**リスクがある。今は問題ないが、設計メモとして残すべき。

### 1.4 Low 級 / 潜在バグ・コード衛生

#### Finding L1 — `make_ankan_optional_candidates` の `tile_type` に `//4` が抜けている (潜在バグ、現状偶然動く)

`python/mahjong_rl/env/response_candidate.py:268`:

```python
tile_type = int(ankan_action.tile) if ankan_action.tile < 255 else -1
```

隣接する `make_kakan_optional_candidates` ([response_candidate.py:286](python/mahjong_rl/env/response_candidate.py#L286)):

```python
tile_type = (kakan_action.tile // 4) if kakan_action.tile < 255 else -1
```

ankan のみ `//4` が無い。実機 (`mahjong_rl._mahjong_core.Action`) で確認:

- Ankan action: `.tile = 16, 23, 33, ...` (= tile_type 値)
- Kakan action: `.tile = 55` (= tile_id, 55//4 = 13 = tile_type)

つまり **engine が Ankan と Kakan で異なる convention** を採用しており、Ankan のみ `action.tile` が既に tile_type 値。だから現状の `int(ankan_action.tile)` は偶然 0-33 範囲に収まっている。

リスク:

- `CandidateEncoder.tile_emb` は `nn.Embedding(35, 8, padding_idx=0)` ([stage2a_model.py:149](python/mahjong_rl/models/stage2a_model.py#L149)) なので index >= 35 で `IndexError` (実測確認済)。
- engine 側で Ankan の `.tile` を将来 tile_id に揃える変更が入ったら、ankan candidate を含む forward が **CUDA assertion / CPU IndexError で即クラッシュ**する。

対応案:

```python
raw = int(ankan_action.tile) if ankan_action.tile < 255 else -1
tile_type = raw if 0 <= raw < 34 else (raw // 4 if 0 <= raw < 136 else -1)
```

加えて unit test で「engine 出力の Ankan `.tile` が 0-33 範囲」を assert すれば、convention 変更時に即検知できる。

#### Finding L2 — `step_response` が SelfActionPhase optional の primary action 経路で `candidates[0]` 決め打ち

`python/mahjong_rl/env/stage2_env.py:254-256`:

```python
# primary action: 通常通り engine step
action = self._response_candidates[0].action
return self._execute_and_advance(action)
```

現状 `_SELF_ACTION_OPTIONAL_SKIP_KEYS` 経路の candidate は `[primary, Skip]` の 2 候補のみで、`candidate_index != 1` なら `==0` を意味するため動作する。だが将来 3 候補以上 (例: 複数 Ankan / 複数 Kakan を提示) が追加されたら、`candidate_index=2` でも `[0]` が実行される silent bug になる。

修正コスト極小 (`candidates[candidate_index].action` に書き換えるだけ)。

#### Finding L3 — Skip candidate の `.action` フィールドが misleading

`response_candidate.py:248`:

```python
action=(skip_action if skip_action is not None else primary_action),
```

`skip_action is None` の場合 (SelfActionPhase optional: TSUMO/ANKAN/KAKAN/KYUUSHU)、Skip 候補の `.action` は **primary action object のまま** 入る。env はこれを実行しないので動作上は問題ないが、shard を offline で見たときに「Skip サンプルなのに action=Ankan オブジェクト」が記録される。デバッグ時の混乱要因。

`None` を保持し読み出し側で defensive にする方が clean。

#### Finding L4 — Ankan/Kakan/Kyuushu の家ごとの `_optional_skipped_this_turn` クリアタイミング

`_execute_and_advance` ([stage2_env.py:432-460](python/mahjong_rl/env/stage2_env.py#L432-L460)) line 436:

```python
self._optional_skipped_this_turn = set()
```

任意の engine step が走るタイミングで skip set を空にする。これは SelfActionPhase 内での連鎖判定 (TSUMO 後すぐ Ankan も legal、など) を考えると正しい挙動。確認のみ。問題なし。

---

## 2. Most Likely Cause

**Confidence: High** — exp_036/exp_037 の性能劣化の支配的要因は **H3 (wide discard mask による auto-riichi 失効) + H1 (決定論 family による勾配バジェット浪費)** の組み合わせ。

### 根拠の要約

1. audit で 6 binary family すべてが **policy_top1_agreement=1.000 / entropy≈0** → 政策は family を完全に解いている → family 自体のバグ・学習失敗ではない。
2. discard family も `best_set_agreement=0.984` (top1 は rule_based の best-set に 98.44% 含まれる) → discard 政策の質は rule_based と競合する水準 → discard 学習自体が壊れてはいない。
3. cycle 0 vs cycle 59 で riichi 回数が 1000 → 1503 と単調増加 → 政策は **「立直すべき」** を slow に再学習中 → 60 cycle では完全には追いつかない。
4. exp_034 baseline は engine が立直を強制 → cycle 0 から最大 riichi 率 → trajectory 品質が高い → eval 成績が高い。
5. exp_037 family ablation の単独 family run はどれも 30 cycle で exp_034 30-cycle (tail10=2.215) を超えない → 単一 family の bug ではない → 共通要因 (wide mask + 決定論 family の orgaisation 圧) が支配的。

### 排除した可能性 (コード読みで確認)

- **値分配バグ**: `_accumulate_pending_rewards` ([stage2_selfplay_worker.py:757-775](python/mahjong_rl/stage2_selfplay_worker.py#L757-L775)) は engine reward を action-side optional sample に正しくルートし、skip-side では 0 を保存して直後の DISCARD sample に reward が乗る。
- **tile_emb OOR**: L1 は **現状** 引き金を引かない (engine 出力が 0-33)。クラッシュ報告も無い。
- **target_kl による optional skip**: CQ-0293 gating は per-family の applied counts を正しく算出 (`test_stage2a_cq0293_applied_diagnostics`)。
- **value trunk shared corruption**: skip optional と直後 DISCARD は同じ engine state → value target も近い → 矛盾信号は発生しない。
- **CQ-0294 の `_make_optional_summary` 拡張**: 4 → 11 行への拡張、`load_stage2a_state_dict` の挿入 zero column 処理は test で確認済み。

### 数値的な確認

exp_036 cycle 59 stats:
- discard: 95,590 / response: 25,484 / 6 optional family 合計: 5,206
- 6 family は全て 100% 一致 (audit) → これらは 0 信号
- 5,206 / 152,000 ≈ 3.4% の sample が PPO 内で「無効化に近い」状態

仮に 6 family を engine 自動化に戻すと:
- imitation/PPO の演算量が ~3.4% 削減
- advantage normalization の denominator が discard/response のみで構成 → normalization が optional-off と同等
- trajectory に no-op transition が無くなる → GAE pure

H1 単独では効果は限定的だが、H3 と組み合わせると **「初期 cycle で auto-riichi を失う」** + **「奪われた勾配を補えるほどの新規信号も無い」** という負のスパイラルを構成する。

---

## 3. Recommended Next Actions

### 3.1 CQ 候補 (優先度順)

#### CQ-A (最優先): 決定論 5 family を engine 自動化に戻す (defaults を off に)

`Tsumo / Ron / Ankan / Kakan / Kyuushu` の 5 family は現状 teacher が完全決定的 (常に Win または常に Skip)。これらを **default で disabled に戻す**:

- env 配線は残す (将来 strategic teacher を入れたいときのため)
- runner config 既定値を `false` に
- exp_034 baseline と同等の trajectory を再現可能にする

**Riichi だけ optional の余地はあり得る** (defensive riichi-skip の戦略性)。残すなら CQ-B と組み合わせる。

##### 期待効果

- imitation/PPO 計算量 ~3% 削減
- trajectory に no-op transition が消える → advantage normalization が optional-off と同等
- 5,200 個の 0 信号 sample が消えて discard/response の有効勾配が相対的に強まる

##### 検証
- 単純な config flip でテスト無しに通せる
- 既存 unit test (`test_stage2a_optional_*`) はすべて pass のまま

#### CQ-B (高優先): Riichi optional の wide-mask ハンデを early cycle で緩和

CQ-A だけでは exp_034 に追いつかない可能性が高い (H3 が残る)。Riichi 単独 optional でも wide mask の影響は残るため:

##### 案 1: Curriculum (推奨)
- 最初 N cycle (e.g., 20) は `optional_riichi.enabled=false` (auto-riichi)
- cycle N+1 から `optional_riichi.enabled=true` に切替
- → 政策が auto-riichi の振る舞いを身につけてから wide-mask を開放

##### 案 2: Imitation 段階で teacher 優先度を強化
- `_imitation_discard_epoch` ([stage2a_learner.py:939-1003](python/mahjong_rl/stage2a_learner.py#L939-L1003)) の CE loss を、立直可能局面で `teacher_top1` 重みを大きくする (例: `riichi_priority_weight=2.0`)
- PPO 側は触らない (wide mask の選択肢は維持)

両案とも実装コスト小。Curriculum は config 追加のみ。

#### CQ-C (中優先): `RON_OPTIONAL` 発火時に Chi/Pon/Daiminkan を含める

`make_ron_optional_candidates` を拡張するか、Ron + (Chi/Pon/Daiminkan) が同時に legal なときは通常 `RESPONSE` にフォールバックする。M1 への対処。実害は小さいので CQ-A/B の後でよい。

#### CQ-D (低優先・潜在バグ): `make_ankan_optional_candidates` の tile_type 取得を defensive にする

L1 で指摘した `//4` の有無問題。engine convention 変更で即クラッシュするので、以下のいずれか:

```python
# 案 a: convention に依存しない
raw = int(ankan_action.tile) if ankan_action.tile < 255 else -1
tile_type = raw if 0 <= raw < 34 else (raw // 4 if 0 <= raw < 136 else -1)

# 案 b: engine が tile_type 値を返すことを assert + 同じコードのまま残す
assert 0 <= ankan_action.tile < 34, f"engine convention changed: {ankan_action.tile}"
tile_type = int(ankan_action.tile)
```

加えて新規 unit test:

```python
def test_engine_ankan_tile_field_is_tile_type():
    # 任意の Ankan action.tile が 0..33 の範囲内であること
    # engine convention が変わったら fail-fast
    ...
```

### 3.2 実験案

- **E1**: **最優先**。現コード (post-CQ-0295) で `exp_034` 相当 (optional-all-off) を再走し、CQ-0290〜CQ-0295 の変更で baseline 自体が劣化していないことを confirm。 もし劣化していたら H1/H2/H3 の議論より前にそれを直す必要がある。
- **E2**: CQ-A 適用後、`optional_riichi` のみ on で 60 cycle seed42 で `exp_037 RII_ONLY` の改善度を測定。
- **E3**: CQ-A + CQ-B (Curriculum 20 cycle) で同条件。exp_034 に tail10 で追いつけるか確認。
- **E4**: CQ-0295 family diagnostics を使い、cycle 別に `ppo_diag["decision_family"][fam]["approx_kl_mean"]` をプロット。H1 (binary family が PPO に与える更新量) を定量化する。

### 3.3 やらなくてよいこと

- **value trunk を family ごとに分離する**: target が consistent なので追加効果は薄い。複雑化のデメリットの方が大きい。
- **gradient_norm / target_kl diagnostics の更なる検証**: CQ-0293 で applied diagnostics の整合は確認済み。今回の劣化原因ではない。
- **optional branch だけ PPO を外して imitation-only**: imitation はすでに 100% 一致まで学習しているため、PPO を外しても挙動はほぼ変わらない。CQ-A の方が筋が良い。
- **`_make_optional_summary` の更なる拡張**: CQ-0294 で 11 family 対応済み。これは効いている (audit で family を識別できている)。

---

## 4. Open Questions

1. **exp_034 の per-cycle riichi 率 / policy_deal_in 率は?**
   exp_036 cycle 59 riichi=1503/2130 ≈ 0.706 per round は単独では妥当に見える。これが exp_034 (cycle 59 相当) と比べて低いかどうかで H3 の支配性を強く確定できる。`exp_034` の summary.json を同様に解析すれば即わかる。

2. **`policy_ratio=1.0` は wide-mask optional unlock 下で適切か?**
   全 4 席が同じ政策を共有する setup で wide mask を開放すると、政策のミス (立直 bypass) が **全 4 席で同時に発生**する → trajectory が悪化する複利効果。`policy_ratio<1.0` (= baseline 対戦相手を混ぜる) にすれば、相手は auto-riichi のままなので最低限の対局品質が保たれる可能性。

3. **`imitation_epochs=8` は 6 決定論 family を考えると過剰?**
   M3 の懸念。CQ-A 適用後でも 4 epoch 程度に下げると imitation 段階の `value_trunk` 過適合が緩和されるかもしれない。

4. **CQ-0295 family diagnostics で各 family の `ppo_diag["decision_family"][fam]["approx_kl_mean"]` は実際どのぐらい動いている?**
   理論上 0 に近いはずだが、measured 値次第で「これらの sample は本当に何もしていない」ことを定量化できる。`ppo_diag["decision_family"][fam]["clip_fraction"]` も同様。

5. **cycle 別の `optional_decision_count` と `avg_rank` の相関は?**
   exp_036 summary.json をスクリプトで walk すれば取れる。もし negative correlation があるなら、optional sample の混入が直接の性能低下シグナル。

6. **exp_037 の RII_ONLY/WIN_ONLY/KAN_ONLY 3 run はすべて同 hyperparameters で実施された?**
   `RII_ONLY` の best_cycle=6 と他 family の best_cycle=28/29 の乖離が大きい。RII_ONLY 単独で early peak しているのは、wide-mask が立直機会を bypass しはじめる cycle 7 以降で性能が落ちている可能性を示唆する (H3 の典型例)。

---

## 5. 補足: なぜ「致命的バグ」ではなく「設計上の負荷」と判断したか

レビュー時に意識的に「明確なバグ」を探したが:

- env: `_auto_advance` / `_execute_and_advance` の skip 集合管理、reward 累積、step_id 順序は破綻していない。
- model: `forward_optional` / `forward_discard` / `_make_optional_summary` / `_compute_value_hidden` / `value_head` の入出力サイズと意味は (CQ-0294 後で) 整合している。`load_stage2a_state_dict` の column insert も `tests/python/test_stage2a_cq0294_riichi_teacher.py::TestLegacyCheckpointLoad` で検証済み。
- learner: PPO/imitation の sample partition、GAE、advantage normalization、target_kl gating は CQ-0287/CQ-0293 で精査済み。CQ-0295 family diagnostics も同じデータパスで矛盾なく動く。
- evaluator: baseline seat の teacher mask 使用 (CQ-0294)、policy seat の wide mask、optional 6 family の baseline_optional_index、全てが selfplay と整合。
- audit: `optional_family_audit` 自体は read-only。learner family diag は applied minibatch の per-sample tensor から純粋集約。学習挙動には影響しない。

唯一の bug 候補は L1 (`make_ankan_optional_candidates` の `//4` 欠落) だが、engine convention で偶然マスクされている。

したがって性能低下は **コードバグの結果ではなく、optional unlock という仕様変更が抱える本質的な学習負荷の表れ** と判断する。

---

## 6. 一文要約

> **CQ-A (5 family の自動化への巻き戻し) と CQ-B (Riichi optional の curriculum 化または imitation teacher 強化) を組み合わせれば、exp_034 baseline と同等以上の性能を維持しながら、戦略性のある Riichi optional だけを残せる可能性が高い。**

---

このレビューは Anthropic の CLI コーディング・アシスタント Claude Code が、
ユーザー (takeo1116) の依頼を受けて、リポジトリ内のソースコード・
実験 summary・CQ-0295 audit 結果・configs のみを直接参照して独立に作成
したものです。

- 作成: Claude Code (Anthropic)
- model: Claude Opus 4.7 (1M context)
- 作成日: 2026-05-11
- 対話セッション: ローカルの Claude Code CLI

(過去レビュー履歴:
`exp_022/claude_code_review.md` で mixed PPO baseline ratio 問題を指摘
→ CQ-0282 で fix。
`exp_025/claude_code_review.md` で `point_delta_scale=1.0` 問題を指摘
→ CQ-0283 で fix。
`exp_027/claude_code_review.md` で `terminal-driven value_trunk shaping`
と `semantic_proj` の dead weight を指摘
→ CQ-0287 / CQ-0286 / CQ-0288 で対応。
`exp_032/claude_code_review.md` でルール拡張前の最終確認、Critical なし。
本レビュー (`exp_037`) は optional unlock 後の初の独立レビューで、
**致命的バグは無く、性能低下は wide discard mask による auto-riichi 喪失と
決定論 family の勾配バジェット浪費が支配的** と結論した。)
