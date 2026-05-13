# Optional Unlock 性能低下 第二次独立レビュー (exp_039)

作成日: 2026-05-13
作成者: Claude Code (Anthropic) / model: Claude Opus 4.7 (1M context)
対象: CQ-0290 〜 CQ-0296 適用後、`exp_039` 3-seed が `exp_034` を
     `tail10 +0.04` 程度上回って劣化していること、および `exp_037
     RII_ONLY` / `exp_038 RII_WIDE1` が optional-on で性能を取り戻せない
     ことの原因を、コード/データから独立に再評価する。

参照ソース:

- `python/mahjong_rl/env/stage2_env.py`
- `python/mahjong_rl/env/response_candidate.py`
- `python/mahjong_rl/stage2_selfplay_worker.py`
- `python/mahjong_rl/stage2a_learner.py`
- `python/mahjong_rl/stage2a_evaluator.py`
- `python/mahjong_rl/stage2a_parallel.py`
- `python/mahjong_rl/models/stage2a_model.py`
- `python/mahjong_rl/encoders/flat_encoder.py`
- `python/mahjong_rl/legal_mask.py`
- `python/mahjong_rl/candidate_encoding.py`
- `python/mahjong_rl/optional_family_audit.py`
- `experiments/Stage02_CallUnlock/exp_034/report.md`
- `experiments/Stage02_CallUnlock/exp_037/report.md`
- `experiments/Stage02_CallUnlock/exp_037/claude_code_review.md`
- `experiments/Stage02_CallUnlock/exp_038/report.md`
- `experiments/Stage02_CallUnlock/exp_039/report.md`
- `experiments/Stage02_CallUnlock/exp_036/optional_family_audit_final_20k/optional_family_audit_exp036_final_20k_summary.md`
- `docs/CHANGE_QUEUE.md`

---

## 0. 結論先出し (4 行)

> 1. 前回 (exp_037 review) で「wide discard mask が auto-riichi を解除した」と推定したが、audit で **riichi bypass rate ≈ 0.04%** であり、現在の主要原因ではない。前回の仮説は **棄却**。
> 2. **新たに発見した主要因**: `optional_riichi=True` で `RIICHI_OPTIONAL` が発火した turn では、**reward が DISCARD サンプルから RIICHI サンプルへ系統的にシフトする**。同じ engine state が 2 サンプルで表現され、value head は同じ状態で 2 つの異なる target に追従させられる。
> 3. 副次的に発見: RIICHI/TSUMO/RON/ANKAN/KAKAN/KYUUSHU 各 optional 分岐の **encoder 入力が `legal_mask` / `riichi_discard_mask` を渡しておらず、対応する DISCARD サンプルと feature が一致しない**。同一 engine state の value 学習が分裂する。
> 4. optional-off (exp_039) と exp_034 の `+0.05 tail` ギャップは、**`_make_optional_summary` の action_type_presence が 4 → 11 に拡張されたことで value_trunk 初段の入力次元が +7 増えた**ことが小さな architecture drift として効いている可能性が高い。CQ-0294 で legacy column 0-pad は実装済みだが、新規初期化重みが optional-off 時に学習されないため 7 列分の dead weight が残る。

優先度の高い対処は CQ 候補 A〜D に整理して §3 で示す。  
最小実験は §4 で 2 種類提案する。

---

## 1. Findings

### 1.1 Severe (性能低下に直結する可能性が高い)

#### Finding S1 — RIICHI_OPTIONAL 発火時、reward が DISCARD サンプルではなく RIICHI サンプルに付与され、GAE / value target がシフトする

##### 根拠
- [stage2_selfplay_worker.py:246-371](python/mahjong_rl/stage2_selfplay_worker.py#L246-L371) の DISCARD branch:
  1. `DecisionSample(decision_family="discard", reward=0.0, ...)` を作成し `pending[P] = sample` に格納
  2. `step_counter += 1`
  3. `env.step_discard_with_snapshot(action, discard_snap)` → `_maybe_open_riichi_optional` が発火するとき **engine step は走らず** `rewards = zeros` を返す ([stage2_env.py:400-419](python/mahjong_rl/env/stage2_env.py#L400-L419))
  4. `_accumulate_pending_rewards(pending, zeros)` → 何も加算されない
- 次イテレーションで `decision_type == RIICHI_OPTIONAL` を処理 ([stage2_selfplay_worker.py:441-503](python/mahjong_rl/stage2_selfplay_worker.py#L441-L503)):
  1. 新しい `DecisionSample(decision_family="riichi", reward=0.0, ...)` を作成
  2. 行 493-496 `if player in pending: prev = pending.pop(player); round_buffer.append(prev); pending[player] = sample` で **`Q.DISCARD` サンプルは `round_buffer` に追放** され、`pending[Q] = Q.RIICHI` に置き換わる
  3. `env.step_response(idx)` → engine が実 discard を実行 → 実 rewards (= 立直供託 -1000 や同巡 Ron 失点等)
  4. `_accumulate_pending_rewards(pending, real_rewards)` → **`pending[Q] = Q.RIICHI` にだけ加算**される (DISCARD は既に round_buffer)

##### 比較 (optional-off 経路)
- exp_034 (`optional_riichi=False`):
  - `_resolve_discard` ([stage2_env.py:610-637](python/mahjong_rl/env/stage2_env.py#L610)) が立直 action を優先選択
  - `_execute_and_advance(action)` で **engine step が即実行** → 実 rewards
  - `pending[Q] = Q.DISCARD` がそのまま受け取る → `Q.DISCARD.reward = -1000` (立直供託)
- exp_036/037/038 (`optional_riichi=True`, riichi 発火 turn):
  - `Q.DISCARD.reward = 0` (engine step deferred)
  - `Q.RIICHI.reward = -1000`

##### GAE への影響
`_compute_returns_advantages` ([stage2a_learner.py:1588-1620](python/mahjong_rl/stage2a_learner.py#L1588)) は `(episode_id, player_id)` でグループ化して step_id 昇順に GAE を計算する。同一 player の `Q.DISCARD` (step_id=t) と `Q.RIICHI` (step_id=t+1) は **同一 engine state を表す** にも関わらず、

- optional-off: `return_d = R_real + γ V_{Q.next_real}`
- optional-on:  `return_d = 0 + γ * return_r = γ (R_real + γ V_{Q.next_real})` = `γ R_real + γ² V_{Q.next_real}`

config の `gamma=0.5`, `gae_lambda=0.0` 下では:
- optional-off: `V_d` のターゲットは `R_real + 0.5 V_next`
- optional-on:  `V_d` のターゲットは `0.5 V_r`、`V_r` のターゲットは `R_real + 0.5 V_next`
- 結果: optional-on の `V_d` は optional-off の `V_d` の **約 (γ-1)/(1-γ) = -1 倍** ではなく、`V_d ≈ 0.5 (R_real + 0.5 V_next)` = optional-off の **約 1/2** に systematic に縮む

##### Policy gradient への影響
PPO の discard step 上の advantage:
- optional-off: `A_d = R_real + γ V_next - V_d` ≈ value convergence で 0、但し収束過程で `|R_real|` 分の信号がある
- optional-on:  `A_d = 0 + γ V_r - V_d` ≈ value convergence で 0、`|R_real|` の即時報酬を bootstrap でしか得られない

つまり「立直する discard かどうか」を学ぶ際、**optional-on は immediate reward 信号を完全に失い、bootstrap (γ V_r - V_d) のみで policy gradient が決まる**。`point_delta_scale=0.0001` × 立直供託 1000 = **0.1 の reward magnitude を毎 RIICHI turn で discard step から消している**ことになる。

##### 性能影響の見積もり
- audit 上 riichi turn は全 discard の 1.6% (246/15231)。直接的な per-step 効果は小さい。
- ただし value head は同一 engine state を 2 サンプル (df=0.0 / df=1.0) で予測することを要求され、**convergence が遅くなる**。
- 加えて `optional_summary` で 2 サンプルの feature が異なる (S2 参照) ため、value head は decision_family + optional_summary の組合せで両者を区別しなくてはならない。これは学習負荷を増やす。
- exp_034 / exp_039 (riichi 発火なし) と exp_036/037/038 (riichi 発火あり) で `tail10` が ~+0.1 ずれる規模感と整合する。

##### Severity: **High** (性能低下の支配的要因の有力候補)

#### Finding S2 — RIICHI_OPTIONAL / その他 optional サンプルの encoder 入力が `legal_mask` / `riichi_discard_mask` を渡しておらず、同一 engine state の DISCARD サンプルと feature が一致しない

##### 根拠
[stage2_selfplay_worker.py:375, 446, 516, 589](python/mahjong_rl/stage2_selfplay_worker.py) で 4 つの call branch 全てが以下:

```python
obs = env._make_observation()
features = self._encode_obs(obs)   # ← 引数 legal_mask, riichi_discard_mask を渡さない
```

`_encode_obs` ([stage2_selfplay_worker.py:143-160](python/mahjong_rl/stage2_selfplay_worker.py#L143-L160)) は `encoder._riichi_discard_mask` フラグを見て、有効でも引数 `riichi_discard_mask=None` のまま encoder に渡す。`FlatFeatureEncoder._coerce_riichi_discard_mask` ([flat_encoder.py:651-669](python/mahjong_rl/encoders/flat_encoder.py#L651-L669)) は `None` を 34-dim zeros に変換する。

つまり同じ engine state での 2 サンプル:

- DISCARD サンプル: `features[riichi_discard_mask_range]` = actual riichi-eligible 34-dim mask (例: 立直可能な tile_type 4 つに 1)
- RIICHI サンプル:   `features[riichi_discard_mask_range]` = **全 0**

`discard_ukeire_hint` も `legal_mask` が必要だが call branch で渡されない (空マスクに退化)。

##### Evaluator も同じ問題
[stage2a_evaluator.py:254-256](python/mahjong_rl/stage2a_evaluator.py#L254-L256) の `_policy_call`:

```python
def _policy_call(self, env, candidates, player):
    obs = env._make_observation()
    features = self._encoder.encode(obs)   # ← legal_mask / riichi_discard_mask 渡さず
```

selfplay 経路 (worker) と eval 経路 (evaluator) で同じ抜けがあるため selfplay/eval 内の整合性は保たれているが、**「同一 engine state を 2 種の feature で表現する」点は同じ**。

##### 性能影響
- value head は `value_trunk` で features 全体を処理。同じ engine state の 2 サンプルで `riichi_discard_mask` 部分 (34 dim) が完全に違う → 同じ状態の V 予測が分裂する。
- model はそれを「decision_family と optional_summary で吸収」しなければならず、representation capacity を浪費。
- `tile_presence_flags=True` 構成では同様の不一致が出る可能性がある (こちらは `obs.hand` から計算するため影響なし、要確認)。

##### Severity: **High** (S1 と組み合わせて value 学習の主要原因候補)

#### Finding S3 — `optional_summary` の action_type_presence 拡張 (4 → 11) で value_trunk 初段に 7 dim の dead column が常時存在する (optional-off でも)

##### 根拠
[models/stage2a_model.py:142-160](python/mahjong_rl/models/stage2a_model.py#L142-L160):

```python
@classmethod
def _summary_fixed_dim(cls) -> int:
    from mahjong_rl.candidate_encoding import NUM_ACTION_TYPE_INDICES
    return 2 + NUM_ACTION_TYPE_INDICES   # = 13 (旧 6)
_SUMMARY_FIXED = 2 + 11
```

[models/stage2a_model.py:242](python/mahjong_rl/models/stage2a_model.py#L242):

```python
summary_dim = self._SUMMARY_FIXED + candidate_dim * 2
val_input = trunk_input_dim + 1 + RESPONSE_CONTEXT_DIM + summary_dim + value_aux_dim
```

value_trunk の入力次元は `+7 (= 11-4)` 増えた。旧 checkpoint からの load は CQ-0294 で zero-pad 互換実装済 ([models/stage2a_model.py:78-123](python/mahjong_rl/models/stage2a_model.py#L78-L123))。

##### exp_039 (optional-off) でも影響する理由
- optional-off では 5 family の `optional_summary` action_type_presence は常に 0 (該当 family の candidate が空)。
- しかし、value_trunk の **初段重みの 7 列はランダム初期化** されたまま optional-off では gradient が 0 (入力が常に 0)。
- 結果: value_trunk の有効 input rank が `(全列 − 7)` になる。これは optional-off では完全に dead weight。
- しかし `_compute_value_loss` の backward は通る (入力 0 × 重み = 0 だが、その他の列との結合で初段の bias / 他の列の重みは学習する)。
- exp_034 (旧 4-row model) と exp_039 (新 11-row model) では value_trunk の **初期化乱数** が変わるだけでなく、**「実質的な representational capacity が 7 dim 分減る」** 影響もある。

##### 性能影響の見積もり
- 7 dim × value_trunk hidden = 7 × 256 = 1792 個の weight が optional-off では更新されないが、forward は通る (= bias 経路に影響なし)。
- 主に乱数初期化依存性。exp_039 vs exp_034 の `+0.05 tail` ギャップは、**乱数バリエーション内 + capacity の小さな低下** で十分説明できる規模。
- exp_038 OFF_WIDE1/2 (value_hidden_dims=[384,192], [512,256]) で大きな改善が出なかったのは、capacity ではなく **input feature の dead column が問題** という仮説と整合する (capacity を増やしても dead column は dead のまま)。

##### Severity: **Medium-High** (exp_034 → exp_039 の `+0.05 tail` ギャップの主要候補)

### 1.2 Medium

#### Finding M1 — `_compute_terminal_weights_cross_branch` で player-round 正規化重みが optional-on により希釈される

##### 根拠
[stage2a_learner.py:360-415](python/mahjong_rl/stage2a_learner.py#L360-L415):

```python
counts = Counter(d_keys)      # (eid, rid, pid) ごとに discard count
counts.update(c_keys)          # call count も足し込む
# 各 sample の weight = 1 / counts[key]
```

同一 player の同一 round 内で sample 数 K に対し 各 sample の重みは 1/K。

##### optional-on での影響
- exp_034 (optional-off): 1 round per player ≈ 14 DISCARD + 0-2 RESPONSE = 14-16 sample → 1/14
- exp_036 (optional-all): ≈ 14 + 2 + 0.7 RIICHI + 0.3 TSUMO + 1 RON + 0.05 ANKAN + 0.1 KAKAN ≈ 18 sample → 1/18

`terminal_loss` / `yaku_loss` の per-sample 寄与が ~22% 縮む。semantic aux loss の総量も若干減る。**性能低下の支配要因ではない** が、optional-on で value/semantic 学習の「速さ」が落ちる方向に効く。

##### Severity: **Medium** (補助要因)

#### Finding M2 — `optional_riichi=True` での policy mask 拡張により、imitation の teacher 信号が「policy が選んだ tile」と乖離する

##### 根拠
- worker DISCARD branch (CQ-0294 後): policy は **wide mask** で選ぶ、teacher_top1 は **riichi-only mask** で計算 ([stage2_selfplay_worker.py:269-303](python/mahjong_rl/stage2_selfplay_worker.py#L269-L303))。
- imitation discard epoch [stage2a_learner.py:939-1003](python/mahjong_rl/stage2a_learner.py#L939-L1003) は teacher_top1 を target とした CE loss。
- PPO discard epoch は `s.action` (政策が実際に選んだ tile) を target に PPO。

##### 影響
- 立直可能 turn で imitation の teacher は「riichi 牌のみ」を示すが、PPO は policy のサンプリング結果に基づく。policy が riichi 牌を確率高く選ぶよう imitation で誘導されるが、wide mask 下で softmax 温度 1.0 の探索が入ると非 riichi 牌も時々選ばれ、PPO がそれを学習する。
- audit によると bypass rate は 0.04% (cycle 後半) で、政策はほぼ「常に riichi 牌を選ぶ」状態に収束済み → この経路自体は致命的ではない。
- ただし初期 cycle (cycle 0-20) では policy が riichi 牌に収束するまで wider mask が探索を許す → 立直機会喪失 → trajectory 品質悪化 → 学習信号の悪循環。

##### Severity: **Medium** (early training に効くが、最終性能の差ではない)

### 1.3 Low

#### Finding L1 — `_compute_terminal_weights` の `(episode_id, round_id, player_id)` グルーピングは `_make_observation` の `round_id` の更新タイミングに依存しており、RIICHI_OPTIONAL inserted step が `round_id` を共有することは確認済 (バグなし、確認のみ)

##### 根拠
- worker `round_id = env.env_state.round_state.round_number` を sample 作成時に取得 ([stage2_selfplay_worker.py:244](python/mahjong_rl/stage2_selfplay_worker.py#L244))。
- DISCARD と RIICHI_OPTIONAL は同一 round 内で連続するため `round_id` は同じ。`episode_id` (= match seed) と `player_id` も同じ。
- `_compute_terminal_weights_cross_branch` で同 group の K samples = 1/K 重み付け → 立直可能 turn では DISCARD + RIICHI の 2 sample が同 group → 重みは更に薄まる。

##### 結論
バグなし、ただし M1 と組み合わさって semantic aux loss を更に dilute する。

#### Finding L2 — `imitation_eval_metrics` での optional flag は selfplay の flag と一致するよう CQ-0292 で配線済み

[runner.py:_run_eval_stage2a](python/mahjong_rl/runner.py) で `_stage2a_optional_flags()` 経由で同じ flags が evaluator/parallel に伝播。問題なし。

#### Finding L3 — `riichi_discard_mask` feature の semantics は train/eval/selfplay で一致している (CQ-0294 follow-up で確認済)

ただし call branch では渡されていない (S2)。

#### Finding L4 — `_make_optional_summary` で valid candidate を `cand_mask > 0.5` で判定するため、padding (0) は含まれない (CQ-0294)

問題なし。

#### Finding L5 — Stage2Env.step_response の skip 判定が action_type ベースに修正済 (CQ-0296)

問題なし。

---

## 2. Hypothesis Ranking

ユーザーから提示された仮説 A/B/C と本レビューの新仮説を統合してランク付けする。

### Rank 1: 仮説 B + S1 + S2 (= 本レビューの主仮説)

**「RIICHI_OPTIONAL が挿入されたことで reward attribution と feature consistency が崩れ、value head が同一 state を 2 通りに学ばされている」**

#### 内容
- S1: 立直 turn の reward が discard sample から riichi sample にシフトし、GAE return が γ 倍縮む。
- S2: 同 state の 2 サンプルが encoder 入力で異なる feature を持つ (`riichi_discard_mask`, `discard_ukeire_hint`, etc)。
- 結果として value head は「同じ状態の 2 種の feature 表現それぞれに違うターゲット」を学ぶことになる。

#### 検証方法
1. **`exp_037 RII_ONLY` の checkpoint で audit + diagnostic を追加実行**:
   - 立直 turn の DISCARD sample と RIICHI sample をペアリングし、value 予測値の差分を計測する
   - `V(DISCARD) ≈ γ * V(RIICHI)` が成立しているか確認 (γ=0.5)。成立していれば S1 の predicted behavior と一致
2. **`riichi_discard_mask` feature を call branch でも渡す patch を当てて、3-seed で 30 cycle 比較**:
   - S2 の影響度が単独で見える
   - 改善が小さければ S1 が主因 / 改善が大きければ S2 が主因
3. **DISCARD sample の reward に立直供託 -1000 を直接コピーする patch** (S1 fix):
   - 立直 turn の discard sample が optional-off 同等の reward signal を得る
   - bypass を導入せず "reward duplication" の形でも実装可能

#### 確信度
**High**。新発見の bug-like behavior であり、コード読みで明確に確認できる。

### Rank 2: 仮説 A (リーチしてるが tile quality が落ちている)

#### 内容
立直するかどうかは正しいが、**どの牌で立直するかが teacher と異なる**可能性。

#### 検証方法
追加 diagnostic を実装し、立直 turn で:

- `policy_discard_tile_type`
- `teacher_top1_tile_type` (riichi-only mask の baseline best)
- `match_rate` を集計

audit ではまだこの粒度の集計はしていない。

#### 確信度
**Medium**。bypass rate が極小なので「立直する/しない」の意思決定は正しい。だが立直時の打牌選択が rule-based teacher と完全一致するかは未検証。  
特に exp_034 の `auto-riichi` は engine が `_resolve_discard` で `(not riichi, _is_red_tile_id)` sort を使うのに対し、wide-mask 下の policy 選択がそれと等価かは確認していない。

### Rank 3: 仮説 C (まだ実装バグが残っている)

#### 内容
optional unlock 周辺の細かい不整合が積み上がっている。本レビューで以下を特定:

- S1 (reward shift) — 仕様の意図 vs 実装の差として "bug-ish"
- S2 (feature inconsistency) — 明確な改善余地
- S3 (dead column) — exp_039 vs exp_034 の小さなギャップに寄与

これらは S1/S2/S3 として既に Rank1 に含めた。**残る "明確なバグ" は本レビューでは見つけられなかった**。

#### 確信度
**Low** (新たな致命バグは見つかっていない)

### Rank 4: 元の wide discard mask + auto-riichi 喪失 仮説

#### 内容
前回 (`exp_037/claude_code_review.md`) で提示。

#### 検証結果
**棄却**。audit で bypass rate=0.04% であり、policy は wide mask 下でもほぼ riichi 牌を選んでいる。前回時点ではこの精度の audit がなかった。

#### 確信度
**Low**。

---

## 3. Recommended CQ

### CQ-A (最優先): RIICHI_OPTIONAL inserted sample で reward attribution の整合性を取り戻す

#### 実装案 (3 通り)

**案 a1: RIICHI_OPTIONAL 発火時、DISCARD サンプルを pending から外さない**

[stage2_selfplay_worker.py:493-496](python/mahjong_rl/stage2_selfplay_worker.py#L493-L496) の `pending.pop(player) → round_buffer.append(prev)` を、`decision_family == "riichi"` のときは **抑制**し、DISCARD サンプルを pending に残す。

```python
# CQ-X: RIICHI_OPTIONAL は同一 engine state の延長なので、
# 直前 DISCARD サンプルを round_buffer に送らず、reward を引き続き
# 累積させる。
if player in pending:
    prev = pending.pop(player)
    if env.decision_type != DecisionType.RIICHI_OPTIONAL:
        round_buffer.append(prev)
    else:
        # DISCARD を round_buffer 送りせず、新 RIICHI とは別管理。
        # この turn の engine step が起きたら、両者に reward 累積する。
        ...
```

pending を `dict[player, list[sample]]` に拡張する変更が必要 (今は 1 sample のみ)。

**案 a2 (より小さい変更): reward を 2 重カウント**

engine の real_rewards を **DISCARD と RIICHI 両方に加算**する。両者は同じ state 遷移を表すので、return の総量が optional-off と一致する。

ただし `_compute_returns_advantages` で GAE をかけると、step_id ベースで両者の reward が時間軸上に並ぶため、γ 効果で過剰反映になる懸念がある。

**案 a3 (最小変更): RIICHI_OPTIONAL を sample 化せず、env 側で「立直するか」を policy に問い合わせる**

worker レベルで discard sample のみを残し、その内部で `optional_riichi_enabled` 時に立直/非立直のラベル付け (= `selected_candidate_index` 相当) を **discard sample の補助フィールド** に格納する。

但しこれは shard schema を変える大改修。scope 外。

#### 推奨
**案 a1** を採用。CQ-0274 の "same-player transition reward" semantics を維持しつつ、optional decision を「同一 engine state の補助 sample」として扱う実装。

#### テスト
- 既存 `tests/python/test_riichi_optional.py` / `test_optional_win.py` / `test_optional_kan_kyuushu.py` の regression
- 新規:
  - `optional_riichi=True` で立直 turn の DISCARD サンプルが engine reward を受け取ることを確認
  - `optional_riichi=False` 経路に影響しないことを確認 (= DISCARD だけが pending に残る既存挙動)
  - GAE return が optional-off と等価になることを確認 (同 seed で reward 系列が一致)

#### 既存挙動を壊さない条件
- `optional_riichi=False` では完全に従来通り (skip 経路を通らない)
- TSUMO/RON/ANKAN/KAKAN/KYUUSHU 経路への影響なし (これらは engine step が必ず起きる or skip 集合管理が違う)
- shard schema 変更なし

### CQ-B (高優先): optional decision branch の encoder 入力に `legal_mask` / `riichi_discard_mask` を渡す

#### 実装案
[stage2_selfplay_worker.py:375, 446, 516, 589](python/mahjong_rl/stage2_selfplay_worker.py) の 4 箇所:

```python
features = self._encode_obs(obs)
```

を:

```python
# CQ-Y: 同一 engine state の DISCARD サンプルと feature を一致させる
features = self._encode_obs(
    obs,
    legal_mask=env.get_legal_mask() if self._needs_legal_mask else None,
    riichi_discard_mask=env.get_riichi_discard_mask()
        if encoder._riichi_discard_mask else None,
)
```

evaluator も同様。`_policy_call` ([stage2a_evaluator.py:254-256](python/mahjong_rl/stage2a_evaluator.py#L254)) を更新。

#### テスト
- DISCARD サンプルと RIICHI サンプルが同 engine state で同じ `features[riichi_discard_mask_range]` を持つことを確認
- `discard_ukeire_hint=True` の場合に、call branch の sample でも同じ hint 値が入ることを確認
- model forward が crash しないこと

#### 既存挙動を壊さない条件
- `feature_encoder.riichi_discard_mask=False` の config では `riichi_discard_mask=None` のまま (zero) なので変化なし
- `feature_encoder.discard_ukeire_hint=False` でも同様
- 旧 shard との互換性: shard の observation は既に保存済なので影響なし。新規 selfplay 以降のみ feature が変わる。

### CQ-C (中優先): `_make_optional_summary` action_type_presence の dead column 対策

#### 実装案 1: 設定で 4-row legacy mode に切替可能にする

```python
class Stage2aModel:
    def __init__(..., legacy_optional_summary_action_types=False):
        ...
        if legacy_optional_summary_action_types:
            self._action_type_presence_size = 4
        else:
            self._action_type_presence_size = NUM_ACTION_TYPE_INDICES
```

`optional_riichi/tsumo/ron/ankan/kakan/kyuushu = false` の場合は legacy mode で動作させる runner ロジックを入れる。

#### 実装案 2: dead column を runtime で削減 (lazy expansion)

active な action_type のみを presence に出す。これは大改造。

#### 推奨
**案 1**。optional-off ベースライン (exp_039 など) で legacy 4-row モードを使えば、exp_034 とほぼ identical な architecture が再現できる。

#### テスト
- `legacy_optional_summary_action_types=True` で value_trunk の input dim が旧 model 同等
- 旧 checkpoint がそのまま load できる
- optional-on で `legacy=False` の挙動が変わらない

#### 既存挙動を壊さない条件
- default は `False` (= 現行 11-row)
- optional-off baseline experiment のみ `legacy=True` で再現

### CQ-D (低優先): 立直 turn の "policy_discard_tile_type vs teacher_top1_tile_type 一致率" を audit に追加

仮説 A の検証用 diagnostic。`optional_family_audit` に立直 turn の discard tile 一致率列を追加。

#### テスト
- audit に新規 column が出る
- JSON serializable

---

## 4. Recommended Experiment

### E1 (最優先): S1 / S2 影響の切り分け

`exp_037 RII_ONLY seed42` の checkpoint で、以下の追加 diagnostic を実行:

1. 立直 turn の DISCARD sample と RIICHI sample をペアリング
2. `V(DISCARD)`, `V(RIICHI)` を model forward で取得
3. `V(DISCARD) - γ * V(RIICHI)` の分布を計算
4. 0 から有意にずれていれば、value head は同 state の 2 表現を別々に学習している (= S1+S2 が実害を伴っている)

#### 何が分かるか
- ずれが小さい (~0): value head は問題なく学べている → S1/S2 は実害なし → 別仮説を探す
- ずれが大きい: S1+S2 が value 学習を歪めていることが確定 → CQ-A/B 適用が正当

#### 実装コスト
小。`optional_family_audit.py` の拡張 1〜2 時間。

### E2 (S1 影響の単体実験): CQ-A 案 a1 を実装し、`exp_037 RII_ONLY` 同条件で 30 cycle 3-seed 比較

#### 条件
- `optional_riichi=True` のみ on
- `feature_encoder.riichi_discard_mask=true`
- 他は exp_037 RII_ONLY と完全同一
- seed 42, 43, 44

#### 何が分かるか
- exp_037 RII_ONLY (`tail10 mean ~2.28`) との差分が `+0.05` 以上改善 → S1 が主因確定
- 改善が小さい (`< 0.02`) → S2 単独か、別仮説 (S3 含む)
- 悪化する → CQ-A 案 a1 の reward 二重カウント等の副作用を疑う

#### 実装コスト
中。pending 構造変更 + tests + smoke で 4〜6 時間。

### E3 (S3 影響の単体実験): `legacy_optional_summary_action_types=true` の optional-off run

#### 条件
- exp_039 同設定 (`optional_*=false`)
- model のみ `legacy_optional_summary_action_types=true` (CQ-C 案 1)
- seed 42, 43, 44

#### 何が分かるか
- exp_039 (`tail10 mean 2.156`) より良くなり、exp_034 (`tail10 mean 2.113`) に近づく → S3 が exp_034→exp_039 の `+0.05 tail` gap を説明する
- 変わらなければ S3 は影響微小、別の architecture drift を探す

#### 実装コスト
小〜中。model param 追加 + tests + smoke で 2〜3 時間。

### E4 (オプション): exp_037 RII_ONLY のシード分散確認

exp_037 RII_ONLY は seed42 単発結果のみ。seed43, 44 を回して 3-seed で再評価。  
**他の E1〜E3 と並行可能**。exp_037 自体の信頼性を上げる目的。

---

## 5. やらなくてよいこと

- **value/semantic trunk capacity を更に上げる**: exp_038 で示されたとおり、capacity 不足ではない。
- **PPO `gamma` を上げて bootstrap を強くする**: S1 の根本問題は reward の物理的位置のズレ。`gamma` を変えても sample-level の reward が DISCARD/RIICHI のどちらにあるかは変わらない。
- **policy temperature の調整**: bypass rate は十分低いので explore-exploit は問題ではない。

---

## 6. Open Questions

1. **E1 の `V(DISCARD) − γ V(RIICHI)` の分布は本当に有意にずれているか?** 想定では平均 0 から離れる方向に分布するはずだが、実測未確認。
2. **S1 fix (CQ-A) で `tail` が exp_034 同等になれば**、性能低下の支配要因は確定。だが exp_038 WIDE で改善が見られなかったため、別要因の可能性も残る。
3. **`legal_mask` を call branch で渡すと `discard_ukeire_hint` の挙動が変わる**。これは call branch sample の features が discard branch と一致するようになる一方、call branch policy 自体の挙動も変わる。eval で副作用が出ないか smoke で要確認。
4. **CQ-0274 の "same-player transition reward" semantics と CQ-A 案 a1 (DISCARD を pending に残す) の合成は問題ないか?** RIICHI step の engine reward が DISCARD と RIICHI の両方に加算されると 2 重カウントになるリスクがある。a1 実装で `pending` の data structure を `list[Sample]` に変えるなどの慎重な設計が必要。

---

## 7. 一文要約

> **性能低下の支配的要因は「optional_riichi=True で立直 turn の reward が DISCARD サンプルから RIICHI サンプルへシフトし、かつ encoder feature も両者で異なるため、value head が同一 engine state を 2 通りに学ばされている」点と推定する。CQ-A (pending semantics 修正) と CQ-B (call branch encoder feature の一致) を併せて入れれば、exp_034 同等の性能まで戻せる可能性が高い。同時に CQ-C で optional_summary action-type-presence を optional-off 構成で legacy 4-row に戻せるオプションを足せば、exp_034 と exp_039 のギャップも消える可能性がある。**

---

このレビューは Anthropic の CLI コーディング・アシスタント Claude Code が、
ユーザー (takeo1116) の依頼を受けて、リポジトリ内のソースコード・
実験 summary・CQ-0290〜CQ-0296 実装ログ・前回 review
(`exp_037/claude_code_review.md`) を直接参照して独立に作成したものです。

- 作成: Claude Code (Anthropic)
- model: Claude Opus 4.7 (1M context)
- 作成日: 2026-05-13
- 対話セッション: ローカルの Claude Code CLI

(過去レビュー履歴:
`exp_022/claude_code_review.md` で mixed PPO baseline ratio 問題を指摘
→ CQ-0282 で fix。
`exp_025/claude_code_review.md` で `point_delta_scale` 問題 → CQ-0283 で fix。
`exp_027/claude_code_review.md` で terminal value_trunk shaping + dead semantic_proj 指摘 → CQ-0287/0286/0288 で対応。
`exp_032/claude_code_review.md` でルール拡張前の最終 sanity (Critical なし)。
`exp_037/claude_code_review.md` で optional unlock 後の初の独立レビュー、
当時の支配仮説 "wide discard mask による auto-riichi 喪失" を提示。
**本レビュー (`exp_039`) では audit データの精緻化と CQ-0294/0295/0296
実装後のコード再読により、その仮説を棄却し、reward attribution shift
+ feature inconsistency を新たな支配要因として提示する。** )
