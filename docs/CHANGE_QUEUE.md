# CHANGE_QUEUE.md

このファイルは **未反映の変更要求だけを置く作業キュー** である。  
履歴台帳ではなく、いま実装・レビューすべき項目だけを管理する。

- `Status: [Proposed]`: 未実装
- `Status: [Implemented]`: 実装済み、レビュー待ち

## 運用ルール

- 新しい CQ は末尾に追記する
- 項目順は並べ替えない
- 実装者は CQ を削除しない
- レビュー完了した CQ はレビュアーが削除する
- Claude Code が編集してよいのは原則として以下のみ
  - `Status` の更新
  - `実装メモ` への短い追記
  - 明確な誤字修正
- 仕様の議論や設計メモは、このファイルに長く書かず `PROJECT.md` / `GAME_SPEC.md` / `RL_SPEC.md` / `reference/stage2/` に置く
- `PROJECT.md` / `GAME_RULE.md` / `GAME_SPEC.md` / `RL_SPEC.md` / 実装は最終的に整合していなければならない

## テンプレート

### CQ-XXXX
- Status: [Proposed]
- Type: Rule | Engine | RL | Training | Eval | Test | Docs | IO
- Priority: High | Medium | Low
- Title: ここに短い変更タイトルを書く

#### 背景
なぜこの変更が必要かを書く。  
既存仕様や既存実装との関係があれば簡潔に書く。

#### 要求内容
実装してほしい変更内容を具体的に書く。  
必要なら箇条書きで列挙する。

#### 関連文書
- PROJECT.md: 段階判断や優先順位が関係する場合に書く
- GAME_RULE.md: 該当セクションがあれば書く
- GAME_SPEC.md: 該当セクションがあれば書く
- RL_SPEC.md: 該当セクションがあれば書く
- その他: 任意

#### 受け入れ条件
- 変更後に満たしてほしい条件を書く
- テストで確認可能な形が望ましい

#### 実装メモ
必要なら補足を書く。

---

## 変更要求一覧

### CQ-0274
- Status: [Implemented]
- Type: RL | Training | Test
- Priority: High
- Title: Stage2a selfplay の reward backfill を same-player transition 累積に修正する

#### 背景
Stage2a learner は same-player trajectory 単位で return / advantage を計算している。

- group key: `(episode_id, player_id)`
- 各 sample の reward は「その player の前回 decision から次の same-player decision までに発生した累積報酬」であるべき

しかし現在の `Stage2SelfPlayWorker` では、`env.step_discard_with_snapshot()` / `env.step_response()` 後の reward を **action owner の pending sample にだけ**代入している。

一方 Stage1 `SelfPlayWorker` では、各 env step の `rewards[p]` を 4 人全員の pending transition に累積している。

この差により Stage2a では、

- 他家ツモ時の被ツモ失点
- 他家ロン時の傍観者または放銃者以外への点数変動
- 自分の decision 後、次の自分の decision までに他家 action / auto advance で発生した点数変動

が、自分の直前 decision sample に乗らない可能性がある。

これは RL 理論上の設計変更ではなく、既存の same-player transition semantics に対する実装不整合であり、Stage1 から Stage2a への機能デグレとみなす。

#### 要求内容
`Stage2SelfPlayWorker` の reward backfill を、Stage1 と同じ発想の cumulative transition reward に修正する。

具体的には:

- env step 後に得た `rewards[p]` を、pending に存在する全 player `p` の sample に累積する
- action owner の新規 pending sample は、当該 action によって発生した reward も受け取れるようにする
- reward は代入ではなく累積とする
  - 現状の `pending[player].reward = float(rewards[player])` ではなく、pending sample ごとに `+=` する
- round end / match end / auto advance をまたぐ reward も、same-player transition に沿って欠落なく入るようにする
- `terminated` / `round_over` / outcome label backfill の既存挙動は維持する
- baseline / policy 混在時も、保存対象 sample だけに対して正しく累積する
  - 保存していない baseline seat の pending は存在しなくてよい
  - 保存している sample については actor_type に関係なく reward を累積する

#### 関連文書
- RL_SPEC.md
- PROJECT.md
- `python/mahjong_rl/stage2_selfplay_worker.py`
- `python/mahjong_rl/stage2a_learner.py`
- `python/mahjong_rl/selfplay_worker.py`
- `experiments/Stage02_CallUnlock/exp_015/report.md`
- `experiments/Stage02_CallUnlock/exp_019/report.md`

#### 受け入れ条件
- Stage2a worker で、1 env step の `rewards[p]` が pending 中の全保存対象 player sample に累積される
- action owner の sample も、自身の action で即時発生した reward を受け取る
- 他家ツモで、被ツモ者の pending sample に失点 reward が入ることをテストで確認する
- ロンで、放銃者の pending sample に失点 reward が入ることをテストで確認する
- 流局 / 途中流局で、pending sample が round_over と reward を保ったまま flush されることを確認する
- reward が代入で上書きされず、複数 step 分が累積されることを単体テストで確認する
- 既存の Stage2a selfplay / learner / runner smoke test が通る
- 修正後の shard reward 分布を確認できる簡易診断を残す、またはテストで旧挙動との差を明示する

#### 実装メモ
- Stage1 `SelfPlayWorker` の CQ-0210/0211 実装が参考になる
- 修正後は `exp_015 A2` 相当条件を再実験し、PPO retain / final improvement を見直す
- この修正が入るまでは、PPO hyperparameter tuning より reward semantics の修復を優先する

実装結果:
- 変更ファイル: `python/mahjong_rl/stage2_selfplay_worker.py`, `tests/python/test_stage2a_reward_backfill.py`
- `Stage2SelfPlayWorker._accumulate_pending_rewards()` static helper 追加
  - pending 中の全 player sample に `+=` で reward 累積
  - terminated フラグも同時に立てる
- discard step / response step 後の `pending[player].reward = ...` 代入を helper 呼び出しに置換
- テスト: 単体 8 件 + smoke 2 件、全 10 件 passed
- CQ-0276 (reward_config 伝播) と CQ-0277 (terminal weight 横断化) は据置き

---

### CQ-0275
- Status: [Implemented]
- Type: RL | Training | Test
- Priority: High
- Title: Stage2a PPO の advantage / return を branch 元順へ正しく scatter する

#### 背景
Stage2a PPO は discard / call samples を結合し、`step_id` 順に並べて same-player grouped GAE を計算している。

現在の実装では、

- `discard_samples` を順に `all_indexed` へ追加
- `call_samples` を順に `all_indexed` へ追加
- `all_indexed.sort(key=lambda x: x[0])` で `step_id` 順に並べる
- sort 後の順に `d_adv_list` / `c_adv_list` へ append する

という流れになっている。

しかし、その後の PPO epoch では `discard_samples[i]` / `call_samples[i]` の **元の branch 順**に対して `d_adv[i]` / `c_adv[i]` を参照する。

実際の Stage2a shard では、`read_all()` の順序も discard branch 内の順序も call branch 内の順序も `step_id` 昇順とは限らない。  
そのため、現在は別 sample の advantage / return を使って policy / value を更新している可能性が高い。

これは PPO の性能と安定性に直接影響する重大な実装不整合である。

#### 要求内容
`Stage2aLearner._train_ppo()` で、GAE 計算後の advantage / return / sample weight を元の branch sample 順へ正しく戻す。

具体的には:

- `all_indexed` に `(step_id, branch, branch_idx, sample)` のように branch 元 index を保持する
- `all_sorted` で GAE を計算した後、
  - `d_ret[branch_idx] = all_ret[sorted_idx]`
  - `d_adv[branch_idx] = all_adv[sorted_idx]`
  - `d_weights[branch_idx] = weight`
  - call 側も同様
  のように scatter する
- `d_ret` / `d_adv` / `d_weights` は `discard_samples` の元順と一致する tensor にする
- `c_ret` / `c_adv` / `c_weights` は `call_samples` の元順と一致する tensor にする
- semantic target tensor は現在どおり branch 元順で作ってよい
- GAE 自体は引き続き `(episode_id, player_id)` の same-player trajectory で計算する

#### 関連文書
- RL_SPEC.md
- `python/mahjong_rl/stage2a_learner.py`
- `python/mahjong_rl/stage2_selfplay_worker.py`
- `experiments/Stage02_CallUnlock/exp_015/report.md`
- `experiments/Stage02_CallUnlock/exp_019/report.md`

#### 受け入れ条件
- branch 内 `step_id` が非単調な discard/call sample を使った単体テストで、各 sample に正しい advantage / return が割り当たる
- discard と call が interleaved した trajectory でも、same-player grouped GAE の手計算と一致する
- `discard_samples[i]` の `reward/value/terminated` に基づく return が `d_ret[i]` に入ることをテストで確認する
- call 側も同様に `c_ret[i]` / `c_adv[i]` が元 sample と一致する
- mixed PPO の baseline / policy sample weight も元 sample 順と一致する
- 既存 Stage2a PPO smoke test が通る

#### 実装メモ
- これはハイパーパラメータ調整ではなく、PPO target の sample alignment 修正である
- `read_all()` や writer 側の保存順を前提にしない実装にする
- `step_id` は GAE の時系列順決定にのみ使い、branch tensor の index には使わない

実装結果:
- 変更ファイル: `python/mahjong_rl/stage2a_learner.py`, `tests/python/test_ppo_branch_targets.py`
- `Stage2aLearner._compute_ppo_branch_targets(discard_samples, call_samples, is_mixed)` helper 追加
  - all_indexed を `(step_id, branch, branch_idx, sample)` で持つ
  - step_id 順に sort してから `_compute_returns_advantages()` で GAE
  - sorted index → branch_idx 位置に scatter
  - mixed PPO の baseline_sample_weight も branch 元順
  - all_sorted / all_adv も返し、既存 mixed_ppo diagnostics と互換
- `_train_ppo` 内のサンプル順 append ロジックを helper 呼び出しに置換
- テスト: scatter 整合 6 件、全 passed
- 既存 PPO smoke / Stage2a model 全 98 件 passed
- CQ-0277 terminal weight 横断化は据置き

---

### CQ-0276
- Status: [Implemented]
- Type: RL | IO | Test
- Priority: Medium
- Title: Stage2a selfplay / eval に reward_config を伝播する

#### 背景
`Stage2Env` は `reward_config: RewardPolicyConfig | None` を受け取り、指定された場合は engine state に反映できる。

しかし現在の `Stage2SelfPlayWorker.generate()` は常に

```python
Stage2Env(observation_mode=self._obs_mode)
```

で環境を作っており、`config.reward` を `RewardPolicyConfig` に変換して渡していない。

また Stage2a parallel worker では `Stage2SelfPlayWorker(config={...})` ではなく `config={}` で worker を作っているため、仮に worker 側で reward config を読むようにしても、parallel path では設定が落ちる。

Stage1 では `reward.point_delta_scale` が selfplay / eval に伝播するテストがあり、報酬スケールの再現性を守っている。  
Stage2a でも同じ性質を保つべきである。

#### 要求内容
Stage2a の selfplay / imitation data generation / eval に `reward_config` を正しく伝播する。

具体的には:

- `Stage2SelfPlayWorker` が `config.reward` から `RewardPolicyConfig` を構築する
- `Stage2SelfPlayWorker.generate()` が `Stage2Env(..., reward_config=...)` を使う
- single-process runner path では `self._as_dict()` の `reward` が反映される
- multi-process `run_stage2a_selfplay_parallel()` にも reward config を渡せるようにする
- subprocess worker function が受け取った reward config を `Stage2SelfPlayWorker` に渡す
- Stage2a evaluator も必要なら同じ reward config を受け取れるようにする
  - evaluation の score/rank そのものは engine score から計算するため、主影響は reward-return 診断と selfplay shard 側

#### 関連文書
- RL_SPEC.md
- `python/mahjong_rl/env/stage2_env.py`
- `python/mahjong_rl/stage2_selfplay_worker.py`
- `python/mahjong_rl/stage2a_parallel.py`
- `python/mahjong_rl/stage2a_evaluator.py`
- `python/mahjong_rl/runner.py`
- `python/mahjong_rl/selfplay_worker.py`

#### 受け入れ条件
- Stage2a selfplay で `reward.point_delta_scale=0.0001` を指定した場合、shard reward が scale 済みになる
- single-process path と multi-process path の両方で reward scale が反映される
- `Stage2Env` に `RewardPolicyConfig` が渡っていることをテストで確認する
- reward config 未指定時は現行 default と互換になる
- Stage1 の reward config 伝播挙動は壊さない

#### 実装メモ
- 直近 Stage2a 実験では reward section が明示されていないため、過去結果への直接影響は限定的かもしれない
- ただし CQ-0274 修正後に reward 分布を評価するため、報酬スケールの伝播は明確にしておくべき

実装結果:
- 変更ファイル:
  - `python/mahjong_rl/stage2_selfplay_worker.py` (`build_reward_policy_config`, worker._reward_config, Stage2Env 渡し)
  - `python/mahjong_rl/stage2a_evaluator.py` (`reward_config` param, Stage2Env 渡し)
  - `python/mahjong_rl/stage2a_parallel.py` (`reward_config_dict` を worker fn / parallel runner 両方に追加)
  - `python/mahjong_rl/runner.py` (selfplay/eval/imitation 全 stage2a 経路に `reward_config_dict=dict(self._config.reward)` を渡す)
  - `tests/python/test_reward_config_propagation.py` (新規)
- single-process 経路: `Stage2SelfPlayWorker(config=self._as_dict(), ...)` 経由で `config["reward"]` が届く
- multi-process 経路: 明示的に `reward_config_dict` 引数で渡す
- reward_config 未指定時は `_reward_config = None` で Stage2Env の default 挙動維持
- テスト 11 件 (build helper 4 / worker 3 / evaluator 3 / parallel sig 1)

---

### CQ-0277
- Status: [Implemented]
- Type: RL | Training | Test
- Priority: Medium
- Title: terminal player-round 正規化を discard/call 横断で一貫させる

#### 背景
`CQ-0268` では terminal semantic loss の duplicated-label bias を抑えるため、同じ `(episode_id, round_id, player_id)` に属する row の terminal weight 合計を 1.0 にする player-round 正規化を導入した。

しかし現在の Stage2a learner では、

- discard samples だけで terminal weights を計算
- call samples だけで terminal weights を計算

している。

そのため、同じ player-round に discard と call の両方が存在する場合、terminal loss の合計が branch ごとに最大 1.0 ずつ入り、意図した「player-round 全体で合計 1.0」にならない。

また `_compute_semantic_aux_loss()` は `terminal_weights` が渡された場合に `tl = (tl_per * terminal_weights).sum()` としており、batch 内の weight 合計に loss scale が依存する。

これは policy PPO 本体ほど直接的なバグではないが、semantic aux の実効スケールと class balance を揺らし、terminal/value/semantic summary を通じて policy に影響しうる。

#### 要求内容
terminal player-round 正規化を discard/call 横断の同一母集団で計算し、loss scale を安定させる。

具体的には:

- PPO path では discard/call を結合した全 sample に対して `(episode_id, round_id, player_id)` count を計算する
- imitation tensor path でも可能な限り discard/call 横断で count を計算する
- 各 branch には元 sample 順に対応する terminal weight を渡す
- 同じ player-round に discard と call が混在しても、weight 合計が全体で 1.0 になる
- weighted terminal loss の reduction を明確にする
  - 現行の `sum()` を維持するなら、batch ごとの実効 loss scale が意図どおりか診断する
  - 必要なら `sum / weight_sum` などへ変更するが、既存 `terminal_loss_coef` との関係を明記する

#### 関連文書
- RL_SPEC.md
- `python/mahjong_rl/stage2a_learner.py`
- `tests/python/test_terminal_weights.py`
- `experiments/Stage02_CallUnlock/exp_015/report.md`

#### 受け入れ条件
- 同じ `(episode_id, round_id, player_id)` に discard 3件 + call 2件がある場合、5件の terminal weights 合計が 1.0 になる
- discard/call が別 branch に分かれていても、各 branch に渡る weights は元 sample と一致する
- 別 round / 別 player / 別 episode は別 group として扱われる
- weighted terminal loss の scale に関するテストを追加する
- 既存 semantic aux / terminal weight tests が通る

#### 実装メモ
- CQ-0275 の branch 元順 scatter と同じ補助構造を使うと実装しやすい
- CQ-0275 / CQ-0274 を先に直した後で対応してよい
- これは `CQ-0268` の意図をより厳密に満たす修正であり、feature 追加ではない

実装結果:
- 変更ファイル:
  - `python/mahjong_rl/stage2a_learner.py` (`_compute_terminal_weights_cross_branch` static helper 追加)
  - `tests/python/test_terminal_weights_cross_branch.py` (新規)
- 旧 `_compute_terminal_weights` は CQ-0268 の単 branch 用として残し、後方互換テスト維持
- PPO path: discard / call の (episode_id, round_id, player_id) を結合し、Counter で横断 count → branch 元順に scatter
- Imitation tensor path: 同様に cross-branch を `if d:` / `if c:` ブロック後に一括計算
- 片 branch のみのケースは旧 `_compute_terminal_weights` と一致 (`test_discard_only_matches_legacy`, `test_call_only_matches_legacy` で確認)
- `_compute_semantic_aux_loss` の `tl = (tl_per * terminal_weights).sum()` は維持 (loss scale を変えない)
- テスト 7 件 (cross-branch 6 / PPO smoke 1)

---
