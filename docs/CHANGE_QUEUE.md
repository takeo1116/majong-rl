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


### CQ-0278
- Status: [Implemented]
- Type: RL | Training | Test
- Priority: High
- Title: Stage2a selfplay の policy sampling RNG と temperature を config に従わせる

#### 背景
Stage2a selfplay は `Stage2Env.reset(seed)` と seat assignment には seed を使っているが、policy actor の action sampling は `torch.multinomial` に依存しており、match seed で `torch.manual_seed(seed)` されていない。

また `Stage2SelfPlayWorker._policy_discard()` / `_policy_call()` は `select_*_sample(..., temperature=1.0)` を固定で呼んでおり、`selfplay.temperature` を設定しても Stage2a selfplay には反映されない。

Stage1 selfplay では match ごとに torch seed を設定し、temperature も config から読んでいる。Stage2a でも同等の再現性と探索制御を保つべきである。

#### 要求内容
Stage2a selfplay の policy sampling を config seed / temperature に従わせる。

具体的には:

- `Stage2SelfPlayWorker.__init__` で `config.selfplay.temperature` を読み、`self._temperature` に保持する
- config が flat dict / runner の `_as_dict()` のどちらでも読めるようにする
- default は現行互換の `1.0`
- `_policy_discard()` / `_policy_call()` の `temperature=1.0` 固定を `self._temperature` に置換する
- `generate()` の match loop 内で、match seed に基づいて `torch.manual_seed(seed)` を設定する
- CUDA device で policy sampling する可能性がある場合は `torch.cuda.manual_seed_all(seed)` も安全に設定する
- `np.random` を追加で使う場合は seed の衝突に注意する。既存の seat assignment は `np.random.RandomState(seed)` なので、global numpy seed 変更は必須ではない
- Stage1 selfplay の挙動は変更しない

#### 関連文書
- RL_SPEC.md
- `python/mahjong_rl/stage2_selfplay_worker.py`
- `python/mahjong_rl/stage2a_selector.py`
- `python/mahjong_rl/selfplay_worker.py`
- `configs/stage2a_*.yaml`

#### 受け入れ条件
- 同じ model / seed / config で Stage2a selfplay を 2 回実行したとき、policy sample の action sequence が一致することをテストで確認する
- seed を変えると action sequence が変わりうることをテストで確認する
- `selfplay.temperature` が `_policy_discard()` / `_policy_call()` に渡ることを単体テストで確認する
- `temperature=1.0` 未指定時は現行互換
- 既存 Stage2a selfplay / runner smoke test が通る

#### 実装メモ
- action sequence のテストは小さい fake model / monkeypatch でよい。実半荘を大量に回す必要はない
- `torch.manual_seed(seed)` は `env.reset(seed)` 直後、policy sampling 前に置くのが自然
- 次の再実験前に入れるべき再現性修正である

実装結果:
- 変更ファイル:
  - `python/mahjong_rl/stage2_selfplay_worker.py`
    - `__init__` で `config["selfplay"]["temperature"]` または `config["temperature"]` を読み `self._temperature` に保持 (default 1.0)
    - `generate()` の match loop で `env.reset(seed)` の直後に `torch.manual_seed(seed)`、CUDA 利用時は `torch.cuda.manual_seed_all(seed)` も呼ぶ
    - `_policy_discard()` / `_policy_call()` の `temperature=1.0` 固定を `self._temperature` に置換
  - `python/mahjong_rl/stage2a_parallel.py` (follow-up)
    - `_stage2a_selfplay_worker_fn` / `run_stage2a_selfplay_parallel` に `temperature` 引数追加 (default 1.0)
    - subprocess で `worker_config["selfplay"] = {"temperature": ...}` を組み立てて Stage2SelfPlayWorker に渡す
  - `python/mahjong_rl/runner.py` (follow-up)
    - 全 Stage2a parallel selfplay 呼び出し (selfplay / multi-chunk imitation 両方) で
      `temperature=float(sp_cfg.get("temperature", 1.0))` を渡す
  - `tests/python/test_stage2a_selfplay_rng.py` (新規 + follow-up テスト追加)
- 既存 seat assignment の `np.random.RandomState(seed)` は維持
- Stage1 selfplay は変更なし
- temperature 未指定時は default 1.0 で現行互換
- テスト 17 件 (config 5 / monkeypatch sampler 2 / torch reproducibility 2 / generate reproducibility 2 / parallel signature 2 / parallel worker config 2 / runner kwarg 1 / parallel run kwarg 1)

---

### CQ-0279
- Status: [Implemented]
- Type: RL | IO | Test
- Priority: High
- Title: Stage2a reward semantics 修正後の sample_semantics_version を v3 に上げて fail-fast する

#### 背景
CQ-0274 により Stage2a selfplay の reward backfill は、action owner の pending sample だけへの代入から、pending 中の全保存対象 player sample への累積に修正された。

これは `DecisionSample.reward` の意味を実質的に変える修正である。

しかし `DecisionSample.sample_semantics_version` は現在も v2 のままであり、CQ-0274 前に生成された旧 shard と CQ-0274 後に生成された新 shard を learner が区別できない。

旧 shard を誤って再学習に使うと、reward / return / advantage の意味が混在し、RL 実験の解釈を壊す。

#### 要求内容
Stage2a の現行 reward semantics を `sample_semantics_version=3` として明示し、learner / reader 側で旧 version を fail-fast する。

具体的には:

- CQ-0274 後の `DecisionSample` default version を v3 に上げる
- `Stage2SelfPlayWorker` が書き出す sample は v3 になるようにする
- Stage2a learner は v3 未満の shard を原則拒否する
- 既存 unit test 用 dummy shard は、現行 semantics を期待するものだけ v3 に更新する
- semantic-only eval など、reward を使わない純 diagnostics path で旧 shard を読む必要がある場合は、明示的な opt-in を設けるか、今回 scope 外として fail-fast でよい
- error message には検出した version と「CQ-0274 前の旧 reward semantics shard である可能性」を含める

#### 関連文書
- RL_SPEC.md
- `python/mahjong_rl/call_shard.py`
- `python/mahjong_rl/stage2_selfplay_worker.py`
- `python/mahjong_rl/stage2a_learner.py`
- `tests/python/test_learner.py`
- `tests/python/test_call_shard.py`

#### 受け入れ条件
- 新規 Stage2a selfplay shard の `sample_semantics_version` が 3 になる
- Stage2a learner が v2 shard を読むと明確な `ValueError` で停止する
- v3 shard は正常に learner が読める
- 既存 dummy shard / tests は v3 に更新される
- 旧 Stage1 `LearningSample` の semantics version と混同しない

#### 実装メモ
- CQ-0274 は reward semantics の修正なので、version bump は後追いだが必須
- `DecisionShardReader.read_as_tensors()` と `read_all()` のどちらの learner path でも検出できるようにする
- 実験ディレクトリ内の過去 shard を自動変換しない。必要なら再生成する

実装結果:
- 変更ファイル:
  - `python/mahjong_rl/call_shard.py`
    - `DecisionSample.sample_semantics_version` default を `2` → `3`
    - `DecisionShardReader.read_as_tensors()` の discard / call 両 dict に
      `sample_semantics_versions` (int64 array) を追加。column 欠如時は 0
    - `DecisionShardReader.read_all()` の `sample_semantics_version` 取得を
      `_col_safe` 風に修正し、column 欠如時は 0 に倒す (旧 shard 互換)
  - `python/mahjong_rl/stage2a_learner.py`
    - `Stage2aLearner.REQUIRED_SAMPLE_SEMANTICS_VERSION = 3`
    - `_check_sample_semantics_version()` classmethod で fail-fast
    - `train()` の入口で discard / call の両 tensor dict を検証
    - PPO path の `read_all()` 後にも min(sample_semantics_version) で fail-fast
    - error message に detected version / required (3) / CQ-0274 言及 / 再生成指示
  - `tests/python/test_sample_semantics_v3.py` (新規)
- Stage2a selfplay (`Stage2SelfPlayWorker`) は `DecisionSample` の default を
  使うため、自動的に v3 shard を吐く (worker の追加変更なし)
- Stage1 `LearningSample` (default v=1) は今回 scope 外で変更なし
- 既存 Stage2a tests は default 利用なので自動的に v3 になり影響なし
- テスト 14 件 (default 3 / selfplay v3 / read_as_tensors versions /
  imitation v2 reject / PPO v2 reject / v3 pass / mixed reject / error
  message / Stage1 unaffected)

---
