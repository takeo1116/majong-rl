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

### CQ-0280
- Status: [Proposed]
- Type: Eval | Training | Test
- Priority: Medium
- Title: Stage2a multi-cycle eval worker failure retry

#### 背景
`experiments/Stage02_CallUnlock/exp_021` の continuation 中に、Stage2a multi-cycle の cycle eval で `eval_worker_9: non-zero exit code -11` が発生し、run 全体が learner failure として停止した。

学習本体は該当 cycle の learner/checkpoint 保存まで進んでおり、失敗箇所は checkpoint 評価用の eval subprocess だった。発症率は低そうだが、長時間実験では 1 回の eval worker crash で run 全体を捨てるのは損失が大きい。

#### 要求内容
Stage2a multi-cycle の cycle eval で worker failure が発生した場合、同じ checkpoint/model state に対して eval だけ retry する。

- retry 対象は eval のみとし、selfplay / learner は再実行しない
- 初回 eval は既存設定の `evaluation.num_workers` を使う
- eval failure 時は同じ checkpoint/model state で retry する
- retry worker 数は config で指定可能にする
  - 例: `evaluation.retry_num_workers`
  - 未指定時は安全側の fallback として `1` を使う
- retry 回数は config で指定可能にする
  - 例: `evaluation.retry_attempts`
  - default は `1`
- retry も失敗した場合は従来通り run を failed にしてよい
- retry で成功した場合は run を継続し、cycle entry に retry が発生した事実を記録する

#### 関連文書
- experiments/Stage02_CallUnlock/exp_021/runbook.md
- python/mahjong_rl/runner.py
- python/mahjong_rl/stage2a_parallel.py

#### 受け入れ条件
- Stage2a multi-cycle の eval が 1 回失敗しても、retry が成功すれば次 cycle に進む
- retry 時に selfplay / learner が再実行されないこと
- `cycles[*].eval_metrics` または隣接する diagnostics に、retry 発生回数・初回 error・retry worker 数が残ること
- retry も失敗した場合は error message に初回 failure と retry failure の両方が含まれること
- unit test で `_run_eval_stage2a` の初回 failure → fallback success を確認すること
- unit test で fallback も failure → run failure を確認すること

#### 実装メモ
exp_021 の暫定 driver は batch-level continue-on-error にしているが、これは run 内の eval failure を救済しない。CQ-0280 は runner 本体側の retry 機構として実装する。

---

### CQ-0281
- Status: [Implemented]
- Type: RL | Training | Test
- Priority: High
- Title: Stage2a PPO diagnostics quantile expansion

#### 背景
`exp_022` の no-anchor 60 cycle 実験では、`cycle40` 以降に avg_rank が明確に崩壊し、同時に entropy collapse と `clip_fraction` 高止まりが見られた。

しかし現在の PPO diagnostics は `ratio_mean` / `ratio_std` / `clip_fraction` / `advantage_mean` / `advantage_std` が中心であり、崩壊時には `ratio_mean` が外れ値で壊れる。

実際に `exp_022` では `ratio_max` が非常に大きくなり、`ratio_mean` が診断指標として使いにくくなった。次の entropy_coef 実験や、その後の根本原因分析に備えて、学習挙動を変えずに PPO diagnostics を強化する。

#### 要求内容
Stage2a PPO learner の diagnostics に、以下 5 系統の統計を追加する。

1. `log_ratio` quantiles
   - `log_ratio = new_log_prob - old_log_prob`
   - `mean`, `std`, `min`, `max`
   - `p01`, `p05`, `p50`, `p95`, `p99`
   - 既存 `ratio_mean` / `ratio_std` / `clip_fraction` は維持する

2. `advantage` sign / quantiles
   - `advantage_pos_frac`
   - `advantage_neg_frac`
   - `advantage_zero_frac`
   - `advantage_abs_mean`
   - `advantage_p01`, `p05`, `p50`, `p95`, `p99`
   - 既存 `advantage_mean` / `advantage_std` は維持する

3. `advantage × log_ratio` cross stats
   - `log_ratio_mean_adv_pos`
   - `log_ratio_mean_adv_neg`
   - `ratio_mean_adv_pos`
   - `ratio_mean_adv_neg`
   - `clip_fraction_adv_pos`
   - `clip_fraction_adv_neg`
   - `num_adv_pos`
   - `num_adv_neg`

4. policy confidence / `max_prob` quantiles
   - 各 sample の合法 action 上の最大確率を集計する
   - `max_prob_mean`
   - `max_prob_p50`
   - `max_prob_p90`
   - `max_prob_p95`
   - `max_prob_p99`

5. branch別 diagnostics
   - discard / call を分けて、少なくとも以下を出す
     - `discard.log_ratio_*`
     - `discard.advantage_*`
     - `discard.clip_fraction`
     - `discard.max_prob_*`
     - `call.log_ratio_*`
     - `call.advantage_*`
     - `call.clip_fraction`
     - `call.max_prob_*`
   - top-level には従来互換の aggregate diagnostics を残す

#### 関連文書
- experiments/Stage02_CallUnlock/exp_022/report.md: 作成予定
- experiments/Stage02_CallUnlock/exp_021/report.md
- python/mahjong_rl/stage2a_learner.py

#### 受け入れ条件
- 学習 loss / optimizer / sampling 挙動を変えないこと
- `summary.json` の `phase_stats.cycles[*].learner_metrics.ppo_diag` に新 diagnostics が保存されること
- 既存 diagnostics key は削除しないこと
- discard-only / call-only / discard+call mixed の全ケースで動作すること
- advantage が片符号のみの batch でも crash しないこと
- unit test で以下を確認すること
  - log_ratio quantiles が出る
  - advantage sign fractions が合計 1.0 になる
  - adv_pos / adv_neg の cross stats が符号ごとに分かれる
  - max_prob quantiles が `[0, 1]` 範囲に入る
  - branch別 diagnostics が discard / call で別々に出る

#### 実装メモ
ratio は外れ値で平均が壊れやすいため、原因分析では `log_ratio` quantile を主指標にする。既存の `ratio_mean` は後方互換のため残すが、report では今後 `log_ratio_p95/p99` と `max_prob_p95/p99` を重視する。

実装結果:
- 変更ファイル:
  - `python/mahjong_rl/stage2a_learner.py`
    - helper 追加: `_safe_np_quantiles`, `_weighted_mean`,
      `_weighted_fraction`, `_compute_ppo_diag_stats` (classmethod)
    - `_ppo_discard_epoch` / `_ppo_call_epoch` に diagnostics 用 buffer
      (`log_ratios`, `advantages`, `max_probs`, `weights`) を追加。
      detach + cpu で回収して GPU tensor を長く保持しない
    - `max_prob` は legal_mask / cand_mask 適用後の softmax sample-wise max
    - `_train_ppo` で branch buffer を集約し `ppo_diag["discard"]` /
      `ppo_diag["call"]` を生成、top-level も拡張 statistics で埋める
    - 既存 `ratio_mean` / `ratio_std` / `clip_fraction` / `advantage_mean` /
      `advantage_std` は上書きせず維持
  - `tests/python/test_stage2a_ppo_diagnostics.py` (新規)
- 学習挙動 (loss / optimizer / sampling) は不変。
  全 diagnostics は `with torch.no_grad()` か `.detach()` で計算
- mean / fraction は weighted、quantile は unweighted (helper docstring 明記)
- 片符号 advantage / 空 branch / one-sided でも crash せず、
  対応 cross stats は None
- テスト 19 件 (helper unit 11 / PPO smoke 5 / branch split 1 / 既存 key
  維持 1 / json serializable 1)
- 検証: pytest tests/python -k "(stage2a or ppo_diag or ppo_branch)"
  → 160 passed
