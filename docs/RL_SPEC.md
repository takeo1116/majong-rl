# RL_SPEC.md

この文書は、本プロジェクトにおける **学習基盤・評価基盤・実験基盤の長期実装仕様** を定義する。  
現在の優先順位と段階判断は `PROJECT.md`、ゲームルールは `GAME_RULE.md`、ゲームエンジン側の責務は `GAME_SPEC.md` を参照する。

---

## 1. この文書の役割

この文書は次の 4 点を扱う。

1. RL 側がどこまでを責務として持つか
2. Stage01 で確立した現仕様は何か
3. Stage02a へ拡張するとき、どの境界を壊し、どの境界を維持するか
4. 成果物・診断・ sample semantics をどう追跡するか

この文書は、細かな実験履歴の置き場ではない。

---

## 2. 現在地

2026-03-27 時点で、RL 基盤は次の状態にある。

- Stage01 `DiscardOnly` の imitation / self-play / PPO / evaluation 基盤は成立している
- `danger_mask` を含む防御特徴により、FullObservation では PPO が imitation を安定して超えうることを確認済み
- したがって、現在の主課題は「PPO が伸びるか」ではなく、**次の行動境界をどう自然に解放するか** である
- 次の主戦場は Stage02a `CallUnlock`

Stage01 の current best は `experiments/Stage01_DiscardOnly/exp_070/report.md` の `C context_plus_danger` を基準とする。

---

## 3. RL 基盤の責務

RL 側の責務:

- observation を feature に変換する
- policy / value を推論する
- legal action から行動を選ぶ
- self-play / imitation データを保存する
- learner / evaluation / batch 集約を行う
- 実験条件と成果物を追跡する

RL 側の責務ではないもの:

- 対局進行そのもの
- 点数計算や和了判定そのもの
- ゲームルール定義そのもの

---

## 4. 現仕様: Stage01 `DiscardOnly`

### 4.1 学習対象

現在の正式対象は `DiscardPolicy` のみ。

- 学習対象行動: 自摸直後の打牌
- 副露なし
- `Ron / Tsumo / Riichi` は自動処理
- 九種九牌なし
- 通常流局あり

### 4.2 観測・特徴・モデル

Stage01 の主系は以下。

- `FullObservation`
- `FlatFeatureEncoder`
- `MLPPolicyValueModel`
- 34-way discard logits + legal mask

利用実績のある補助特徴:

- `shanten_hint`
- `discard_ukeire_hint`
- `current_shanten`
- `shape_hint`
- `opponent_current_shanten`
- `opponent_tenpai_flag`
- `danger_mask`

重要なのは、これらのうち **どれが良いかをさらに Stage01 で掘り続けるより、次段でどの情報をどう使うかに進むこと** である。

### 4.3 reward / sample semantics

Stage01 の主系 reward は point delta ベースで、必要に応じて shaping を合成する。

Stage01 の学習サンプルは、単なる環境 1 step ではなく、**同一 actor の次の decision までを意識した decision 単位** として扱う。  
この same-player semantics は、現在の advantage 解釈の基礎なので維持する。

補足:

- cross-round の継続は sample 側で追跡する
- reward shaping は主タスクを置き換えるものではなく補助信号である
- 旧 shard との互換は必要最小限に留め、意味が変わる場合は fail-fast を優先する

### 4.4 imitation / PPO / evaluation

Stage01 では次の流れが成立している。

- baseline self-play から imitation warm start を作る
- self-play + PPO で更新する
- rotation eval を主評価に使う
- summary / batch_summary に診断を残す

imitation teacher は、現状では baseline actor が実際に選んだ action と一致している。

---

## 5. 現仕様: 成果物と追跡

最低限追跡する成果物:

- `config.yaml`
- `summary.json`
- `checkpoints/`
- `batch_summary.json`
- 実験ディレクトリ配下の `runbook.md` / `report.md`

summary / batch summary には、少なくとも次を残せることが望ましい。

- 主要 score 指標
- imitation / PPO 診断
- teacher 再現指標
- reward shaping 設定
- optional feature / direct hint 設定

multi-process 実行では、異常終了時の triage に必要な sidecar / heartbeat 情報も保つ。

---

## 6. Stage02a `CallUnlock` で壊す境界

Stage02a では、discard-only 前提の次を壊す。

- `Stage1Env`
- 34-way discard 固定の action / legal mask 前提
- `action:int + legal_mask(34)` に寄った shard 形式
- discard head のみを前提にした selector / learner

ここは Stage02a の本丸であり、単なる feature 追加ではない。

---

## 7. Stage02a の方針

### 7.1 対象行動

追加対象:

- `Chi`
- `Pon`
- `Daiminkan`
- `Skip`

当面は自動処理のまま:

- `Ron`
- `Tsumo`
- `Riichi`
- `Ankan`
- `Kakan`
- `Kyuushu`

### 7.2 モデル構造

Stage02a では policy を概念上分ける。

- `DiscardPolicy`
- `CallPolicy`

初手設計:

- 共通なのは observation / feature encoder まで
- learned trunk は分ける
  - `discard_trunk`
  - `call_trunk`
- `CallPolicy` は legal candidate ごとの scalar score を出す
- `Skip` も candidate の 1 つとして比較する
- 同一 response decision 内では `h_call` を 1 回だけ計算し、candidate 間で再利用する

長期的には、Stage02 系を今後の標準アーキテクチャとみなし、decision family を次の 2 系統へ整理する。

- `DISCARD`
- `OPTIONAL_ACTION`

ここでいう `OPTIONAL_ACTION` は、「その場で legal なら、実行するか `Skip` するかを比較できる行動」の総称である。  
現在の `CallPolicy` はその最初の具体化であり、将来の

- `Chi`
- `Pon`
- `Daiminkan`
- `Riichi`
- `Ankan`
- `Kakan`
- `Ron`
- `Tsumo`
- `Kyuushu`

は、原則として別 head を増やすのではなく、**optional candidate の語彙と metadata を拡張して同じ経路へ載せる**方針を優先する。  
実装名として当面 `call` が残っていてもよいが、設計上は `forward_optional` 相当の一般化を目標にする。

### 7.3 特徴量方針

Stage02a v1 では次を採る。

- shared state 側は Stage01 trunk 用の重い局面特徴を基本流用
- response 固有文脈は `response_context` として call 側に持つ
  - `last_discard_tile_type`
  - `discarder_relative_seat`
  - `my_menzen_flag`
- candidate は compact action feature で表現する
- `phase` one-hot は trunk に入れない
- `danger_mask` を call 側へ直接入れない
- 副露後打牌を読む advanced summary は後回し

### 7.4 baseline / imitation

Stage02a v1 では、actor と teacher を分けない。

- `RuleBasedCallPolicy.select_action()` が選んだ action を、そのまま imitation ラベルに使う
- `abstain` を含む teacher 分離は後段の設計課題とする

---

## 8. Stage02a のデータ表現

Stage02a では、固定 34-way discard 決め打ちから離れて、**decision 単位 + candidate 集合** を表現できる shard 形式へ移る。

推奨する考え方:

- 共通の `DecisionRecord`
- head 別 payload
- variable-length candidate table

最低限ほしい情報:

- `decision_type`
  - 現実装では `discard`, `call` などを持ちうる
  - 長期的には `discard`, `optional_action` の 2 family へ寄せる
- selected action / selected candidate
- legal candidates
- actor 視点 reward / return / advantage

この設計は、将来の `Riichi` / `Kan` / `Win` を別 pipeline として増やすのではなく、optional candidate の拡張として吸収できることを目指す。

---

## 9. round outcome / yaku labels

将来の diagnostics と auxiliary head に備え、Stage02a では round outcome を Python 側で持ち上げられる設計を採る。

### 9.1 round-level summary

少なくとも次を持てることを目指す。

- `round_end_reason`
- winner / loser
- tenpai / noten 情報
- `yakus`
- `total_han`
- `fu`

### 9.2 sample-level future labels

各 decision sample に actor 視点の terminal label を付ける。
全 sample が 1 つの label を持つ (total terminal label)。

- `round_terminal_label` — actor 視点の局結果
  - `win` — この actor が和了した
  - `ron_loss` — この actor が放銃した
  - `ron_bystander` — ron 和了局の第三者 (winner でも放銃者でもない)
  - `tsumo_loss` — 他家がツモ和了した
  - `ryukyoku_tenpai` — 流局テンパイ (副露考慮判定)
  - `ryukyoku_noten` — 流局ノーテン (副露考慮判定)
  - `abortive_draw` — 途中流局 (九種九牌等)
- `eventual_win_yaku_ids`
- `eventual_total_han`
- `eventual_fu`

初期段階では学習に使わなくてもよいが、後から無理なく診断や `yaku head` に使えることを重視する。

---

## 10. 評価と回帰基準

- Stage01 は今後も regression harness として残す
- Stage02a を進めても、Stage01 の sanity check は維持する
- 新機能追加時は
  - unit
  - integration
  - small end-to-end
  の 3 層で確認する

特に Stage02a では、次を最低限確認したい。

- `Stage2Env` が legal candidate を正しく返す
- baseline call policy で self-play / imitation データが作れる
- call policy を含む learner/eval が smoke レベルで動く

---

## 11. 実装上の優先順位

Stage02a の推奨順序:

1. 役追加と response candidate の露出
2. `Stage2Env` の新設
3. shard/schema 拡張
4. `RuleBasedCallPolicy` と imitation/self-play 生成
5. `DiscardPolicy` / `CallPolicy` 分離モデル
6. learner / selector / PPO / eval 統合
7. round outcome / yaku summary
8. smoke config / 最小 end-to-end 確認

---

## 12. Stage02a: CallUnlock (CQ-0220〜CQ-0222)

### 12.1 追加役 (CQ-0220)

Stage2a で追加された役:

| 役 | 門前翻 | 食い下がり | 備考 |
|---|---|---|---|
| 対々和 | 2 | 2 | 全刻子/槓子。門前/副露問わず同翻 |
| 一気通貫 | 2 | 1 | 同一スートで 123+456+789 |
| 三色同順 | 2 | 1 | 3スートで同数字の順子 |

### 12.2 Response Candidate API (CQ-0221)

- `extract_response_candidates(legal_actions, current_player)` で response phase の候補を列挙
- `ResponseCandidate`: action, action_type, tile_type, target_rel_seat, consumed_tile_ids
- `Skip` は通常 candidate と同列（末尾に配置）
- Stage2a 学習対象: `Chi / Pon / Daiminkan / Skip`

### 12.3 Stage2Env (CQ-0222)

- `Stage1Env` とは別の環境ラッパー
- `decision_type`: `"discard"` / `"response"` を区別
- `step_discard(tile_type)`: 打牌決定
- `step_response(candidate_index)`: 応答決定
- 自動処理: Ron, TsumoWin, Riichi, Ankan, Kakan, Kyuushu
- candidates が Skip のみなら自動スキップ

---

## 13. 一文要約

**Stage01 で PPO 基盤は成立したため、今後の RL 仕様の中心課題は discard-only 前提を壊して Stage02a の call decision を自然に扱えるようにすることにある。**
