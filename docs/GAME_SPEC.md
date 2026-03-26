# GAME_SPEC.md

この文書は、本プロジェクトにおける **ゲームエンジン側の長期実装仕様** を定義する。  
採用ルールは `GAME_RULE.md`、学習基盤と Stage ごとの運用境界は `RL_SPEC.md`、現在地と優先順位は `PROJECT.md` を参照する。

---

## 1. この文書の役割

この文書は次の 3 点だけを扱う。

1. ゲームエンジンがどこまでを責務として持つか
2. 現在のエンジンがどの程度まで実麻雀ルールを表現できるか
3. 今後の拡張で崩してはいけない設計原則は何か

この文書は、実験の優先順位や個別ハイパラを決める文書ではない。

---

## 2. 現在地

2026-03-27 時点で、ゲームエンジン側は次の意味で十分に立ち上がっている。

- 対局進行、合法手列挙、精算、観測生成は成立している
- `Chi / Pon / Daiminkan / Kakan / Ankan / Ron / Tsumo / Riichi` を含む通常の行動遷移をエンジン自体は扱える
- Debug / Fast の 2 モードを持ち、deterministic replay と state copy を前提にできる
- 現在の discard-only 制約は、主に Python 側の RL 境界 (`Stage1Env`, selector, shard, model) にある

したがって、次の主課題は engine の全面書き換えではなく、**Stage02a `CallUnlock` に向けて既存 engine 能力を RL 側へ安全に露出すること** である。

---

## 3. 設計原則

### 3.1 責務分離

ゲームエンジン側の責務:

- 対局進行
- 合法手列挙
- 和了 / 流局 / 終局判定
- 点数計算と精算
- 観測生成
- エンジン由来 reward の計算

学習基盤側の責務:

- feature 化
- policy / value 推論
- 行動選択
- self-play 保存
- learner / evaluation / batch 集約

### 3.2 変えてはいけない原則

- 同一 seed + 同一 action 列で同一結果を再現できること
- `EnvironmentState` の値コピーで完全複製できること
- Debug / Fast でルール結果が一致すること
- engine 内に学習都合の特殊分岐を増やしすぎないこと

---

## 4. 実行モード

### Debug

- 整合性チェックを優先する
- assert や検証コストを許容する
- バグ調査時の基準系とする

### Fast

- 高速 self-play / eval 用
- ログや重い検証を抑制できる
- ただしルール結果は Debug と一致しなければならない

---

## 5. 状態モデル

エンジンは少なくとも次の層を持つ。

- `PlayerState`
- `RoundState`
- `MatchState`
- `EnvironmentState`

要件:

- 状態は値コピー可能であること
- RNG 状態を状態複製に含むこと
- replay / 探索補助 / debug 再現の基礎になること

---

## 6. 牌・行動・ phase

### 6.1 牌表現

- 実際の局進行では 136 枚 ID を使う
- 判定・集計・学習補助では 34 種表現を使う
- 赤牌は実牌 ID として区別し、34 種上は通常 5 と同一視する

### 6.2 Action 表現

Action は単なる整数 ID ではなく、意味を持つ構造体として扱う。

最低限含むべき要素:

- `ActionType`
- actor
- tile
- target_player
- meld 情報
- 必要なら `riichi` フラグ

### 6.3 phase モデル

代表的な phase:

- `SelfActionPhase`
- `ResponsePhase`
- `ResolveResponsePhase`
- `ResolveWinPhase`
- `ResolveDrawPhase`
- `EndRound`
- `EndMatch`

現在の engine は、自摸側行動と応答側行動を phase machine として扱える前提である。

---

## 7. 合法手列挙

engine は、現在 phase に応じた合法手列挙を返す。

- `SelfActionPhase`: 打牌、立直打牌、暗槓、加槓、ツモ和了 など
- `ResponsePhase`: ロン、チー、ポン、大明槓、スキップ など

重要なのは、**Stage の簡略化は合法手列挙そのものではなく、Python 側 wrapper が何を学習対象として露出するかで制御する** ことである。

---

## 8. Observation

engine は少なくとも次の観測系をサポートする。

- `PartialObservation`
- `FullObservation`

観測には、学習側が次段で必要とする局面情報をすでに含められることが望ましい。

例:

- 手牌 / 河 / 公開副露
- 点数
- 巡目
- 現在 phase
- current player
- 直前打牌に関する文脈

Stage02a では、response decision 用の candidate 表現と組み合わせる前提で observation を使う。

---

## 9. 精算と reward

- engine は和了、流局、終局の判定と精算を担う
- point delta のようなエンジン起源の reward は engine 側で計算してよい
- learner 側の advantage 計算や shaping 合成は RL 側の責務とする

この分離は今後も維持する。

---

## 10. ログ / replay / determinism

engine は次を支える必要がある。

- seed 固定 replay
- state copy による再現
- Debug / Fast 比較
- 対局進行の最小限のトレース

今後 crash triage や multi-process 再現を進める際も、再現性を損なわない設計を優先する。

---

## 11. Stage01 と Stage02a の境界

### Stage01

- discard-only 学習
- 副露・和了判断は主に自動処理
- engine 能力の一部だけを RL 側に露出している状態

### Stage02a

- engine 自体は大きく変えず、response phase の legal candidate を Python 側へ露出する
- `Stage2Env` を新設し、`Chi / Pon / Daiminkan / Skip` を学習対象にする
- engine の role は引き続き stage-agnostic に保つ

---

## 12. 今後のエンジン側拡張

直近で予定する拡張:

- `対々和`
- `一気通貫`
- `三色同順`
- round outcome / yaku summary を Python 側へ運べる出力整備
- response candidate を Python 側で安全に扱うための binding 整備

優先しないもの:

- engine を discard-only 前提で最適化し続けること
- Stage ごとの学習都合を engine に埋め込むこと

---

## 13. 一文要約

**ゲームエンジンはすでに副露を含む麻雀進行を扱える水準にあり、今後はその能力を壊さずに RL 側へ段階的に露出する。**
