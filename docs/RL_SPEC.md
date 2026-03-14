# RL_SPEC.md

この文書は、このプロジェクトにおける **学習基盤・評価基盤・実験基盤の長期実装仕様** を定義する。  
日常の実験判断や優先順位は [PROJECT.md](/home/takeo1116/Git/majong-rl/docs/PROJECT.md) を正とし、個別実験の条件と結果は `experiments/exp_xxx/runbook.md` / `report.md` を正とする。  
本書はそれらの土台になる **RL システムの設計基準** を扱う。

ゲームルールそのものは `GAME_RULE.md`、ゲームエンジン仕様は `GAME_SPEC.md`、研究上の方針は `RL_RULE.md` を参照する。

---

## 1. この文書の役割

この文書の用途は次の3つに限定する。

1. RL 基盤を変更する CQ の設計基準にする  
2. 現在実装済みの学習・評価・実験基盤の境界を明確にする  
3. 将来拡張時に、どこまでが現仕様でどこからが構想かを分けて記録する  

この文書は「今すぐ実験でどう動くか」を毎回説明する文書ではない。  
その役割は `PROJECT.md` と各 runbook/report が担う。

---

## 2. 文書の読み方

本書では内容を次の2層に分ける。

- **現仕様**
  - 現在のコードベースで成立している実装仕様
  - レビュー時に「実装とズレていないか」を確認する対象
- **将来案**
  - まだ未実装だが、今後の設計余地として維持したい方向
  - 現実装への適用を前提に読まない

実装修正を伴う議論では、まず現仕様を優先し、将来案は必要なときだけ参照する。

---

## 3. スコープ

### 3.1 この文書が扱うもの

- observation / feature / model の境界
- self-play / imitation / learner / evaluation の責務
- shard / checkpoint / summary / batch_summary の成果物仕様
- 実験設定と run 成果物の最低限の追跡仕様
- 現在の Stage 1 学習基盤

### 3.2 この文書が主対象にしないもの

- ゲームルールの細部
- 実験優先順位そのもの
- 個別のハイパラ採否
- GUI / MLOps / クラウド運用

---

## 4. 現在地の要約

2026-03-08 時点の RL 基盤は次の段階にある。

- Stage 1（DiscardOnly）の学習基盤は成立
- imitation / self-play / PPO / evaluation / batch 集約 / phase 再利用 / resume は実装済み
- `shanten_hint`、`teacher_best_mask`、`tie_aware_best_set`、`ppo_diag` などの診断補助は実装済み
- 現在の主要課題は **PPO が imitation 初期方策を平均的に悪化させること**

したがって本書は、学習基盤の「立ち上げ仕様」ではなく、  
**診断と拡張に耐える基盤仕様** として維持する。

---

## 5. システム全体構成

学習システムは次の層に分ける。

1. **Game Engine Layer (C++)**
   - 対局進行
   - 合法手列挙
   - 観測生成
   - 精算・reward 計算
2. **Binding Layer**
   - Python から C++ の環境を呼び出す境界
3. **Training Layer (Python)**
   - FeatureEncoder
   - Model
   - Imitation / PPO learner
   - shard 読み込み
4. **Self-Play / Evaluation Layer**
   - self-play worker
   - baseline 対戦
   - rotation evaluation
   - 集計
5. **Experiment Management Layer**
   - YAML config
   - run directory
   - checkpoint / metrics / notes / summary
   - batch_summary / batch_table

責務の原則:

- ゲーム進行・精算は C++
- 学習・集計・実験運用は Python
- observation / feature / model / action selection を密結合にしない

---

## 6. 現仕様: 学習対象

### 6.1 Stage 1 の正式対象

現在の正式対象は **DiscardOnly** である。

- 学習対象行動: 自摸直後の打牌のみ
- 副露なし
- ロン / ツモ和了は自動
- テンパイ時自動立直
- 九種九牌なし
- 通常流局あり

### 6.2 Policy 抽象

将来的には policy を分割可能とする。

- `DiscardPolicy`
- `CallPolicy`
- `RiichiPolicy`
- `WinPolicy`

ただし現仕様で学習しているのは `DiscardPolicy` のみ。

---

## 7. 現仕様: Observation / Feature / Model

### 7.1 原則

Observation、Feature 表現、Model は分離する。

- **Observation**: エンジン由来のモデル非依存情報
- **FeatureEncoder**: Observation をモデル入力へ変換
- **Model**: 特徴量から policy / value を出力
- **ActionSelector**: legal mask を適用して行動を選ぶ

### 7.2 Observation モード

現仕様で扱う observation mode:

- `full`
- `partial`

運用上の主系は `full`。  
`partial` は比較・将来拡張のために維持する。

### 7.3 FeatureEncoder

現仕様で利用可能:

- `FlatFeatureEncoder`
- `ChannelTensorEncoder`

ただし現在の主力実験は `FlatFeatureEncoder`。

共通 API (CQ-0173):

- `encode(obs, *, legal_mask=None) -> np.ndarray`
- `legal_mask` は optional。利用するかどうかは各 encoder 実装の責務。
- `FlatFeatureEncoder` は `discard_ukeire_hint` 有効時のみ `legal_mask` を利用する。
- `ChannelTensorEncoder` は `legal_mask` を受け取るが無視する。

#### 7.3.1 FlatFeatureEncoder のオプション特徴量

以下のオプションはすべて既定 `false` で、`true` 時のみ特徴量が追加される。
`false` 時は既存入力次元を維持する（完全後方互換）。

**shanten_hint** (`feature_encoder.shanten_hint.enabled`)

- `delta_shanten_sign[34]` を追加（+34次元）
- 各打牌候補について、打牌後シャンテン変化の符号を補助特徴として持つ

**discard_ukeire_hint** (`feature_encoder.discard_ukeire_hint.enabled`) (CQ-0168, CQ-0172)

- `discard_ukeire_norm[34]` を追加（+34次元）
- 各打牌候補の受け入れ枚数を局面内 max で正規化した値 ([0, 1])
- 非合法打牌候補は 0.0: `legal_mask` が `encode()` に渡された場合は mask 基準、未指定時は手牌カウント基準
- selfplay / evaluator の `encode()` 呼び出しでは `legal_mask=mask` を渡しており、立直後などの非合法候補も正しく 0.0 になる
- 受け入れ枚数の定義は `RuleBasedBaseline._count_acceptance()` と整合

**current_shanten** (`feature_encoder.current_shanten.enabled`) (CQ-0169)

- `current_shanten / 8.0` を追加（+1次元）
- policy/value 共通 trunk 入力として共有される
- 既存の `model.value_features.current_shanten.enabled`（value_aux）との併用可
  - 併用時は trunk 入力と value_aux の両方に同等の情報が入る

**shape_hint** (`feature_encoder.shape_hint.enabled`) (CQ-0170)

- 手牌形状ヒント `closed_chi[21] + closed_outside_wait[24] + closed_inside_wait[21]` を追加（+66次元）
- 手牌のみ対象（副露情報は除外）
- binary multihot（有無のみ）
- 各スートの数牌について: 順子(中心牌2-8)、塔子(隣接ペア12~89)、嵌張(中心牌2-8)を検出

現運用上の注意:

- これらはあくまで補助特徴であり、教師方策そのものを直接埋め込むものではない
- 比較実験では各オプションの on/off を config 上で明示的に追跡する
- 全オプション有効時の追加次元: +34+34+1+66 = +135

### 7.4 Model

現仕様での主系:

- `MLPPolicyValueModel`
- FullObservation + FlatFeatureEncoder + MLP

現在の標準的な value head は:

- `round_delta`

複数 value head を許容する設計だが、日常実験の主系はまだ単一 head 前提に近い。

#### 7.4.1 value head 専用補助特徴 (CQ-0151)

`MLPPolicyValueModel` は `value_aux_dim` パラメータをサポートする。

- `value_aux_dim > 0` の場合、value heads の入力次元は `trunk_out + value_aux_dim` になる
- `forward(features, legal_mask, value_aux_features=None)` で追加特徴を渡す
- `value_aux_features` が与えられた場合、trunk 出力 `h` と concat してから value heads に渡す
- `value_aux_features=None` かつ `value_aux_dim > 0` の場合、zero pad して value head 入力次元を合わせる (CQ-0153)
- policy head は trunk 出力のみを使用し、value_aux_features の影響を受けない
- evaluator 推論経路でも `value_shanten_enabled=true` 時は current_shanten を計算して渡す (CQ-0153)

現在サポートする補助特徴:

| 特徴名 | dim | config | 正規化 |
|---|---|---|---|
| current_shanten | 1 | `model.value_features.current_shanten.enabled` | raw / 8.0 |

config 例:
```yaml
model:
  value_features:
    current_shanten:
      enabled: true
```

#### 7.4.2 task-specific tower (CQ-0157)

shared trunk の後に task-specific tower (1 hidden layer) を追加できる。

- `policy_tower`: enabled 時、trunk 出力に `Linear(trunk_out, hidden_dim) + ReLU` を挟んでから policy head に渡す
- `value_tower`: enabled 時、trunk+aux 出力に `Linear(trunk_out + value_aux_dim, hidden_dim) + ReLU` を挟んでから value heads に渡す
- 両方 off で現行構造を完全維持（後方互換）
- `current_shanten` は value tower の入力に含まれる（policy tower には入らない）

config 例:
```yaml
model:
  policy_tower:
    enabled: false
    hidden_dim: 128
  value_tower:
    enabled: false
    hidden_dim: 128
```

成果物追跡:
- `summary.json.model_features.policy_tower.{enabled, hidden_dim}`
- `summary.json.model_features.value_tower.{enabled, hidden_dim}`
- `batch_summary.json.runs[*].model_features` に同上

実験方針注記:
- tower 比較実験では `current_shanten=true` を全条件で固定し、差分を tower 構造だけにする
- `exp_024 D` は旧診断世代（CQ-0155/0156 以前）のため今回の baseline に使わず、small model + `current_shanten=true` を新規取得する

---

## 8. 現仕様: Action / legal mask

Stage 1 では policy は **34 種の打牌ロジット** を出す。  
legal mask により、実行可能な打牌へ射影する。

現仕様:

- action space は牌種ベース
- 実際の `Action` との対応付けは env / selector 側が行う
- legal mask は feature と分離して扱う

---

## 9. 現仕様: reward

### 9.1 現在の主系 reward

現在の主系 config は次である。

- `reward.type = "point_delta"`
- `reward.point_delta_scale = 0.0001`

意味:

- 点数変化に比例した reward を返す
- 学習では scale を掛けた値を用いる

### 9.2 reward config の適用経路 (CQ-0162)

`reward.point_delta_scale` は C++ エンジン (`Stage1Env`) の `RewardPolicyConfig` を通じて
env 内部で reward 計算時に適用される。すべての `Stage1Env` 生成経路で `reward_config` を渡す必要がある。

適用経路:
- **self-play**: `selfplay_worker.py` が config["reward"] から `RewardPolicyConfig` を構築し `Stage1Env` に渡す
- **evaluation**: `runner.py` の `_eval_worker_fn` が `reward_config_dict` から `RewardPolicyConfig` を構築し `EvaluationRunner` に渡す。`EvaluationRunner` が `Stage1Env` に転送する
- **imitation**: selfplay_worker が self-play と同じ config を使うため同一経路

`reward_config` を渡さない場合、C++ デフォルト `point_delta_scale=1.0` が使われ、
learner に流れる reward の単位が config 設定と不整合になる。

### 9.3 Stage 1 での実際の挙動

運用上の理解としては次の通り。

- 局中の大半の打牌 step の reward は 0
- 和了 / 放銃 / 流局精算 / リーチ棒支払い / 半荘終了時など、点数変化が起きた step に非ゼロ reward が乗る
- learner はその系列 reward から GAE を計算して advantage / return を作る

したがって現 reward は、
**点数変化ベースの sparse な即時報酬**
として扱うのが正確である。

### 9.4 現仕様での reward 関連の既知課題

- sparse
- tail が強い
- 打牌ごとの credit assignment が粗い

これは現在の `PROJECT.md` にある主要診断課題の一つであり、  
reward 設計変更を伴う CQ ではこの節を起点に議論する。

### 9.5 Reward Shaping (CQ-0139, CQ-0140)

Python `selfplay_worker` 内で shaping reward を計算し、`point_delta` と合成した total を `LearningSample.reward` に格納する。
shard / learner 側の構造変更はなし（reward は単一 float のまま）。

#### composition
```
total_reward = point_delta_reward + shaping_reward
```

#### shanten_delta_reward
- 各プレイヤーの打牌後 13 枚手牌のシャンテンを追跡
- `delta = prev_shanten − current_shanten` （正 = 改善）
- 初回打牌は delta = 0
- `mode="both"`: 改善 `+scale*delta`, 悪化 `-scale*|delta|`, 維持 `0`
- `mode="improve_only"`: 改善 `+scale*delta`, 悪化/維持 `0`
- match 単位で tracker をリセット

#### schedule
- `constant`: factor = 1.0
- `linear_decay`: factor = 1.0 − progress （progress = match_index / num_matches）
- 有効 scale = base_scale × factor

#### config
```yaml
reward:
  shaping:
    shanten_delta:
      enabled: false     # デフォルト off（後方互換）
      scale: 0.01
      mode: "both"       # "both" | "improve_only"
      schedule:
        type: "constant"  # "constant" | "linear_decay"
```

#### 成果物キー: reward_composition (CQ-0142)
各成分 (point_delta, shanten_delta, total) に以下の統計を保持:
- `count`, `sum`, `mean`, `std`, `nonzero_count`
- `p50`, `p90`, `p99` （quantile、sparse/tail 診断用）

出力先:
- `selfplay_stats.reward_composition`: worker 単位
- `summary.json.phase_stats.selfplay.reward_composition`: run 単位
- `batch_summary.json.runs[].reward_composition`: per-run
- `batch_summary.json.aggregate.reward_composition`: cross-run 集約（mean/std/p50/p90/p99 の run 間統計）

multi-worker 時: 各 worker の raw values を `reward_raw_values.npz` に保存し、runner が結合して正確な quantile を計算する。

#### 成果物キー: reward_shaping (CQ-0143)
shaping 設定を構造化して保存（config.yaml に依存せず比較可能）:
```json
{
  "shanten_delta": {
    "enabled": true,
    "scale": 0.01,
    "mode": "both",
    "schedule_type": "constant"
  }
}
```

出力先:
- `summary.json.phase_stats.selfplay.reward_shaping`
- `batch_summary.json.runs[].reward_shaping`

#### LearningSample.shanten_delta (CQ-0145, CQ-0148)
selfplay_worker は **raw シャンテン差分** (`prev_shanten - next_shanten`) を `LearningSample.shanten_delta` に格納する。
この値は reward shaping の `mode` / `scale` / `schedule` に一切依存しない。
- 正 = 改善、0 = 維持、負 = 悪化
- 初回打牌（prev 未定義）は NaN（unavailable 扱い、CQ-0149）

shard には条件付きカラムとして書き出す（`teacher_best_mask` と同パターン）。
shaping 無効時は `shanten_delta = None`（カラム省略）。

#### 成果物キー: shanten_diag (CQ-0145, CQ-0146, CQ-0148)
PPO learner が shanten_delta 付き shard を受け取った場合、advantage / return / value_error を
raw `shanten_delta` の符号で 3 群に分割した診断統計を `ppo_diag.shanten_diag` に格納する。
群分けは raw 差分の符号のみで行い、reward shaping の mode/scale/schedule には依存しない。

3 群:
- `improve`: delta > 0
- `same`: delta == 0（真に delta == 0 のサンプルのみ）
- `worsen`: delta < 0

各群:
- `count`
- `advantage`: mean, std, p50, p90, p99, positive_ratio, negative_ratio
- `return`: mean, std, p50, p90, p99
- `old_value`: mean, std, p50, p90, p99 (CQ-0155)
- `value_error`: mean, std, p50, p90, p99
- `new_value`: mean, std, p50, p90, p99 (CQ-0155, PPO 学習完了後の value 予測)
- `value_update_delta`: mean, std, p50, p90, p99 (CQ-0155, = new_value - old_value)
- `reward`: mean, std, p50, p90, p99 (CQ-0160, total reward = point_delta + shanten_delta)
- `point_delta_reward`: mean, std, p50, p90, p99 (CQ-0160, 点数差分報酬成分。shard に成分あり時のみ)
- `shanten_delta_reward`: mean, std, p50, p90, p99 (CQ-0160, shanten shaping 報酬成分。shard に成分あり時のみ)
- `delta_t`: mean, std, p50, p90, p99 (CQ-0160, 1-step TD 誤差 = reward_t + gamma * next_value_t - old_value_t)
- `post_riichi_discard_count` (CQ-0163, 群内の立直後打牌数。shard に is_post_riichi_discard あり時のみ)
- `post_riichi_discard_ratio` (CQ-0163, 群内の立直後打牌比率。同上)

`new_value` / `value_update_delta` は PPO 学習ループ完了後に全データに対して 1 回推論して取得する per-sample 値。

`delta_t` の定義は GAE 計算時の 1-step TD 誤差と一致する。terminated 境界では next_value = 0。
`point_delta_reward` / `shanten_delta_reward` は shard に reward 成分カラムがある場合のみ出力される。
`reward` / `delta_t` は rewards / terminateds / old_values から計算可能なため、reward 成分カラムがなくても出力される。

count == 0 の群は `{"count": 0}` のみ。

欠落データの扱い (CQ-0146):
- `shanten_delta` カラムが存在しない shard のサンプルは NaN として読み込まれる
- NaN サンプルは 3 群のいずれにも分類されない（unavailable として除外）
- `status` フィールドで状態を明示:
  - `complete`: 全サンプルに shanten_delta あり
  - `partial`: 一部のサンプルのみ shanten_delta あり（mixed shard）
  - `unavailable`: 全サンプルに shanten_delta なし（この場合 shanten_diag 自体は出力される）
- `total_samples`, `available_samples`, `unavailable_samples` を併記

立直後打牌統計 (CQ-0163):
- `total_post_riichi_discards`: 全サンプル中の立直後打牌数（shard に `is_post_riichi_discard` あり時のみ。なければ `null`）
- `available_post_riichi_discards`: available サンプル中の立直後打牌数（同上）

出力先:
- `metrics/train_metrics.json.ppo_diag.shanten_diag`
- `summary.json.phase_stats.learner.ppo_diag.shanten_diag`（既存 ppo_diag 転送パス経由）
- `batch_summary.json.runs[].learner_diag.shanten_diag`（同上）

#### 成果物キー: turn_diag (CQ-0156)
PPO learner が turn_number 付き shard を受け取った場合、巡目バケット別の診断統計を
`ppo_diag.turn_diag` に格納する。

バケット定義（固定）:
- `early`: turn 0-5
- `mid`: turn 6-11
- `late`: turn 12+

各バケット:
- `count`
- `advantage`: mean, std, p50, p90, p99, positive_ratio, negative_ratio
- `return`: mean, std, p50, p90, p99
- `old_value`: mean, std, p50, p90, p99
- `value_error`: mean, std, p50, p90, p99
- `new_value`: mean, std, p50, p90, p99 (PPO 学習完了後)
- `value_update_delta`: mean, std, p50, p90, p99 (= new_value - old_value)

count == 0 のバケットは `{"count": 0}` のみ。

データフロー:
- `turn_number` は `LearningSample` のオプショナルフィールド（shard 条件付きカラム）
- selfplay_worker が `env.env_state.round_state.turn_number` から取得
- learner が shard 読み込み時に `turn_numbers` として受け取り、`_train_ppo` に転送

出力先:
- `metrics/train_metrics.json.ppo_diag.turn_diag`
- `summary.json.phase_stats.learner.ppo_diag.turn_diag`（既存 ppo_diag 転送パス経由）
- `batch_summary.json.runs[].learner_diag.turn_diag`（同上）

#### 後方互換
- `reward.shaping` 未指定 or `shanten_delta.enabled: false` → shaping 無効、`total = point_delta` のまま
- 既存 config で動作が変わらないことをテストで保証
- 新規キーは追加のみ、既存キーの削除・改名なし
- shanten_delta カラムがない shard → `shanten_deltas = None`、shanten_diag 省略
- mixed shard（旧/新混在）→ 欠落サンプルは NaN、same 群に混入しない
- turn_number カラムがない shard → `turn_numbers = None`、turn_diag 省略
- point_delta_reward / shanten_delta_reward カラムがない shard → 成分別統計省略、reward/delta_t は出力
- reward 成分の mixed shard（旧/新混在）→ 欠落サンプルは NaN
- is_post_riichi_discard カラムがない shard → `is_post_riichi_discards = None`、post_riichi 統計省略
- is_post_riichi_discard の mixed shard（旧/新混在）→ sentinel -1 が混入するため `None` にフォールバック

#### LearningSample.is_post_riichi_discard (CQ-0163)
selfplay_worker は打牌前の `env.env_state.round_state.players[current].is_riichi` を取得し、
`LearningSample.is_post_riichi_discard` に bool として格納する。

定義: **そのサンプルの打牌時点で学習対象プレイヤーが立直済みである場合 `True`**。
立直宣言打牌そのものは `False`（宣言前の状態で判定）。

shard には条件付きカラムとして書き出す（int: 1/0、sentinel: -1）。

#### 立直後打牌の学習除外 (CQ-0164)
```yaml
training:
  exclude_post_riichi_discards:
    enabled: false  # デフォルト off
```

`enabled: true` のとき、learner は shard 読み込み後・学習前に `is_post_riichi_discard=True` のサンプルを除外する。
- PPO / imitation の両方に適用される
- shard 保存は変更しない（除外は learner 側で実行）
- 診断統計（shanten_diag の post_riichi_discard_count 等）は除外後のデータで計算される
- 除外件数は `train_metrics.post_riichi_exclusion` に記録:
  - `total_before_exclusion`, `excluded_post_riichi_discards`, `used_samples`
- 出力先 (CQ-0166):
  - `metrics/train_metrics.json.post_riichi_exclusion`
  - `summary.json.phase_stats.learner.post_riichi_exclusion`
  - `batch_summary.json.runs[*].post_riichi_exclusion`

### 9.5 将来案

将来的には以下を許容する。

- `final_rank`
- `combined`
- shaped / intermediate reward (shanten_delta 以外)
- reward normalization / clipping の明示仕様

ただし現仕様では主系ではない。

---

## 10. 現仕様: 学習データと shard

### 10.1 基本単位

学習データの基本単位は **step sample**。

1 サンプルは少なくとも次を持つ。

- observation / encoded feature に必要な情報
- legal mask
- chosen action
- reward
- log_prob
- value
- terminated flag
- round / episode / worker / seed などの追跡情報

### 10.2 shard 方針

現仕様:

- self-play データは file-based shard で保存
- 保存形式は Parquet
- learner は shard 群を読んで学習する

### 10.3 後方互換

shard は列追加に耐える方針を取る。  
既存 reader / learner を壊さない追加を優先する。

### 10.4 現在追加済みの補助列

現在の主な追加済み情報:

- `teacher_best_mask`
- self-play round end 系の補助出力（別ログ / stats）

これらは診断・模倣学習改善のための列であり、既存列の意味は変えない。

---

## 11. 現仕様: Self-Play

### 11.1 役割

Self-Play Worker は対局を生成し、学習サンプルと統計を保存する。

### 11.2 対戦相手構成

現運用では設定可能だが、主系では

- 学習中 policy
- ルールベース baseline

の混合を用いる。

### 11.3 Stage 1 baseline

現行 baseline の設計意図:

- シャンテン数最小
- 同点なら受け入れ最大
- テンパイ時自動立直

この baseline は imitation 教師と self-play 対戦相手の両方に関与するため、  
変更時は模倣学習と self-play 分布の双方に影響する。

### 11.4 self-play の可観測性

現仕様で run 成果物から確認可能:

- `policy_wins`
- `policy_deal_ins`
- `policy_draws`
- `tsumo_count`
- `ron_count`
- `ryukyoku_count`
- `num_rounds`
- `round_results.jsonl`

これらは PPO 改善診断の基礎材料として扱う。

---

## 12. 現仕様: Learner

### 12.1 役割

Learner は shard を読み込み、model を更新し、training metrics / checkpoint / summary を出力する。

### 12.2 現在の主系アルゴリズム

- imitation warm start
- PPO

### 12.3 warm start

現運用では、ルールベース打牌の軽い imitation を先に行い、その後 self-play + PPO に入る構成を標準とする。

### 12.4 現在の既知課題

現在の実験上の主要課題:

- imitation 改善は見える
- しかし PPO 後に平均悪化が起きやすい

このため learner は、単なる更新器ではなく **診断対象** でもある。

### 12.5 現仕様で成果物に残る learner 情報

- imitation metrics
- PPO loss 系
- `ppo_diag`

`ppo_diag` の主なカテゴリ:

- `advantage_*`
- `return_*`
- `old_value_*`
- `new_value_*`
- `value_error_*`
- `ratio_*`
- `clip_fraction`

出力先:

- `metrics/train_metrics.json`
- `summary.json.phase_stats.learner.ppo_diag`
- `batch_summary.json.runs[].learner_diag`
- `batch_summary.json.aggregate.learner_diag`

### 12.6 現仕様での解釈上の注意

- reuse 実験では self-play が共通なので `value_error` など一部統計は条件間で同じになりうる
- learner 診断統計は「改善そのもの」ではなく「悪化理由の候補切り分け」に使う

---

## 13. 現仕様: imitation 診断

### 13.1 教師再現指標

imitation フェーズ完了後、少なくとも次を出力する。

- `teacher_top1_match_rate`
- `teacher_best_set_hit_rate`

### 13.2 best-set の定義

教師最良候補集合は、RuleBasedBaseline の評価規則

1. シャンテン最小  
2. 受け入れ最大  

で同率最良となる全合法手の集合。

### 13.3 imitation loss mode

現仕様で利用可能な mode:

| 値 | 意味 |
|---|---|
| `strict_top1` | 教師 action への cross-entropy |
| `tie_aware_best_set` | `-log(sum_{a in best_set} pi(a))` |

### 13.4 現仕様上の原則

- strict 経路は既定として維持する
- tie-aware は追加モードとして扱う
- shard に必要な補助情報がなければ fail-fast する

### 13.5 joint imitation (CQ-0150)

imitation フェーズで policy loss に加えて value loss を同時最適化する機能。

config:
```yaml
training:
  imitation_value_warmstart:
    enabled: false  # デフォルト off（後方互換）
    coef: 0.5       # value loss の重み
```

loss 式:
```
total_loss = policy_loss + coef * value_loss - entropy_coef * entropy
```

- value target: `_compute_gae(rewards, old_values, terminateds)` の returns（PPO と同一定義）
- enabled=false のとき従来通り policy_loss - entropy のみ
- strict_top1 / tie_aware_best_set とは独立に併用可能

出力先:
- `summary.json.phase_stats.imitation.value_loss`
- `summary.json.phase_stats.imitation.imitation_value_warmstart`
- `batch_summary.json.runs[].imitation_metrics.value_loss`
- `batch_summary.json.runs[].imitation_metrics.imitation_value_warmstart`
- `batch_summary.json.aggregate.imitation.value_loss`

### 13.6 成果物追跡 (CQ-0152)

joint imitation / value 補助特徴の設定と指標を追跡するため、以下を成果物に記録する。

summary.json:
- `phase_stats.imitation.value_loss`
- `phase_stats.imitation.imitation_value_warmstart.{enabled, coef}`
- `model_features.value_features.current_shanten.enabled`

batch_summary.json:
- `runs[].imitation_metrics.value_loss`
- `runs[].imitation_metrics.imitation_value_warmstart`
- `runs[].model_features`
- `aggregate.imitation.value_loss` (mean/std/count/min/max)

---

## 14. 現仕様: 評価

### 14.1 主指標

評価の主指標は次。

- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

### 14.2 評価モード

現仕様で特に重要なのは `rotation`。

rotation では:

- aggregate 結果
- seat 別結果

を保存する。

### 14.3 `eval_before -> eval`

現在の PPO 比較では、after 指標よりも

- `eval_before`
- `eval`
- `eval_diff`

の前後差分が主評価になることが多い。

したがって、reuse を含む運用でも `eval_before` の復元と `eval_diff` の生成は後方互換を壊さず維持する必要がある。

---

## 15. 現仕様: 実験設定

### 15.1 正本

人間可読な正本は YAML。

主要カテゴリ:

- `experiment`
- `feature_encoder`
- `model`
- `reward`
- `selfplay`
- `training`
- `evaluation`

### 15.2 運用上の原則

- 個別実験の具体条件は runbook に書く
- YAML と CLI override の組み合わせで条件を固定する
- 実験比較で重要な差分は `summary` / `notes` / `batch_summary` から追跡可能であることを重視する

---

## 16. 現仕様: 実験成果物

### 16.1 run directory

run ごとに少なくとも次を持つ。

- `config.yaml`
- `summary.json`
- `notes.md`
- `metrics/`
- `selfplay/`
- `eval/`
- `checkpoints/`

### 16.2 追跡すべき成果物

現在の主な分析対象:

- `summary.json`
- `metrics/train_metrics.json`
- `eval/eval_rotation.json`
- `eval/eval_diff.json`
- `batch_summary.json`

### 16.3 phase 再利用

現仕様では phase 成果物再利用をサポートする。

主な再利用パターン:

- imitation checkpoint 再利用
- `imitation,selfplay,eval_before` 再利用
- phase 単位 resume

設計原則:

- 通常 run を壊さない
- 再利用時は整合性チェックを行う
- summary / manifest から再利用元を追える

---

## 17. 現仕様: batch 集約

batch 実行時は、run ごとの主要成果物を集約する。

少なくとも:

- per-run metrics
- aggregate mean/std

を扱えること。

現時点で特に重要なのは:

- imitation metrics
- learner diagnostic metrics
- eval aggregate metrics

batch 集約は「詳細全部を持つ」よりも、
**runbook / report で横並び比較できる最小十分なキーを固定する**
ことを優先する。

---

## 18. 現在の既知課題

2026-03-08 時点の主要な既知課題は次の通り。

1. PPO が imitation 初期方策を平均的に悪化させる  
2. reward が sparse で tail が強い  
3. learner が見ている target / advantage / value の質が十分良いか未確定  
4. 更新強度の調整だけでは問題が解けない可能性が高い  

この節は `PROJECT.md` の判断と整合すること。  
詳細な優先順位は `PROJECT.md` に置き、本書では「基盤として意識すべき技術課題」に留める。

---

## 19. テスト要件

### 19.1 最低限必要な層

- unit
- integration
- runner / batch 集約
- reproducibility

### 19.2 診断機能追加時の原則

診断メトリクスや補助列を追加するときは、少なくとも次を満たす。

- 既存 run / shard / summary / batch を壊さない
- 追加のみで入れる
- NaN / 欠落 / 型不整合を検知する
- run 単位と batch 単位の最低限の読取り経路をテストする

### 19.3 実行レーン

テストは、

- 変更に直接関係する targeted test
- 必要時のみ smoke / broader integration

を原則とする。  
実験速度を不必要に損なう一律実行は避ける。

---

## 20. CHANGE_QUEUE 運用との関係

RL 基盤の仕様変更は `CHANGE_QUEUE.md` で管理する。

推奨 Type:

- `RL`
- `Training`
- `Experiment`
- `Eval`

レビュー運用上の原則:

- レビュー NG 時は旧 CQ を削除し、新規 CQ を起票する
- 仕様変更と実験解釈を混ぜない
- 本書に関わる変更は、必要なら本書も同期更新する

---

## 21. 将来案

以下は保持したい将来拡張の方向であり、現仕様ではない。

- PartialObservation 主系化
- Full → Partial 蒸留
- CNN / token / transformer 系 encoder
- 複数 policy の本格分離
- `final_rank` や `combined` の本格運用
- shaped reward / curriculum
- search policy / MCTS 接続
- async actor-learner / distributed learner
- ONNX Runtime を使った高速推論運用

将来案を実装対象に昇格させるときは、まず CQ と `PROJECT.md` で優先順位を確定し、その後本書の現仕様へ移す。

---

## 22. 一文要約

**RL_SPEC.md は、現在の Stage 1 学習基盤を壊さず拡張するための長期設計基準であり、日常の実験判断は PROJECT.md、個別条件は runbook/report を正とする。**
