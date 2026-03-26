# Bug Report: Cross-Player / Cross-Round Bootstrap in GAE and Diagnostics (exp_065)

作成日: 2026-03-22  
起点実験: `experiments/exp_065`  
状態: **修正後初回検証済み**

## 1. 概要

現行 learner は、PPO / imitation value warmstart の return 計算で、

- `reward_t`: **その時に打牌したプレイヤー本人**の reward
- `next_value`: **単純に次サンプル `t+1` の value**

を組み合わせている。

しかし実際の shard 順序では、`t+1` はほぼ常に**次のプレイヤーの打牌サンプル**であり、さらに `terminated=False` のまま**局を跨ぐ**ケースもある。

その結果、現行の GAE / return / advantage / value_error / `delta_t` は、かなり高い確率で

- 別プレイヤーの value
- 次局の value

を bootstrap に用いており、常識的な強化学習の前提である

- 「同一主体の次状態 value」
- 少なくとも「bootstrap を切るべき境界で打ち切る」

から逸脱している。

## 2. 発見経緯

`exp_065` までの PPO 分析で、`shanten_diag` において

- `same` の advantage が正
- `improve` の advantage が強く負
- `improve < worsen`

という不自然な傾向が継続していた。

これを reward 設計だけの問題とみなす前に、GAE / TD target の定義をソースから再確認したところ、

- `_compute_gae()` が `v[t+1]` をそのまま `next_value` に使用
- shard reader が sample を `player_id` / `episode_id` / `round_id` ごとに組み直していない

ことが分かった。

その後、`exp_065` の実際の self-play shard を直接読み、`t -> t+1` の `player_id` 遷移を集計したところ、問題が実データでも高頻度に発生していることを確認した。

## 3. 確認した事実

### 3.1 reward 自体は「今打った本人」のもの

self-play worker は sample 保存時に

- `point_delta_reward = rewards[current]`
- `reward = point_delta_reward + shanten_delta_reward`

を保存している。

したがって、問題は reward 定義そのものではなく、**bootstrap に使う `next_value` の主体がずれている**点にある。

関係箇所:
- `python/mahjong_rl/selfplay_worker.py`

### 3.2 GAE は単純に `t+1` の value を使っている

現行の `_compute_gae()` は

- `t == n - 1` または `terminated[t]` のときだけ `next_value = 0`
- それ以外は `next_value = values[t + 1]`

としている。

関係箇所:
- `python/mahjong_rl/learner.py`

### 3.3 shard reader は系列を組み直していない

`ShardReader.read_as_tensors()` は shard を path 順に連結し、`filter_actor_type` を使う場合も

- 順序はそのまま
- mask で該当サンプルだけ抜き出す

だけである。

`episode_id` / `round_id` / `player_id` を使った系列再構成はしていない。

関係箇所:
- `python/mahjong_rl/shard.py`

### 3.4 `terminated` は match 終了でしか立たない

env 側では `terminated=True` になるのは `match_over` のときだけであり、`round_over` では立たない。

そのため現行 GAE は、**局終了をまたいでも bootstrap を継続**する。

関係箇所:
- `python/mahjong_rl/env/stage1_env.py`

## 4. 実データでの確認結果

`exp_065` の `rule-only PPO` run を spot check した。

対象:
- PPO cycle shard (`cycle_00`, `cycle_08`, `cycle_29`)
- いずれも `actor_type='baseline'` のみ

結果:
- `terminated=False` の遷移のうち、`player_id[t+1] != player_id[t]` の割合は **99.5756%**
- 同じプレイヤーに留まる割合は **0.4244%**
- さらに `terminated=False` のまま `round_id` が変わる遷移が **1263 件 / cycle** 存在

つまり現行 learner は、実質的にほぼ常に

- 別プレイヤーの value
- ときに次局の value

で bootstrap している。

また、top-level imitation 用 shard でも同様に spot check したところ、

- `terminated=False` 遷移での `player_id` 切替率は **99.5398%**

であり、**imitation value warmstart 側にも同じ問題がある**ことを確認した。

## 5. 影響箇所

### 5.1 直接 learning target を壊す箇所

1. PPO の GAE / returns
   - `python/mahjong_rl/learner.py::_compute_gae`

2. imitation value warmstart の returns
   - `python/mahjong_rl/learner.py::_train_imitation`
   - `imitation_value_warmstart.enabled=true` のため現行 baseline でも有効

### 5.2 診断値を壊す箇所

以下は学習 target そのものではないが、解釈を大きく歪める。

1. `shanten_diag.delta_t`
   - `reward_t + gamma * old_value[t+1] - old_value[t]`
   - 現在の `improve/same/worsen` 議論に直結

2. `ppo_diag.return`
3. `ppo_diag.value_error`
4. `shanten_diag.advantage`
5. `shanten_diag.return`
6. `shanten_diag.value_error`
7. `turn_diag.advantage`
8. `turn_diag.return`
9. `turn_diag.value_error`

これらはすべて、汚染済みの GAE / return 系テンソルに依存している。

## 6. 追加の注意点

### 6.1 `filter_actor_type` は問題を悪化させうる

`filter_actor_type='policy'` / `'baseline'` は単に mask を当てるだけなので、mixed 条件では

- `t` と `t+1` の距離がさらに開く
- それでも `next_value = values[t+1]` が使われる

という形で、主体不整合がさらに悪化しうる。

`rule-only` baseline では cycle shard が全部 `baseline` なのでこの追加悪化は起きていないが、mixed PPO や separated learner 条件では注意が必要である。

### 6.2 `round_over` は保存されているが learner は未使用

sample には `round_over` が保存されているが、現行 GAE はこれを打ち切り条件に使っていない。

したがって、少なくとも

- `terminated`
- `round_over`
- `player_id` 変化

のどこで bootstrap を切るべきかを明示設計する必要がある。

## 7. いま何が言えるか

この問題は「起こりうる」ではなく、現行 mainline で**実際に高頻度で起きている**。

特に強く言えるのは次の 3 点である。

1. 現行 GAE は、ほぼ常に**別プレイヤーの value**で bootstrap している
2. さらに `round_over` をまたいで**次局の value**でも bootstrap している
3. `improve / worsen` の advantage 逆転や PPO の不自然な学習ダイナミクスに対する、**かなり有力な主因候補**である

## 8. 未確定点

まだ切れていない点もある。

- これが単独主因か、reward shaping 側の問題と複合か
- 修正後に `improve < worsen` がどこまで解消するか
- どの単位で bootstrap を切るのが最も妥当か
  - `player_id`
  - `round_over`
  - `(episode_id, player_id)` 系列
  - その組み合わせ

したがって、現時点では **「根本原因候補として非常に強い」** という位置づけであり、修正方針は別途設計が必要である。

## 9. 修正後の初回検証で確認できたこと

CQ-0210 / CQ-0211 実装後、現時点の baseline 条件

- 新モデル
- rule-only PPO
- `policy_anchor.coef=0.5`
- `clip_epsilon=0.15`
- `gamma=0.75`
- `gae_lambda=0.3`
- `value_loss_coef=0.25`

で 1 seed (`seed=42`) を再実行した。

### 9.1 `shanten_diag` の向きはかなり自然化した

修正前 final では、おおむね

- `improve advantage < 0`
- `same advantage > 0`
- `improve < worsen`

が継続していた。

修正後 final では、

- `improve advantage = +0.105`
- `same advantage = -0.0318`
- `worsen advantage = -0.0087`

となり、少なくとも「`improve` が強く負で、`same` だけが正」という以前の不自然さは大きく後退した。  
`reward` と `delta_t` も同方向に変化しており、**壊れた bootstrap semantics が `improve / worsen` 解釈を大きく歪めていた**可能性が強い。

### 9.2 ただし性能はこの 1 seed では悪化した

同条件の修正前 `g075_gae030` seed42 と比べると、

- 修正前
  - imitation直後: `avg_score = 1986.0`
  - final: `2385.5`
  - best: `2686.75`
- 修正後
  - imitation直後: `avg_score = 2175.75`
  - final: `1249.5`
  - best: `2547.25`

となり、**修正後は PPO が一時的に伸びても final で imitation 直後を下回った**。

このことから、次の可能性が高い。

1. CQ-0210 / CQ-0211 によって学習信号の意味が本当に変わった
2. 以前の「最良ハイパラ」は、壊れた return semantics に対して最適化されていた
3. 修正後は、`gamma / gae / anchor / clip` を再チューニングする必要がある

### 9.3 いま言えること

- 修正は性能を即座に改善したわけではない
- しかし **advantage の意味はかなり健全化した**
- したがって、これは「修正が間違い」ではなく  
  **「正しい土台に戻した結果、旧ハイパラが外れた」**  
  と解釈するのが自然である

## 10. 次のアクション候補

1. 修正後 baseline を 3 seeds で再確認する
   - `gamma=0.75`
   - `gae_lambda=0.3`
   - `policy_anchor.coef=0.5`
   - `clip_epsilon=0.15`
   - `value_loss_coef=0.25`

2. 修正後 semantics 前提で PPO ハイパラを再探索する
   - `gamma / gae`
   - 必要なら `anchor / clip`
   - 以前の最良条件を前提にしない

3. 修正後に再確認する
   - `shanten_diag.improve/same/worsen`
   - PPO の peak 保持
   - imitation 直後 vs final の比較
   - `teacher_best_set_hit_rate`
   - imitation value warmstart の影響

## 11. 暫定評価

現時点では、

- `gamma`
- `gae_lambda`
- `value_loss_coef`
- `clip_epsilon`

よりも先に、**GAE / return の主体整合性そのもの**を正す必要がある可能性が高い。

少なくとも、この問題を未修正のまま advantage の意味を深く解釈するのは危険である。

一方で、修正後は

- `improve / same / worsen` の符号関係
- PPO の final 性能

の両方が従来と変わるため、**修正後 baseline での PPO 再探索は必須**である。
