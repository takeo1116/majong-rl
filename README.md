# majong-rl
麻雀の強化学習をゲームエンジンから作る

## ビルド方法

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## テスト実行

```bash
cd build
ctest --output-on-failure
```

## サンプル実行

```bash
cd build
./mahjong_example
```

## ディレクトリ構成

```
docs/       - CLAUDE.md, RULE.md, SPEC.md, CHANGE_QUEUE.md
src/core/   - 牌・副露・状態・行動・イベントの基礎構造
src/rules/  - 和了判定、役判定、符計算、点数計算
src/engine/ - reset/step、合法手列挙、応答解決
src/rl/     - Observation、Reward、環境ラッパー
src/io/     - ログ、文字列表現、CLI
tests/unit/ - 単体テスト
tests/integration/ - 統合テスト
tests/replay/      - 再現テスト
examples/   - サンプル実行
```
