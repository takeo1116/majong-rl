"""Stage 1 評価対戦ランナー: 主要指標の集計"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mahjong_rl.env import Stage1Env
from mahjong_rl.action_selector import ActionSelector, SelectionMode
from mahjong_rl.baseline import RuleBasedBaseline
from mahjong_rl import RoundEndReason


@dataclass
class PartialEvalMetrics:
    """worker 部分結果 (加算可能統計)

    各 worker が独立に計算した部分結果を保持する。
    aggregate_partials() で最終 EvalMetrics に集約できる。
    """
    sum_rank: float
    sum_score: float
    wins: int
    deal_ins: int
    num_rounds: int
    num_matches: int
    policy_seats: list[int] | None = None
    worker_id: int | None = None
    metadata: dict | None = None

    def to_dict(self) -> dict:
        """JSON 互換 dict に変換する"""
        d: dict = {
            "sum_rank": self.sum_rank,
            "sum_score": self.sum_score,
            "wins": self.wins,
            "deal_ins": self.deal_ins,
            "num_rounds": self.num_rounds,
            "num_matches": self.num_matches,
        }
        if self.policy_seats is not None:
            d["policy_seats"] = self.policy_seats
        if self.worker_id is not None:
            d["worker_id"] = self.worker_id
        if self.metadata is not None:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PartialEvalMetrics:
        """dict から復元する"""
        return cls(
            sum_rank=d["sum_rank"],
            sum_score=d["sum_score"],
            wins=d["wins"],
            deal_ins=d["deal_ins"],
            num_rounds=d["num_rounds"],
            num_matches=d["num_matches"],
            policy_seats=d.get("policy_seats"),
            worker_id=d.get("worker_id"),
            metadata=d.get("metadata"),
        )

    def save(self, path: Path) -> None:
        """JSON に保存する"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> PartialEvalMetrics:
        """JSON から読み込む"""
        with open(path) as f:
            return cls.from_dict(json.load(f))


def aggregate_partials(partials: list[PartialEvalMetrics]) -> EvalMetrics:
    """部分結果を集約して最終 EvalMetrics を生成する

    Args:
        partials: worker ごとの部分結果リスト

    Returns:
        集約された EvalMetrics
    """
    if not partials:
        raise ValueError("partials が空です")

    total_sum_rank = sum(p.sum_rank for p in partials)
    total_sum_score = sum(p.sum_score for p in partials)
    total_wins = sum(p.wins for p in partials)
    total_deal_ins = sum(p.deal_ins for p in partials)
    total_rounds = sum(p.num_rounds for p in partials)
    total_matches = sum(p.num_matches for p in partials)

    # policy_seats: 全 partial で共通であればそれを使う
    all_seats = [p.policy_seats for p in partials if p.policy_seats is not None]
    policy_seats = all_seats[0] if all_seats else None

    return EvalMetrics(
        avg_rank=total_sum_rank / max(total_matches, 1),
        avg_score=total_sum_score / max(total_matches, 1),
        win_rate=total_wins / max(total_rounds, 1),
        deal_in_rate=total_deal_ins / max(total_rounds, 1),
        num_matches=total_matches,
        num_rounds=total_rounds,
        policy_seats=policy_seats,
    )


@dataclass
class EvalMetrics:
    """評価指標"""
    avg_rank: float
    avg_score: float
    win_rate: float
    deal_in_rate: float
    num_matches: int
    num_rounds: int
    policy_seats: list[int] | None = None

    def save(self, path: Path) -> None:
        """JSON に保存"""
        data = {
            "avg_rank": self.avg_rank,
            "avg_score": self.avg_score,
            "win_rate": self.win_rate,
            "deal_in_rate": self.deal_in_rate,
            "num_matches": self.num_matches,
            "num_rounds": self.num_rounds,
        }
        if self.policy_seats is not None:
            data["policy_seats"] = self.policy_seats
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


@dataclass
class RotationEvalResult:
    """席ローテーション評価の集計結果"""
    per_seat: dict[int, EvalMetrics]
    aggregate: EvalMetrics

    def save(self, eval_dir: Path) -> None:
        """席別・総合の結果を eval ディレクトリに保存"""
        eval_dir.mkdir(parents=True, exist_ok=True)
        # 席別
        for seat, metrics in self.per_seat.items():
            metrics.save(eval_dir / f"eval_seat{seat}.json")
        # 総合
        self.aggregate.save(eval_dir / "eval_rotation.json")


def save_partial(partial: PartialEvalMetrics, partials_dir: Path,
                  worker_id: int) -> Path:
    """worker 部分結果を保存する

    Args:
        partial: 部分結果
        partials_dir: 保存先ディレクトリ (例: eval/partials)
        worker_id: worker 識別子

    Returns:
        保存先パス
    """
    path = partials_dir / f"worker_{worker_id}.json"
    partial.save(path)
    return path


def load_partials(partials_dir: Path) -> list[PartialEvalMetrics]:
    """ディレクトリから全 worker 部分結果を読み込む

    Args:
        partials_dir: partials が保存されたディレクトリ

    Returns:
        PartialEvalMetrics のリスト (worker_id 順でソート)
    """
    paths = sorted(partials_dir.glob("worker_*.json"))
    return [PartialEvalMetrics.load(p) for p in paths]


def aggregate_and_save(partials_dir: Path, eval_dir: Path,
                       filename: str = "eval_metrics.json") -> EvalMetrics:
    """partials を集約して最終結果を保存する

    Args:
        partials_dir: partials が保存されたディレクトリ
        eval_dir: 最終結果の保存先
        filename: 保存ファイル名

    Returns:
        集約された EvalMetrics
    """
    partials = load_partials(partials_dir)
    metrics = aggregate_partials(partials)
    metrics.save(eval_dir / filename)
    return metrics


def aggregate_rotation_partials(
    partials_dir: Path,
    eval_dir: Path,
    seats: list[int],
) -> RotationEvalResult:
    """rotation eval の worker 部分結果を席別に集約する

    partials_dir 内のファイル命名規則:
    - 単席 partial: worker_{worker_id}.json (policy_seats=[seat])
    - 各 worker の partial.policy_seats から席を特定して振り分ける

    Args:
        partials_dir: partials ディレクトリ
        eval_dir: 最終結果の保存先
        seats: 評価対象席リスト

    Returns:
        RotationEvalResult (席別 + 総合)
    """
    all_partials = load_partials(partials_dir)

    # policy_seats で席別に振り分け
    seat_partials: dict[int, list[PartialEvalMetrics]] = {s: [] for s in seats}
    for p in all_partials:
        if p.policy_seats and len(p.policy_seats) == 1:
            seat = p.policy_seats[0]
            if seat in seat_partials:
                seat_partials[seat].append(p)

    per_seat: dict[int, EvalMetrics] = {}
    all_partials_flat: list[PartialEvalMetrics] = []
    for seat in seats:
        partials = seat_partials[seat]
        if partials:
            per_seat[seat] = aggregate_partials(partials)
            all_partials_flat.extend(partials)

    # 総合集計
    aggregate = aggregate_partials(all_partials_flat) if all_partials_flat else EvalMetrics(
        avg_rank=0.0, avg_score=0.0, win_rate=0.0, deal_in_rate=0.0,
        num_matches=0, num_rounds=0, policy_seats=seats,
    )
    aggregate.policy_seats = seats

    result = RotationEvalResult(per_seat=per_seat, aggregate=aggregate)
    result.save(eval_dir)
    return result


def compute_eval_diff(before: dict, after: dict) -> dict:
    """学習前後の評価差分を計算する

    Args:
        before: 学習前の eval_metrics dict (avg_rank, avg_score, win_rate, deal_in_rate)
        after:  学習後の eval_metrics dict

    Returns:
        diff dict: 各指標の before, after, delta を含む
    """
    keys = ["avg_rank", "avg_score", "win_rate", "deal_in_rate"]
    diff: dict = {
        "eval_mode_before": before.get("eval_mode", "single"),
        "eval_mode_after": after.get("eval_mode", "single"),
    }
    for key in keys:
        b = before.get(key)
        a = after.get(key)
        diff[key] = {
            "before": b,
            "after": a,
            "delta": round(a - b, 6) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None,
        }
    return diff


class EvaluationRunner:
    """評価対戦ランナー

    指定席=学習ポリシー（argmax）、残り=ベースライン。
    """

    def __init__(self, model: torch.nn.Module, encoder, observation_mode: str = "full",
                 inference_device: torch.device | None = None):
        self._device = inference_device or torch.device("cpu")
        self._model = model.to(self._device)
        self._encoder = encoder
        self._observation_mode = observation_mode
        self._baseline = RuleBasedBaseline()
        self._selector = ActionSelector(mode=SelectionMode.ARGMAX)

    def evaluate_partial(
        self,
        num_matches: int = 100,
        seed_start: int = 0,
        policy_seats: list[int] | None = None,
        worker_id: int | None = None,
        match_seeds: list[int] | None = None,
    ) -> PartialEvalMetrics:
        """評価対戦を実行して加算可能な部分結果を返す

        Args:
            policy_seats: ポリシー席のリスト。None なら [0]。
            worker_id: worker 識別子（partial 保存時に使用）。
            match_seeds: match ごとの seed リスト。指定時は seed_start の代わりに使用。
                         len(match_seeds) == num_matches であること。
        """
        if policy_seats is None:
            policy_seats = [0]

        sum_rank = 0.0
        sum_score = 0.0
        total_wins = 0
        total_deal_ins = 0
        total_rounds = 0

        for seat in policy_seats:
            for i in range(num_matches):
                seed = match_seeds[i] if match_seeds else seed_start + i
                result = self._play_one_match(seed, policy_seat=seat)
                sum_rank += result["rank"]
                sum_score += result["score"]
                total_wins += result["wins"]
                total_deal_ins += result["deal_ins"]
                total_rounds += result["rounds"]

        total_matches = num_matches * len(policy_seats)
        return PartialEvalMetrics(
            sum_rank=sum_rank,
            sum_score=sum_score,
            wins=total_wins,
            deal_ins=total_deal_ins,
            num_rounds=total_rounds,
            num_matches=total_matches,
            policy_seats=policy_seats,
            worker_id=worker_id,
        )

    def evaluate(
        self,
        num_matches: int = 100,
        seed_start: int = 0,
        eval_dir: Path | None = None,
        policy_seats: list[int] | None = None,
    ) -> EvalMetrics:
        """評価対戦を実行して指標を返す

        Args:
            policy_seats: ポリシー席のリスト。None なら [0]。
                          複数席を指定すると各席で全半荘を実行し平均する。
        """
        partial = self.evaluate_partial(
            num_matches=num_matches,
            seed_start=seed_start,
            policy_seats=policy_seats,
        )
        metrics = aggregate_partials([partial])

        if eval_dir:
            metrics.save(eval_dir / "eval_metrics.json")

        return metrics

    def evaluate_rotation(
        self,
        num_matches: int = 100,
        seed_start: int = 0,
        eval_dir: Path | None = None,
        seats: list[int] | None = None,
    ) -> RotationEvalResult:
        """全席ローテーション評価を実行し、席別・総合の指標を返す

        Args:
            seats: 評価する席のリスト。None なら [0,1,2,3]。
        """
        if seats is None:
            seats = [0, 1, 2, 3]

        per_seat_partials: dict[int, PartialEvalMetrics] = {}
        per_seat: dict[int, EvalMetrics] = {}
        for seat in seats:
            partial = self.evaluate_partial(
                num_matches=num_matches,
                seed_start=seed_start,
                policy_seats=[seat],
            )
            per_seat_partials[seat] = partial
            per_seat[seat] = aggregate_partials([partial])

        # 総合集計: 全席の partial を統合
        all_partials = list(per_seat_partials.values())
        aggregate = aggregate_partials(all_partials)
        aggregate.policy_seats = seats

        result = RotationEvalResult(per_seat=per_seat, aggregate=aggregate)
        if eval_dir:
            result.save(eval_dir)

        return result

    def _play_one_match(self, seed: int, policy_seat: int = 0) -> dict:
        """1 半荘を実行して結果を返す"""
        env = Stage1Env(observation_mode=self._observation_mode)
        obs, info = env.reset(seed=seed)

        policy_player = policy_seat
        wins = 0
        deal_ins = 0
        round_count = 0
        prev_round_number = info["round_number"]
        prev_scores = list(info["scores"])

        max_steps = 10000
        for _ in range(max_steps):
            current = env.current_player
            mask = env.get_legal_mask()

            if current == policy_player:
                tile_type = self._policy_step(obs, mask)
            else:
                tile_type = self._baseline_step(env, mask)

            obs, rewards, terminated, truncated, info = env.step(tile_type)

            # 局終了判定
            round_number = info["round_number"]
            round_over = (round_number != prev_round_number) or terminated
            if round_over:
                round_count += 1
                scores = info["scores"]

                # 和了判定: ポリシープレイヤーのスコアが増加
                score_diff = scores[policy_player] - prev_scores[policy_player]
                if score_diff > 0:
                    wins += 1

                # 放銃判定: ポリシープレイヤーのスコアが減少（他家和了による）
                if score_diff < 0:
                    # 他家のスコアが大きく増加していれば放銃の可能性
                    for p in range(4):
                        if p != policy_player and (scores[p] - prev_scores[p]) > 0:
                            deal_ins += 1
                            break

                prev_scores = list(scores)
                prev_round_number = round_number

            if terminated:
                break

        # 最終順位計算
        final_scores = info["scores"]
        rank = self._compute_rank(final_scores, policy_player)

        return {
            "rank": rank,
            "score": final_scores[policy_player] - 25000,
            "wins": wins,
            "deal_ins": deal_ins,
            "rounds": round_count,
        }

    def _policy_step(self, obs, mask: np.ndarray) -> int:
        """ポリシーモデルで打牌を選択する（argmax）"""
        features = self._encoder.encode(obs)
        features_flat = features.flatten() if features.ndim > 1 else features
        features_t = torch.from_numpy(features_flat).unsqueeze(0).to(self._device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self._device)

        with torch.no_grad():
            output = self._model(features_t, mask_t)

        tile_type, _ = self._selector.select(output.logits[0], mask_t[0])
        return tile_type

    def _baseline_step(self, env: Stage1Env, mask: np.ndarray) -> int:
        """ベースラインで打牌を選択する"""
        hand = env.env_state.round_state.players[env.current_player].hand
        return self._baseline.select_discard(list(hand), mask)

    @staticmethod
    def _compute_rank(scores: list, player: int) -> int:
        """スコアから順位を計算する (1-indexed)"""
        player_score = scores[player]
        rank = 1
        for i, s in enumerate(scores):
            if i != player and s > player_score:
                rank += 1
        return rank
