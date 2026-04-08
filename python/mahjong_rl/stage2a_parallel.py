"""CQ-0233/0234/0235: Stage2a multi-process orchestration

imitation / selfplay / eval の並列実行を担う。
各 worker は subprocess で Stage2SelfPlayWorker / Stage2aEvaluator を使う。
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch


def derive_worker_seed(base_seed: int, worker_id: int) -> int:
    return (base_seed * 31 + worker_id * 7919) % (2**31)


def derive_match_seed(worker_seed: int, match_index: int) -> int:
    return (worker_seed * 37 + match_index * 1013) % (2**31)


def distribute_matches(total: int, num_workers: int) -> list[int]:
    base = total // num_workers
    remainder = total % num_workers
    return [base + (1 if i < remainder else 0) for i in range(num_workers)]


# ============ Selfplay Worker ============

def _stage2a_selfplay_worker_fn(
    worker_id: int,
    output_dir: str,
    obs_mode: str,
    encoder_config: dict,
    model_state_path: str | None,
    model_config: dict,
    num_matches: int,
    base_seed: int,
    experiment_id: str,
    run_id: str,
    result_queue: mp.Queue,
    error_queue: mp.Queue,
    inference_device: str = "cpu",
    policy_ratio: float = 1.0,
    save_baseline_actions: bool = False,
    num_threads: int = 1,
):
    """subprocess で Stage2a selfplay を実行"""
    try:
        from mahjong_rl.runner import configure_worker_threads, _parse_encoder_flag
        configure_worker_threads(num_threads)

        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder

        device = torch.device(inference_device)

        # encoder 再構築
        enc = FlatFeatureEncoder(
            observation_mode=obs_mode,
            shanten_hint=_parse_encoder_flag(encoder_config, "shanten_hint"),
            discard_ukeire_hint=_parse_encoder_flag(encoder_config, "discard_ukeire_hint"),
            current_shanten_input=_parse_encoder_flag(encoder_config, "current_shanten"),
            shape_hint=_parse_encoder_flag(encoder_config, "shape_hint"),
            turn_context=_parse_encoder_flag(encoder_config, "turn_context"),
            opponent_current_shanten=_parse_encoder_flag(encoder_config, "opponent_current_shanten"),
            opponent_tenpai_flag=_parse_encoder_flag(encoder_config, "opponent_tenpai_flag"),
            danger_mask=_parse_encoder_flag(encoder_config, "danger_mask"),
        )

        # model 再構築 (selfplay 用)
        model = None
        if model_state_path:
            from mahjong_rl.models.stage2a_model import Stage2aModel
            meta = enc.metadata()
            input_dim = int(np.prod(meta.output_shape))
            hint_ranges = {s: meta.feature_ranges[s]
                           for s in ("shanten_hint", "discard_ukeire_hint")
                           if s in meta.feature_ranges}
            model = Stage2aModel(
                input_dim=input_dim,
                discard_hidden_dims=model_config.get("discard_hidden_dims", [256, 128]),
                optional_hidden_dims=model_config.get("optional_hidden_dims",
                    model_config.get("call_hidden_dims", [128, 64])),
                value_hidden_dims=model_config.get("value_hidden_dims", [128, 64]),
                candidate_dim=model_config.get("candidate_dim", 16),
                optional_scorer_hidden=model_config.get("optional_scorer_hidden", 32),
                semantic_aux_config=model_config.get("semantic_aux"),
                direct_hint_ranges=hint_ranges if hint_ranges else None,
            )
            sd = torch.load(model_state_path, map_location="cpu", weights_only=True)
            model.load_state_dict(sd)

        worker = Stage2SelfPlayWorker(
            config={},
            output_dir=output_dir,
            observation_mode=obs_mode,
            encoder=enc,
            model=model,
            device=device,
            policy_ratio=policy_ratio,
            save_baseline_actions=save_baseline_actions,
        )
        stats = worker.generate(
            num_matches=num_matches,
            base_seed=base_seed,
            experiment_id=experiment_id,
            run_id=run_id,
            worker_id=f"stage2_w{worker_id}",
        )
        result_queue.put((worker_id, stats))
    except Exception as e:
        error_queue.put((worker_id, str(e)))


def run_stage2a_selfplay_parallel(
    output_dir: Path,
    num_workers: int,
    num_matches: int,
    base_seed: int,
    obs_mode: str,
    encoder_config: dict,
    model_state_path: str | None,
    model_config: dict,
    experiment_id: str = "",
    run_id: str = "",
    inference_device: str = "cpu",
    policy_ratio: float = 1.0,
    save_baseline_actions: bool = False,
    num_threads: int = 1,
) -> dict:
    """Stage2a selfplay を multi-process で実行"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matches_per_worker = distribute_matches(num_matches, num_workers)
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    error_queue = ctx.Queue()
    processes = []

    for i, wm in enumerate(matches_per_worker):
        if wm == 0:
            continue
        worker_seed = derive_worker_seed(base_seed, i)
        worker_output = output_dir / f"worker_{i}"
        p = ctx.Process(
            target=_stage2a_selfplay_worker_fn,
            args=(i, str(worker_output), obs_mode, encoder_config,
                  model_state_path, model_config, wm, worker_seed,
                  experiment_id, run_id, result_queue, error_queue,
                  inference_device, policy_ratio, save_baseline_actions,
                  num_threads),
        )
        p.start()
        processes.append(p)

    _wait_and_validate(processes, error_queue, "selfplay")

    all_stats: list[dict] = []
    while not result_queue.empty():
        _, stats = result_queue.get_nowait()
        all_stats.append(stats)

    return _aggregate_stats(all_stats, num_workers)


# ============ Eval Worker ============

def _stage2a_eval_worker_fn(
    worker_id: int,
    obs_mode: str,
    encoder_config: dict,
    model_state_path: str,
    model_config: dict,
    num_matches: int,
    seed_start: int,
    policy_seat: int,
    result_queue: mp.Queue,
    error_queue: mp.Queue,
    inference_device: str = "cpu",
    num_threads: int = 1,
):
    """subprocess で Stage2a eval を実行"""
    try:
        from mahjong_rl.runner import configure_worker_threads, _parse_encoder_flag
        configure_worker_threads(num_threads)

        from mahjong_rl.stage2a_evaluator import Stage2aEvaluator
        from mahjong_rl.models.stage2a_model import Stage2aModel
        from mahjong_rl.encoders import FlatFeatureEncoder

        device = torch.device(inference_device)

        enc = FlatFeatureEncoder(
            observation_mode=obs_mode,
            shanten_hint=_parse_encoder_flag(encoder_config, "shanten_hint"),
            discard_ukeire_hint=_parse_encoder_flag(encoder_config, "discard_ukeire_hint"),
            current_shanten_input=_parse_encoder_flag(encoder_config, "current_shanten"),
            shape_hint=_parse_encoder_flag(encoder_config, "shape_hint"),
            turn_context=_parse_encoder_flag(encoder_config, "turn_context"),
            opponent_current_shanten=_parse_encoder_flag(encoder_config, "opponent_current_shanten"),
            opponent_tenpai_flag=_parse_encoder_flag(encoder_config, "opponent_tenpai_flag"),
            danger_mask=_parse_encoder_flag(encoder_config, "danger_mask"),
        )

        meta = enc.metadata()
        input_dim = int(np.prod(meta.output_shape))
        hint_ranges = {s: meta.feature_ranges[s]
                       for s in ("shanten_hint", "discard_ukeire_hint")
                       if s in meta.feature_ranges}
        model = Stage2aModel(
            input_dim=input_dim,
            discard_hidden_dims=model_config.get("discard_hidden_dims", [256, 128]),
            optional_hidden_dims=model_config.get("optional_hidden_dims",
                    model_config.get("call_hidden_dims", [128, 64])),
            value_hidden_dims=model_config.get("value_hidden_dims", [128, 64]),
            candidate_dim=model_config.get("candidate_dim", 16),
            optional_scorer_hidden=model_config.get("optional_scorer_hidden", 32),
            semantic_aux_config=model_config.get("semantic_aux"),
            direct_hint_ranges=hint_ranges if hint_ranges else None,
        )
        sd = torch.load(model_state_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)

        evaluator = Stage2aEvaluator(
            model=model, encoder=enc,
            observation_mode=obs_mode,
            device=device,
        )
        metrics = evaluator.evaluate(
            num_matches=num_matches,
            seed_start=seed_start,
            policy_seat=policy_seat,
        )
        result_queue.put((worker_id, metrics))
    except Exception as e:
        error_queue.put((worker_id, str(e)))


def run_stage2a_eval_parallel(
    num_workers: int,
    num_matches: int,
    seed_start: int,
    obs_mode: str,
    encoder_config: dict,
    model_state_path: str,
    model_config: dict,
    eval_mode: str = "single",
    policy_seat: int = 0,
    inference_device: str = "cpu",
    num_threads: int = 1,
) -> dict:
    """Stage2a eval を multi-process で実行"""
    if eval_mode == "rotation":
        # rotation: 各席×各worker に分配
        all_results = []
        for seat in range(4):
            r = run_stage2a_eval_parallel(
                num_workers=num_workers,
                num_matches=num_matches,
                seed_start=seed_start + seat * num_matches,
                obs_mode=obs_mode,
                encoder_config=encoder_config,
                model_state_path=model_state_path,
                model_config=model_config,
                eval_mode="single",
                policy_seat=seat,
                inference_device=inference_device,
                num_threads=num_threads,
            )
            all_results.append(r)
        # 集約
        total_rounds = sum(r["total_rounds"] for r in all_results)
        total_wins = sum(r.get("wins", 0) for r in all_results)
        total_deal_ins = sum(r.get("deal_ins", 0) for r in all_results)
        return {
            "avg_rank": float(np.mean([r["avg_rank"] for r in all_results])),
            "avg_score": float(np.mean([r["avg_score"] for r in all_results])),
            "win_rate": total_wins / total_rounds if total_rounds > 0 else 0.0,
            "deal_in_rate": total_deal_ins / total_rounds if total_rounds > 0 else 0.0,
            "num_matches": num_matches * 4,
            "total_rounds": total_rounds,
            "eval_mode": "rotation",
        }

    matches_per_worker = distribute_matches(num_matches, num_workers)
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    error_queue = ctx.Queue()
    processes = []

    offset = 0
    for i, wm in enumerate(matches_per_worker):
        if wm == 0:
            continue
        p = ctx.Process(
            target=_stage2a_eval_worker_fn,
            args=(i, obs_mode, encoder_config, model_state_path,
                  model_config, wm, seed_start + offset, policy_seat,
                  result_queue, error_queue, inference_device,
                  num_threads),
        )
        p.start()
        processes.append(p)
        offset += wm

    _wait_and_validate(processes, error_queue, "eval")

    all_metrics: list[dict] = []
    while not result_queue.empty():
        _, m = result_queue.get_nowait()
        all_metrics.append(m)

    return _aggregate_eval_metrics(all_metrics)


# ============ Helpers ============

_WORKER_TIMEOUT = 600  # seconds


def _wait_and_validate(
    processes: list,
    error_queue: mp.Queue,
    label: str,
) -> None:
    """worker の完了を待ち、timeout / 異常 exitcode を厳密に検知する"""
    for p in processes:
        p.join(timeout=_WORKER_TIMEOUT)

    errors: list[str] = []

    # error queue から報告されたエラー
    while not error_queue.empty():
        wid, msg = error_queue.get_nowait()
        errors.append(f"{label}_worker_{wid}: {msg}")

    # timeout / 非ゼロ exitcode
    for i, p in enumerate(processes):
        if p.is_alive():
            errors.append(f"{label}_worker_{i}: timeout ({_WORKER_TIMEOUT}s)")
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
        elif p.exitcode is not None and p.exitcode != 0:
            errors.append(
                f"{label}_worker_{i}: non-zero exit code {p.exitcode}")

    if errors:
        raise RuntimeError(
            f"Stage2a {label} worker errors:\n" + "\n".join(errors))


_SUM_KEYS = [
    "num_matches", "total_matches", "total_steps",
    "discard_count", "call_count",
    "num_rounds", "tsumo_count", "ron_count", "ryukyoku_count",
    "policy_wins", "policy_deal_ins", "policy_draws",
    "policy_win_by_tsumo", "policy_win_by_ron",
]


def _aggregate_stats(stats_list: list[dict], num_workers: int) -> dict:
    """worker stats を合算"""
    if not stats_list:
        return {"total_steps": 0, "num_matches": 0}
    agg: dict = {}
    for key in _SUM_KEYS:
        agg[key] = sum(s.get(key, 0) for s in stats_list)
    agg["num_workers"] = num_workers
    return agg


def _aggregate_eval_metrics(metrics_list: list[dict]) -> dict:
    """eval worker metrics を集約"""
    if not metrics_list:
        return {"avg_rank": 0, "avg_score": 0, "num_matches": 0}
    total_matches = sum(m["num_matches"] for m in metrics_list)
    total_rounds = sum(m.get("total_rounds", 0) for m in metrics_list)
    total_wins = sum(m.get("wins", 0) for m in metrics_list)
    total_deal_ins = sum(m.get("deal_ins", 0) for m in metrics_list)

    # weighted avg_rank / avg_score
    w_rank = sum(m["avg_rank"] * m["num_matches"] for m in metrics_list)
    w_score = sum(m["avg_score"] * m["num_matches"] for m in metrics_list)

    return {
        "avg_rank": w_rank / total_matches if total_matches > 0 else 0.0,
        "avg_score": w_score / total_matches if total_matches > 0 else 0.0,
        "win_rate": total_wins / total_rounds if total_rounds > 0 else 0.0,
        "deal_in_rate": total_deal_ins / total_rounds if total_rounds > 0 else 0.0,
        "num_matches": total_matches,
        "total_rounds": total_rounds,
        "wins": total_wins,
        "deal_ins": total_deal_ins,
        "eval_mode": metrics_list[0].get("eval_mode", "fixed_seat"),
    }
