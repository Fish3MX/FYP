#!/usr/bin/env python3
"""Batch-compile many test requests and collect acceptance statistics.

This is the measurement harness behind RQ2: how often does a generation attempt
produce a scenario that is actually usable as a test, how many retries does it
take, and when it fails, why?

The important property is that the loop under test is the production one. The
script does not reimplement compilation; it calls `compile_request` and only
instruments the transport to count how many model calls each request consumed.

Typical use
-----------
    # A quick sanity batch: 6 requests, one trial each
    python scripts/batch_compile.py

    # The ablation condition: same requests, geometric gate disabled
    python scripts/batch_compile.py --no-verify --tag ablation

    # A bigger run for the paper
    python scripts/batch_compile.py --trials 3 --tag full

    # Control condition: the deterministic template instead of a model
    python scripts/batch_compile.py --template --tag template

Outputs, under `runs/batch_<timestamp>_<tag>/`:
    specs/<id>_t<N>.json     every accepted scenario
    results.csv              one row per (request, trial)
    summary.json             aggregate statistics
    failures.log             human-readable failure details

Because `runs/` is git-ignored, nothing here can pollute the repository.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

import llm_compiler  # noqa: E402
import llm_env  # noqa: E402
import verifier  # noqa: E402
from scene_spec import SpecError, parse_spec  # noqa: E402

LOG = logging.getLogger("batch_compile")

# A spread of domains, so the aggregate is not just "one task works".
DEFAULT_REQUESTS: List[Tuple[str, str]] = [
    ("fire_apartment",
     "Test a drone searching for a survivor inside a burning two-storey "
     "apartment with smoke on the upper floor."),
    ("fire_warehouse",
     "Test a drone locating a trapped worker in a burning warehouse with "
     "collapsed shelving."),
    ("urban_car_search",
     "Test a drone searching a city street for a specific parked car after an "
     "accident."),
    ("urban_person_search",
     "Test a drone finding a missing person in a dense urban block with narrow "
     "alleys."),
    ("forest_rescue",
     "Test a drone searching a forest clearing for a lost hiker near a tent."),
    ("disaster_rubble",
     "Test a drone searching an earthquake rubble site for survivors under "
     "collapsed concrete."),
    ("industrial_inspection",
     "Test a drone inspecting an industrial facility with pipes and pillars for "
     "a leaking valve."),
    ("indoor_office",
     "Test a drone searching a two-storey office building for a person hiding "
     "in a back room."),
]

# Failure taxonomy, matched against the error text in this order. The first
# match wins, so more specific patterns come first.
FAILURE_PATTERNS: List[Tuple[str, str]] = [
    ("json_parse", "not valid JSON"),
    ("json_parse", "no parsable JSON"),
    ("structural", "failed validation"),
    ("asset_hallucination", "not in the permitted catalogue"),
    ("unsolvable", "no feasible path"),
    ("occlusion", "can see the target directly"),
    ("too_easy", "too easy"),
    ("trivial_detour", "almost directly reachable"),
    ("requirement", "cannot be satisfied"),
    ("sanity", "spawn altitude"),
    ("sanity", "too short"),
    ("sanity", "physics=false"),
    ("other", ""),
]


def classify(error_text: str) -> str:
    """Map an error message onto a failure category."""
    for category, needle in FAILURE_PATTERNS:
        if not needle or needle in error_text:
            return category
    return "other"


@dataclass
class Trial:
    request_id: str
    trial: int
    success: bool
    attempts: int
    failure: str = ""
    error: str = ""
    path_length_m: Optional[float] = None
    detour_ratio: Optional[float] = None
    occluded: Optional[bool] = None
    seconds: float = 0.0
    spec_path: str = ""


@dataclass
class BatchConfig:
    mode: str = "llm"                 # llm | template
    verify_geometry: bool = True
    max_attempts: int = 3
    temperature: float = 0.2
    trials: int = 1
    tag: str = "run"
    requests: List[Tuple[str, str]] = field(default_factory=list)
    out_dir: Path = Path("runs")


# --------------------------------------------------------------------------- #
# One trial
# --------------------------------------------------------------------------- #
def run_llm_trial(
    request_id: str,
    request_text: str,
    trial_index: int,
    config: BatchConfig,
    env,
    out_dir: Path,
) -> Trial:
    """Compile once through the production loop, counting model calls."""
    counter = {"n": 0}
    real_call = llm_compiler.call_chat_api

    def counting_call(*args, **kwargs):
        counter["n"] += 1
        return real_call(*args, **kwargs)

    started = time.time()
    llm_compiler.call_chat_api = counting_call
    try:
        spec = llm_compiler.compile_request(
            request_text,
            base_url=env.base_url,
            api_key=env.api_key,
            model=env.model,
            max_attempts=config.max_attempts,
            temperature=config.temperature,
            verify_geometry=config.verify_geometry,
        )
    except RuntimeError as exc:
        elapsed = time.time() - started
        return Trial(
            request_id=request_id,
            trial=trial_index,
            success=False,
            attempts=counter["n"] or config.max_attempts,
            failure=classify(str(exc)),
            error=str(exc)[:2000],
            seconds=elapsed,
        )
    finally:
        llm_compiler.call_chat_api = real_call

    elapsed = time.time() - started
    spec_path = out_dir / "specs" / f"{request_id}_t{trial_index}.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        json.dumps(llm_compiler.spec_to_dict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = verifier.verify(spec)
    return Trial(
        request_id=request_id,
        trial=trial_index,
        success=True,
        attempts=counter["n"] or 1,
        path_length_m=result.path_length_m,
        detour_ratio=result.detour_ratio,
        occluded=result.initially_occluded,
        seconds=elapsed,
        spec_path=str(spec_path),
    )


def run_template_trial(
    request_id: str,
    request_text: str,
    trial_index: int,
    config: BatchConfig,
    out_dir: Path,
) -> Trial:
    """Control condition: the deterministic template instead of a model."""
    started = time.time()
    data = llm_compiler.template_spec(request_text)
    try:
        spec = parse_spec(data)
    except SpecError as exc:
        return Trial(
            request_id=request_id,
            trial=trial_index,
            success=False,
            attempts=1,
            failure="structural",
            error=str(exc)[:2000],
            seconds=time.time() - started,
        )

    problems = llm_compiler.validate_all(spec)
    if problems:
        return Trial(
            request_id=request_id,
            trial=trial_index,
            success=False,
            attempts=1,
            failure=classify("\n".join(problems)),
            error="\n".join(problems)[:2000],
            seconds=time.time() - started,
        )

    result = verifier.verify(spec)
    elapsed = time.time() - started
    if not result.usable:
        return Trial(
            request_id=request_id,
            trial=trial_index,
            success=False,
            attempts=1,
            failure=classify("\n".join(result.failures)),
            error="\n".join(result.failures)[:2000],
            seconds=elapsed,
        )

    spec_path = out_dir / "specs" / f"{request_id}_t{trial_index}.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return Trial(
        request_id=request_id,
        trial=trial_index,
        success=True,
        attempts=1,
        path_length_m=result.path_length_m,
        detour_ratio=result.detour_ratio,
        occluded=result.initially_occluded,
        seconds=elapsed,
        spec_path=str(spec_path),
    )


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def summarise(trials: Sequence[Trial], config: BatchConfig) -> Dict[str, Any]:
    total = len(trials)
    successes = [t for t in trials if t.success]
    failures = [t for t in trials if not t.success]

    histogram: Dict[str, int] = {}
    for trial in failures:
        histogram[trial.failure] = histogram.get(trial.failure, 0) + 1

    attempts = [t.attempts for t in trials]
    first_try = [t for t in successes if t.attempts == 1]

    return {
        "mode": config.mode,
        "verify_geometry": config.verify_geometry,
        "trials_per_request": config.trials,
        "requests": len({t.request_id for t in trials}),
        "total_trials": total,
        "accepted": len(successes),
        "acceptance_rate": round(len(successes) / total, 4) if total else 0.0,
        "first_try_accepted": len(first_try),
        "first_try_rate": round(len(first_try) / total, 4) if total else 0.0,
        "mean_attempts": round(sum(attempts) / len(attempts), 3) if attempts else 0.0,
        "mean_seconds": round(
            sum(t.seconds for t in trials) / total, 2
        ) if total else 0.0,
        "failure_histogram": dict(sorted(histogram.items(), key=lambda kv: -kv[1])),
        "path_lengths_m": [
            round(t.path_length_m, 2) for t in successes if t.path_length_m
        ],
        "detour_ratios": [
            round(t.detour_ratio, 3) for t in successes if t.detour_ratio
        ],
    }


def write_csv(path: Path, trials: Sequence[Trial]) -> None:
    width = max((len(t.error) for t in trials), default=0)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "request_id", "trial", "success", "attempts", "failure",
                "path_length_m", "detour_ratio", "occluded", "seconds", "spec_path",
            ]
        )
        for trial in trials:
            writer.writerow(
                [
                    trial.request_id, trial.trial, int(trial.success), trial.attempts,
                    trial.failure,
                    "" if trial.path_length_m is None else round(trial.path_length_m, 2),
                    "" if trial.detour_ratio is None else round(trial.detour_ratio, 3),
                    "" if trial.occluded is None else int(trial.occluded),
                    round(trial.seconds, 2),
                    trial.spec_path,
                ]
            )
    LOG.info("results table written to %s (%d rows)", path, len(trials))


def print_summary(summary: Dict[str, Any]) -> None:
    print()
    print("=" * 68)
    print(f"batch summary  (mode={summary['mode']}, "
          f"geometry={summary['verify_geometry']})")
    print("=" * 68)
    print(f"  requests          : {summary['requests']}")
    print(f"  trials            : {summary['total_trials']}")
    print(f"  accepted          : {summary['accepted']} "
          f"({summary['acceptance_rate'] * 100:.1f}%)")
    print(f"  accepted first try: {summary['first_try_accepted']} "
          f"({summary['first_try_rate'] * 100:.1f}%)")
    print(f"  mean attempts     : {summary['mean_attempts']}")
    print(f"  mean seconds      : {summary['mean_seconds']}")

    if summary["failure_histogram"]:
        print("  failures:")
        for category, count in summary["failure_histogram"].items():
            print(f"    {category:<22} {count}")

    if summary["path_lengths_m"]:
        lengths = summary["path_lengths_m"]
        print(f"  path length (m)   : min {min(lengths)}, "
              f"median {sorted(lengths)[len(lengths) // 2]}, max {max(lengths)}")
    print("=" * 68)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch-compile test requests and report acceptance statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--template", action="store_true",
        help="control condition: use the deterministic template, no model calls",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="ablation: disable geometric verification inside the loop",
    )
    parser.add_argument("--trials", type=int, default=1, help="repeats per request")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--tag", default="run", help="label for the output folder")
    parser.add_argument(
        "--requests-file", default="",
        help="JSON file: [[\"id\", \"request text\"], ...]; defaults to a built-in set",
    )
    parser.add_argument("--limit", type=int, default=0, help="use only the first N requests")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between trials")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- requests ----
    if args.requests_file:
        raw = json.loads(Path(args.requests_file).read_text(encoding="utf-8"))
        requests = [(str(item[0]), str(item[1])) for item in raw]
    else:
        requests = list(DEFAULT_REQUESTS)
    if args.limit:
        requests = requests[: args.limit]

    # ---- credentials, only needed for the model condition ----
    env = None
    if not args.template:
        env = llm_env.resolve_config()
        if not env.configured:
            LOG.error(
                "no API key. Put LLM_API_KEY in .env or the environment, or run "
                "--template for the control condition."
            )
            return 2
        LOG.info("endpoint: %s (%s)", env.base_url, env.model)

    config = BatchConfig(
        mode="template" if args.template else "llm",
        verify_geometry=not args.no_verify,
        max_attempts=args.max_attempts,
        temperature=args.temperature,
        trials=args.trials,
        tag=args.tag,
        requests=requests,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = config.out_dir / f"batch_{stamp}_{config.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("output directory: %s", out_dir)

    # ---- run ----
    trials: List[Trial] = []
    total = len(requests) * config.trials
    index = 0

    for request_id, request_text in requests:
        for trial_index in range(1, config.trials + 1):
            index += 1
            LOG.info("[%d/%d] %s trial %d", index, total, request_id, trial_index)

            if args.template:
                trial = run_template_trial(
                    request_id, request_text, trial_index, config, out_dir
                )
            else:
                trial = run_llm_trial(
                    request_id, request_text, trial_index, config, env, out_dir
                )

            trials.append(trial)
            status = "ok" if trial.success else f"FAIL ({trial.failure})"
            LOG.info(
                "       -> %s, attempts=%d, %.1f s",
                status, trial.attempts, trial.seconds,
            )
            if args.delay:
                time.sleep(args.delay)

    # ---- report ----
    summary = summarise(trials, config)
    summary["generated_at"] = datetime.now().astimezone().isoformat()
    summary["out_dir"] = str(out_dir)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(out_dir / "results.csv", trials)

    with (out_dir / "failures.log").open("w", encoding="utf-8") as handle:
        for trial in trials:
            if trial.success:
                continue
            handle.write(
                f"=== {trial.request_id} trial {trial.trial} "
                f"[{trial.failure}] ===\n{trial.error}\n\n"
            )

    print_summary(summary)
    LOG.info("artefacts in %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())