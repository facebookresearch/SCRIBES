# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import sys
from pathlib import Path

# code_execution.py and triple_accuracy_evaluator.py are vendored next to this file.
sys.path.append(str(Path(__file__).resolve().parent))

from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

# Import the concrete implementations under private aliases so we can expose
# typed, documented wrappers with the original public names.
from code_execution import execute_code as _execute_code  # noqa: N812 – keep alias
from triple_accuracy_evaluator import main as _evaluate_triples  # noqa: N812


def execute_code(
    code: str,
    html_content: str,
    timeout_sec: int = 60,
    *,
    return_stderr: bool = False,
) -> Union[Optional[Any], Tuple[Optional[Any], str]]:
    """Typed wrapper around :pyfunc:`code_execution.execute_code`."""

    return _execute_code(
        code,
        html_content,
        timeout_sec,
        return_stderr=return_stderr,
    )


def evaluate_triples(
    ground_truth_triples: List[List[str]],
    predicted_triples: List[List[str]],
    model_name: str = "llama3.3-70b-instruct",
) -> Dict[str, float]:
    """Typed wrapper around :pyfunc:`triple_accuracy_evaluator.main`."""

    if len(predicted_triples) >= 10 * len(ground_truth_triples):
        return {"fuzzy_f1": 0.0}

    return _evaluate_triples(
        ground_truth_triples,
        predicted_triples,
        use_llm=False,
        model_name=model_name,
        type_check=True,
        use_greedy=True,
        max_eval_seconds=60
    )


def compute_member_reward_for_process(args: Tuple[str, str, Union[str, List[List[str]]]]) -> float:
    solution_str, html_content, raw_ground_truth = args
    execution_result = execute_code(solution_str, html_content)
    if execution_result is None:
        return 0.0

    if isinstance(raw_ground_truth, str):
        parsed_ground_truth = eval(raw_ground_truth)
    else:
        parsed_ground_truth = raw_ground_truth
        assert isinstance(parsed_ground_truth, list)

    results = evaluate_triples(parsed_ground_truth, execution_result)
    return results["fuzzy_f1"]


def reward_func(data_source, solution_str, ground_truth, extra_info=None):
    # Prepare all members (this member + others) to be processed in parallel
    keyed_specs: List[Tuple[str, Tuple[str, str, Union[str, List[List[str]]]]]] = []
    keyed_specs.append(("this", (solution_str, extra_info["html_content"], ground_truth)))
    for idx, other_member in enumerate(extra_info["other_members"]):
        keyed_specs.append((
            f"other_{idx}",
            (solution_str, other_member["html_content"], other_member["ground_truth"])  # type: ignore[arg-type]
        ))

    # Execute all calculations concurrently (process-based for CPU-bound workloads) and collect rewards
    rewards: Dict[str, float] = {}
    max_workers = max(1, min(32, len(keyed_specs)))
    # Avoid nested process pools: if we're already in a child process, fall back to threads
    in_subprocess = multiprocessing.current_process().name != "MainProcess"
    ExecutorClass = ThreadPoolExecutor if in_subprocess else ProcessPoolExecutor
    with ExecutorClass(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(compute_member_reward_for_process, spec): key
            for key, spec in keyed_specs
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            rewards[key] = float(future.result())

    this_member_reward = rewards.get("this", 0.0)
    other_member_rewards = [rewards[k] for k in rewards.keys() if k != "this"]

    # NOTE: this reward is only suited for examples with at least 1 other member
    return (this_member_reward + sum(other_member_rewards)) / (len(other_member_rewards) + 1)
