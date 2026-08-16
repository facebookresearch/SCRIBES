# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Extract knowledge triples from a single HTML file using a fine-tuned model.

Starts a local vLLM OpenAI-compatible server for the given model checkpoint
(or talks to one you've already started), asks the model to generate a
Python extraction script for the given HTML, executes that script in a
sandbox, and prints the resulting triples.
"""

import argparse
import atexit
import json
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

# code_execution.py lives in recipe/triple_extraction/, HTMLCompressor lives
# in examples/data_preprocess/ — both siblings of this file's parent dir.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(_THIS_DIR.parent / "data_preprocess"))
sys.path.append(str(_THIS_DIR.parent.parent / "recipe" / "triple_extraction"))

from code_execution import execute_code as _execute_code  # noqa: E402
from preprocessing import HTMLCompressor  # noqa: E402

SYSTEM_INSTRUCTION = (
    "Your task is to generate semantic triples from a given HTML. "
    "A triple contains a subject, a predicate, and an object. "
    "You should write python code to extract triples from the HTML. "
    "The final executable function should be called `def main(html) -> List[List[str]]:`, "
    "where the inner list is a 3-list. You should output the python code only. "
    "Feel free to add comments to explain your code. Do not include any text other than the code in your response."
    "The same script will also be used for similar webpages, so you should make the code generalizable."
)

_started_procs: list[subprocess.Popen] = []


def _cleanup_started_procs():
    for p in _started_procs:
        try:
            stop_proc(p)
        except Exception:
            pass


atexit.register(_cleanup_started_procs)


def execute_code(code: str, html_content: str, timeout_sec: int = 120):
    return _execute_code(code, html_content, timeout_sec, return_stderr=False)


def wait_for_server(url: str, timeout: int = 500):
    """Poll the vLLM server until /health returns status 200 or timeout."""
    for _ in range(timeout):
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for server {url}")


def start_vllm(model_dir: str, port: int, tp: int = 8, max_num_seqs: int | None = None) -> subprocess.Popen:
    """Launch `vllm serve` in the background and return the process."""
    cmd = ["vllm", "serve", model_dir, "--tensor-parallel-size", str(tp), "--port", str(port)]
    if isinstance(max_num_seqs, int) and max_num_seqs > 0:
        cmd.extend(["--max-num-seqs", str(max_num_seqs)])

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True)
    wait_for_server(f"http://localhost:{port}/health")

    _started_procs.append(proc)
    return proc


def stop_proc(proc: subprocess.Popen):
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def llm_completion(model: str, messages, server_url: str, temperature: float = 0.0):
    """Make a chat completion request to an OpenAI-compatible vLLM server."""
    endpoint = f"{server_url}/v1/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": temperature}
    resp = requests.post(endpoint, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_triples(html_content: str, model_name: str, server_url: str) -> dict:
    """Ask the model for an extraction script, run it, and return the result."""
    compressor = HTMLCompressor()
    prompt = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": compressor.compress_html(html_content)},
    ]

    try:
        generated_code = llm_completion(model_name, prompt, server_url, temperature=0.0)
        execution_result = execute_code(generated_code, html_content)
        predicted_triples = execution_result if execution_result is not None else []
        return {
            "predicted_triples": predicted_triples,
            "extraction_success": execution_result is not None,
            "generated_code": generated_code,
            "empty_result": len(predicted_triples) == 0,
            "error": None,
        }
    except Exception as e:
        return {
            "predicted_triples": [],
            "extraction_success": False,
            "generated_code": generated_code if "generated_code" in locals() else "",
            "empty_result": True,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html_file", help="Path to an HTML file to extract triples from.")
    parser.add_argument("--model_dir", help="Path to the fine-tuned model checkpoint to serve with vLLM.")
    parser.add_argument(
        "--model_name",
        default=None,
        help="Model name to request from the server (defaults to --model_dir, or is required with --server_url).",
    )
    parser.add_argument("--port", type=int, default=8001, help="Port to serve/reach the model on.")
    parser.add_argument("--tp", type=int, default=8, help="Tensor-parallel size for vLLM.")
    parser.add_argument(
        "--server_url",
        default=None,
        help="Use an already-running OpenAI-compatible server (e.g. http://localhost:8001) instead of launching one.",
    )
    parser.add_argument("--output", default=None, help="Write JSON result here instead of stdout.")
    args = parser.parse_args()

    if not args.server_url and not args.model_dir:
        parser.error("pass --model_dir (to launch a server) or --server_url (to use an existing one)")

    html_content = Path(args.html_file).read_text(encoding="utf-8", errors="replace")

    proc = None
    if args.server_url:
        server_url = args.server_url.rstrip("/")
        model_name = args.model_name or args.model_dir
        if not model_name:
            parser.error("--model_name is required when using --server_url without --model_dir")
    else:
        server_url = f"http://localhost:{args.port}"
        model_name = args.model_name or args.model_dir
        logging.info(f"Starting vLLM for {args.model_dir} on port {args.port} ...")
        proc = start_vllm(args.model_dir, args.port, tp=args.tp)

    try:
        result = extract_triples(html_content, model_name, server_url)
    finally:
        if proc is not None:
            stop_proc(proc)

    output = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
