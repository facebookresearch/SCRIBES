# Testing: extract triples from an HTML file with the fine-tuned model

Runs the fine-tuned model on a single HTML file: the model generates a
Python extraction script for that page, the script is executed in a sandbox,
and the resulting triples are printed as JSON.

## 1. Install dependencies

```bash
pip install vllm requests beautifulsoup4
```

`vllm` is only needed if you want this script to launch the model server
itself (the default); skip it if you're pointing at an already-running
OpenAI-compatible server with `--server_url`.

## 2. Run it

Launching a server for a local checkpoint:

```bash
python3 eval_single_html.py page.html \
    --model_dir /path/to/checkpoint \
    --port 8001 \
    --tp 8
```

Or against a server you've already started (e.g. `vllm serve <model> --port 8001`):

```bash
python3 eval_single_html.py page.html \
    --server_url http://localhost:8001 \
    --model_name /path/to/checkpoint
```

Output (stdout, or `--output result.json`):

```json
{
  "predicted_triples": [["subject", "predicate", "object"], ...],
  "extraction_success": true,
  "generated_code": "def main(html): ...",
  "empty_result": false,
  "error": null
}
```

`predicted_triples` is empty and `error` is set if the model's generated
code failed the sandbox's static-import check, raised an exception, timed
out, or produced no output — the raw `generated_code` is still returned so
you can see what the model produced.

## Notes

- This runs the same prompt, HTML compression (`preprocessing.HTMLCompressor`,
  vendored in `../data_preprocess/`), and sandboxed execution
  (`code_execution.py`, vendored in `../../recipe/triple_extraction/`) used
  during training, so results are directly comparable to training-time
  reward.
- The sandbox (`code_execution.py`) enforces POSIX resource limits and is
  Linux/macOS only.
- For evaluating many pages at once with aggregate metrics (not just one
  file), see the group-generalization eval pipeline this script was
  extracted from — not included in this release.
