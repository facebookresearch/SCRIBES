# Training: GRPO fine-tuning for triple extraction

Fine-tunes a base model (Qwen2.5-14B-Instruct) with GRPO so it generates
better HTML-to-triple extraction scripts, using the reward function in this
directory and the train/test parquet built in the preprocessing step.

## 1. Set up verl

Follow `../../examples/data_preprocess/README.md` section "0. verl version"
to clone the official verl at the pinned commit and apply the patches in
`verl_patches/`.

## 2. Install dependencies

```bash
pip install fuzzywuzzy python-Levenshtein munkres scipy beautifulsoup4
```

## 3. Build the training data

Run `examples/data_preprocess/triple_extraction_group_generalization.py`
with `--reuse_test_groups_dir` (see its README) to produce `train.parquet`
and `test.parquet`.

## 4. Launch training

From the root of your verl checkout:

```bash
MODEL_PATH=/path/to/Qwen2.5-14B-Instruct \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/test.parquet \
CKPT_DIR=/path/to/checkpoints \
bash /path/to/this/recipe/run_qwen2.5_14b_grpo.sh
```

This runs `verl.trainer.main_ppo` with `config/triple_extraction_trainer.yaml`
(the recipe's fixed hyperparameters — GRPO, KL settings, batch sizes, rollout
config) layered on top of verl's stock `ppo_trainer.yaml`; the run script
fills in the parts that are specific to each launch (model path, data paths,
checkpoint dir, experiment name). Pass extra `key=value` Hydra overrides as
extra arguments to the script to change anything else — e.g. GPU count,
`trainer.total_epochs`, or context length.

Config composition (`defaults: [ppo_trainer, _self_]` pulling in verl's own
`ppo_trainer.yaml` via `hydra.searchpath`) has been verified with Hydra's
`compose()` API — resolves cleanly and all overrides take effect as expected.

## How the reward works

`reward_score_group_generalization_avg.py` (loaded via
`custom_reward_function.path`/`.name=reward_func`) scores a generated
extraction script by:
1. Running it in a resource-limited sandboxed subprocess
   (`code_execution.py`) against the page's HTML.
2. Comparing the extracted triples to the gold triples with fuzzy bipartite
   matching (`triple_accuracy_evaluator.py`), returning `fuzzy_f1`.
3. Doing this not just for the page the script was written against, but for
   every other page in the same website group (`extra_info["other_members"]`,
   populated by the preprocessing step), and averaging — rewarding scripts
   that generalize across a site rather than overfitting to one page.

`triple_accuracy_evaluator.py` also supports an LLM-based semantic-match
mode (`use_llm=True`), but the reward function here always calls it with
`use_llm=False`, so that path (and its extra `llm_modules` dependency) is
never exercised and isn't included here.

## Notes

- `custom_reward_function.path` in the run script points at the vendored
  `reward_score_group_generalization_avg.py` next to it; move the file and
  update the path if you restructure this repo.
- The reward function's sandboxed code execution (`code_execution.py`) uses
  POSIX resource limits (`RLIMIT_CPU`, `RLIMIT_AS`, ...) and is Linux/macOS
  only.
