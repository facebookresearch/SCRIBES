# SCRIBES

Code for reproducing the main results of *[SCRIBES: Web-Scale Script-Based
Semi-Structured Data Extraction with Reinforcement
Learning](https://arxiv.org/abs/2510.01832)*. Semi-structured content in
HTML tables, lists, and infoboxes accounts for a substantial share of
factual data on the web, yet reliably extracting structured information
from it remains challenging. Rather than processing each page individually,
SCRIBES fine-tunes an LLM with reinforcement learning to write reusable
Python extraction scripts, using layout similarity across webpages on the
same site as a reward signal so scripts generalize across a site instead of
overfitting to one page.

This repository covers two things:
1. Fine-tuning an LLM to generate triple-extraction scripts.
2. Running the fine-tuned model to extract triples from a given HTML page.

## Installation

This repo does not include [verl](https://github.com/volcengine/verl)
itself — clone it separately and apply the patches here:

```bash
git clone https://github.com/volcengine/verl.git
cd verl && git checkout d57bfb02b3bf54c4547aaf83fa055f7a1c1cdc6b
git apply /path/to/scribes/verl_patches/constants_ppo.patch
git apply /path/to/scribes/verl_patches/ray_trainer.patch
git apply /path/to/scribes/verl_patches/fsdp_sft_trainer.patch
git apply /path/to/scribes/verl_patches/reward.patch
git apply /path/to/scribes/verl_patches/dp_actor.patch
pip install -e .   # follow verl's own install instructions for your hardware/GPU stack
```

Then install this repo's own dependencies:

```bash
pip install beautifulsoup4 fuzzywuzzy python-Levenshtein munkres scipy vllm requests
```

## Usage

Run these from the root of your verl checkout. Each step's directory has a
full README with every flag; this is the minimal path through all three.

**1. Build training data** ([`examples/data_preprocess/`](examples/data_preprocess/)):

```bash
python3 /path/to/scribes/examples/data_preprocess/triple_extraction_group_generalization.py \
    --reuse_test_groups_dir /path/to/an/existing/test_groups_dir \
    --local_dir /path/to/output_dir
```

Clones [SemiBench](https://github.com/facebookresearch/SemiBench)
automatically and builds `train.parquet` / `test.parquet` from it.

**2. Fine-tune with GRPO** ([`recipe/triple_extraction/`](recipe/triple_extraction/)):

```bash
MODEL_PATH=/path/to/Qwen2.5-14B-Instruct \
TRAIN_FILE=/path/to/output_dir/train.parquet \
VAL_FILE=/path/to/output_dir/test.parquet \
CKPT_DIR=/path/to/checkpoints \
bash /path/to/scribes/recipe/triple_extraction/run_qwen2.5_14b_grpo.sh
```

**3. Extract triples from an HTML page with the fine-tuned model** ([`examples/grpo_trainer/`](examples/grpo_trainer/)):

```bash
python3 /path/to/scribes/examples/grpo_trainer/eval_single_html.py page.html \
    --model_dir /path/to/checkpoints/<checkpoint> \
    --port 8001 --tp 8
```

## Contents

| Stage | Directory | What it does |
|---|---|---|
| 0 | [`verl_patches/`](verl_patches/) | Patches to apply to a stock verl checkout (see `examples/data_preprocess/README.md` §0 for the pinned commit) |
| 1 | [`examples/data_preprocess/`](examples/data_preprocess/) | Builds training data from the public [SemiBench](https://github.com/facebookresearch/SemiBench) benchmark |
| 2 | [`recipe/triple_extraction/`](recipe/triple_extraction/) | Reward function + GRPO training config/launcher |
| 3 | [`examples/grpo_trainer/`](examples/grpo_trainer/) | Extracts triples from a single HTML file with the fine-tuned model |

Each directory has its own README with exact setup/run instructions.

## Citation

```bibtex
@inproceedings{liu2026scribes,
  title={{SCRIBES}: Web-Scale Script-Based Semi-Structured Data Extraction with Reinforcement Learning},
  author={Liu, Shicheng and Sun, Kai and Fu, Lisheng and Chen, Xilun and Zhang, Xinyuan and Lin, Zhaojiang and Shao, Rulin and Liu, Yue and Kumar, Anuj and Yih, Wen-tau and Dong, Xin Luna},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=gQSnEIA3Z3}
}
```

## License

This code is released under [CC BY-NC 4.0](LICENSE) (non-commercial use
only). It builds training data from
[SemiBench](https://github.com/facebookresearch/SemiBench), which is
separately released under the same CC BY-NC 4.0 license — see that repo for
details.