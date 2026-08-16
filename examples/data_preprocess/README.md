# Building the triple-extraction training data

This builds the train/test parquet files used to fine-tune the model, from
the [SemiBench](https://github.com/facebookresearch/SemiBench) benchmark
(Sun et al., 2025): 351 webpages with human-annotated (subject, predicate,
object) triples, drawn from `triple_whole.json` (251 pages) and
`triple_whole_extra.json` (100 pages).

## 0. verl version

This pipeline's output is meant to be trained on with the official
[volcengine/verl](https://github.com/volcengine/verl), pinned to the exact
commit used as the base for this project:

```bash
git clone https://github.com/volcengine/verl.git
cd verl && git checkout d57bfb02b3bf54c4547aaf83fa055f7a1c1cdc6b  # "bump main branch version to v0.5.0.dev (#2718)"
```

On top of that commit, apply the patches in `verl_patches/` (each one
touches a single file), from inside the verl checkout:

```bash
git apply verl_patches/constants_ppo.patch
git apply verl_patches/ray_trainer.patch
git apply verl_patches/fsdp_sft_trainer.patch
git apply verl_patches/reward.patch
git apply verl_patches/dp_actor.patch
```

Each patch has been verified to apply cleanly at this exact commit. In short,
what they do:

- `constants_ppo.patch` — forwards `VERL_AUTO_PADDING` into Ray workers'
  runtime env instead of dropping it.
- `ray_trainer.patch` — sets `drop_last=False` on the train dataloader, and
  defensively slices batches back into alignment before each `.union()` call
  (generation, reward, log-prob, ref log-prob, values) in case a worker
  returns a different batch size than it was given.
- `fsdp_sft_trainer.patch` — sets `drop_last=False` on the SFT
  train/val dataloaders and samplers.
- `reward.patch` — loads each custom reward function file into a
  path-keyed module name (instead of always `"custom_module"`), so the same
  file maps to a stable module object across repeated calls and doesn't hit
  cloudpickle errors from duplicate function names.
- `dp_actor.patch` — calls `torch.cuda.empty_cache()` and `gc.collect()`
  after each optimizer step, to reduce GPU memory fragmentation.

## 1. Install dependencies

```bash
pip install datasets beautifulsoup4
```

`git` must also be available on your `PATH` (used to clone SemiBench).

## 2. Run the script

```bash
python3 triple_extraction_group_generalization.py \
    --reuse_test_groups_dir /path/to/triple_extraction_group_generalization_all_per_group_reuse_prev_test \
    --local_dir /path/to/output_dir
```

On first run this will:
1. Clone `https://github.com/facebookresearch/SemiBench` to a `SemiBench/`
   folder next to this script (override with `--semibench_dir`, or
   pre-clone it yourself and pass `--no_auto_clone`).
2. Download each page's HTML from its Wayback Machine snapshot URL, caching
   it under `<semibench_dir>/.html_cache/` so re-runs don't re-download.
3. Group pages by website, split them into train/test, compress the HTML
   with `preprocessing.HTMLCompressor`, and write `train.parquet` /
   `test.parquet` to `--local_dir`.

Other flags control the train/test split strategy — use exactly one of
`--alt_preprocess`, `--train_all_webpages`, `--pair_two_webpages`,
`--reuse_test_groups_dir`, or `--domain_split_json`. See `--help` for details
on each.

## Notes

- Page HTML is fetched with a plain HTTP GET against the Wayback Machine
  snapshot URL. This returns that snapshot's HTML as archived; if you need
  the JavaScript-rendered version of a page (some sites load content
  dynamically), fetch it yourself with a headless browser instead and feed
  it through `HTMLCompressor.compress_html()` in place of
  `fetch_html_with_cache`.
- SemiBench is released under CC BY-NC 4.0 (non-commercial, benchmarking
  use) — see the license in that repo before redistributing data built from
  it.
