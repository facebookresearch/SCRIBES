# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
"""
Preprocess the Triple Extraction dataset to parquet format.
The resulting parquet files follow the unified schema used across the
other preprocessing scripts in this directory:

    {
        "data_source": str,
        "prompt": List[Dict[str, str]],
        "ability": str,
        "reward_model": Dict[str, Any],
        "extra_info": Dict[str, Any],
    }

Each example consists of:
1. A system instruction that asks the model to generate executable Python
   code to extract semantic triples from an HTML page.
2. The user message containing the (compressed) HTML content.
"""

# Standard library imports
import argparse
import json
import os
import re
import subprocess
import sys
import random
import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import datasets


# HTMLCompressor lives in preprocessing.py, vendored next to this script.
sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import HTMLCompressor  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Human-readable name for the dataset. This will be stored in the "data_source"
# field so downstream consumers know where the data came from.
DATA_SOURCE = "triple_extraction_dataset_fullpage"

# System instruction that precedes the HTML in the prompt.
SYSTEM_INSTRUCTION = (
    "Your task is to generate semantic triples from a given HTML. "
    "A triple contains a subject, a predicate, and an object. "
    "You should write python code to extract triples from the HTML. "
    "The final executable function should be called `def main(html) -> List[List[str]]:`, "
    "where the inner list is a 3-list. You should output the python code only. "
    "Feel free to add comments to explain your code. Do not include any text other than the code in your response."
    "The same script will also be used for similar webpages, so you should make the code generalizable."
)

# Public source of the raw annotations: https://github.com/facebookresearch/SemiBench
SEMIBENCH_REPO_URL = "https://github.com/facebookresearch/SemiBench.git"

# The two (triples, urls) file pairs SemiBench ships, covering all 351 pages
# used to build this dataset (251 "whole" + 100 "whole_extra").
SEMIBENCH_TRIPLE_FILES = ["triple_whole.json", "triple_whole_extra.json"]
SEMIBENCH_URL_FILES = ["url_whole.json", "url_whole_extra.json"]

# Default clone location: a directory alongside this script, so a checkout of
# this repo is self-contained instead of reaching into the user's home dir.
DEFAULT_SEMIBENCH_DIR = Path(__file__).resolve().parent / "SemiBench"


# ---------------------------------------------------------------------------
# SemiBench loading: fetch triples + archive.org URLs from the SemiBench repo,
# then re-download each page's full HTML from the Wayback Machine so records
# match the schema this script was originally written against (page_id,
# website, page_url, html_content, triples_annotation).
# ---------------------------------------------------------------------------


def ensure_semibench_repo(semibench_dir: Path, auto_clone: bool) -> Path:
    """Return a local checkout of SemiBench, cloning it if necessary."""
    marker = semibench_dir / "triple_whole.json"
    if marker.exists():
        return semibench_dir
    if not auto_clone:
        raise FileNotFoundError(
            f"SemiBench checkout not found at {semibench_dir}. Clone it manually with "
            f"`git clone {SEMIBENCH_REPO_URL} {semibench_dir}` or pass --semibench_dir, "
            "or drop --no_auto_clone to let this script clone it for you."
        )
    print(f"Cloning {SEMIBENCH_REPO_URL} into {semibench_dir} ...")
    semibench_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", SEMIBENCH_REPO_URL, str(semibench_dir)],
        check=True,
    )
    return semibench_dir


def derive_website(page_id: str) -> str:
    """Recover the website domain from a page_id like 'espn.com-2-2' -> 'espn.com'.

    SemiBench keys pages as '<domain>-<n>' (and occasionally
    '<domain>-<n>-<m>' for grouped sub-pages), so we strip all trailing
    '-<digits>' suffixes rather than just the last one.
    """
    return re.sub(r"(-\d+)+$", "", page_id)


def fetch_html_with_cache(
    page_id: str,
    url: str,
    cache_dir: Path,
    request_delay: float,
    timeout: float,
    max_retries: int = 3,
) -> str:
    """Download a page's archived HTML from the Wayback Machine, with a local cache.

    Note: this issues a plain HTTP GET against the archive.org snapshot URL.
    That is *not* the same fetching pipeline used to build the original
    internal dataset (which used a headless browser to fully render pages),
    so re-fetched HTML may differ slightly in size/content from what
    produced the paper's numbers. See README.md, "Known gaps".
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{page_id}.html"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                html = response.read().decode("utf-8", errors="replace")
            cache_path.write_text(html, encoding="utf-8")
            time.sleep(request_delay)
            return html
        except (urllib.error.URLError, TimeoutError) as err:
            last_err = err
            print(f"[{page_id}] fetch attempt {attempt}/{max_retries} failed: {err}")
            time.sleep(request_delay * attempt)
    raise RuntimeError(f"Failed to fetch {url} for page_id={page_id}") from last_err


def load_semibench_records(
    semibench_dir: Path,
    html_cache_dir: Path,
    request_delay: float,
    timeout: float,
) -> List[Dict[str, Any]]:
    """Load and merge SemiBench's triple/url files into the internal record schema."""
    triples: Dict[str, Any] = {}
    for fname in SEMIBENCH_TRIPLE_FILES:
        with (semibench_dir / fname).open() as fp:
            triples.update(json.load(fp))

    urls: Dict[str, str] = {}
    for fname in SEMIBENCH_URL_FILES:
        with (semibench_dir / fname).open() as fp:
            urls.update(json.load(fp))

    missing_urls = set(triples.keys()) - set(urls.keys())
    if missing_urls:
        raise ValueError(f"{len(missing_urls)} page_ids have triples but no URL, e.g. {sorted(missing_urls)[:5]}")

    records: List[Dict[str, Any]] = []
    for i, page_id in enumerate(sorted(triples.keys())):
        page_url = urls[page_id]
        html_content = fetch_html_with_cache(page_id, page_url, html_cache_dir, request_delay, timeout)
        records.append(
            {
                "page_id": page_id,
                "website": derive_website(page_id),
                "page_url": page_url,
                "html_content": html_content,
                "triples_annotation": triples[page_id],
            }
        )
        if (i + 1) % 25 == 0 or (i + 1) == len(triples):
            print(f"Fetched {i + 1}/{len(triples)} pages from the Wayback Machine")

    return records

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--semibench_dir",
        default=str(DEFAULT_SEMIBENCH_DIR),
        help=(
            "Path to a local checkout of https://github.com/facebookresearch/SemiBench "
            "(the public source of the triple annotations and archive.org page URLs). "
            "Cloned automatically if missing, unless --no_auto_clone is set."
        ),
    )
    parser.add_argument(
        "--no_auto_clone",
        action="store_true",
        help="Do not auto-clone SemiBench if --semibench_dir does not already contain it.",
    )
    parser.add_argument(
        "--html_cache_dir",
        default=None,
        help=(
            "Directory to cache downloaded page HTML in, so re-runs don't re-fetch from "
            "the Wayback Machine. Defaults to <semibench_dir>/.html_cache."
        ),
    )
    parser.add_argument(
        "--request_delay",
        type=float,
        default=0.5,
        help="Seconds to sleep between Wayback Machine requests (politeness delay).",
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=30.0,
        help="Per-request timeout (seconds) when fetching page HTML.",
    )
    parser.add_argument("--local_dir", default=None, help="Output directory for parquet files; if omitted, a default will be chosen based on the preprocessing mode.")
    parser.add_argument(
        "--alt_preprocess",
        "--alt_proprocess",
        dest="alt_preprocess",
        action="store_true",
        help=(
            "Enable alternative preprocessing: keep only 3- and 13-sized groups; "
            "split groups 70/30 into train/val (no within-group split); "
            "for train keep 1 sample per group (others go to other_members); "
            "for val keep all samples (others listed in other_members)."
        ),
    )
    parser.add_argument(
        "--train_all_webpages",
        dest="train_all_webpages",
        action="store_true",
        help=(
            "Use all webpages for training within each selected group (70/30 group split, "
            "keeping only groups of size 3 or 13). Training data generation mirrors validation."
        ),
    )
    parser.add_argument(
        "--pair_two_webpages",
        dest="pair_two_webpages",
        action="store_true",
        help=(
            "Pairing mode: keep only one training example per group, but each example contains two webpages. "
            "For training, randomly select two different pages per train group using a fixed seed. "
            "For validation, for each group of size n, append n examples by pairing (1st,2nd),(2nd,3rd),..., (n-th,1st)."
        ),
    )
    parser.add_argument(
        "--pair_random_seed",
        type=int,
        default=42,
        help="Random seed used to select two webpages per group in pairing mode.",
    )
    parser.add_argument(
        "--reuse_test_groups_dir",
        type=str,
        default=None,
        help=(
            "Directory containing an existing test.parquet whose groups should be reused as the test split. "
            "Training will include all webpages from remaining groups (keeping only groups of size 3 or 13)."
        ),
    )
    parser.add_argument(
        "--domain_split_json",
        type=str,
        default=None,
        help=(
            "Path to a JSON file defining a domain-by-domain split with two sides: "
            "side A domains go to training and side B domains go to test. "
            "Pages whose domains are not listed are excluded."
        ),
    )

    args = parser.parse_args()

    # Guard against incompatible flags.
    enabled_flags = [
        bool(args.alt_preprocess),
        bool(args.train_all_webpages),
        bool(args.pair_two_webpages),
        bool(args.reuse_test_groups_dir),
        bool(args.domain_split_json),
    ]
    if sum(enabled_flags) > 1:
        raise ValueError(
            "Use only one of --alt_preprocess, --train_all_webpages, --pair_two_webpages, "
            "--reuse_test_groups_dir, or --domain_split_json."
        )

    # Choose default output directory if not provided explicitly.
    if args.local_dir is None:
        if args.train_all_webpages:
            args.local_dir = "~/verl/datasets/triple_extraction_group_generalization_all_per_group"
        elif args.pair_two_webpages:
            args.local_dir = "~/verl/datasets/triple_extraction_group_generalization_pair_two_webpages"
        elif args.alt_preprocess:
            args.local_dir = "~/verl/datasets/triple_extraction_group_generalization_1_per_group"
        elif args.reuse_test_groups_dir:
            args.local_dir = "~/verl/datasets/triple_extraction_group_generalization_all_per_group_reuse_prev_test"
        elif args.domain_split_json:
            args.local_dir = "~/verl/datasets/triple_extraction_group_generalization_domain_seperation"
        else:
            args.local_dir = "~/verl/datasets/triple_extraction_group_generalization"

    semibench_dir = Path(os.path.expanduser(args.semibench_dir))
    semibench_dir = ensure_semibench_repo(semibench_dir, auto_clone=not args.no_auto_clone)
    html_cache_dir = (
        Path(os.path.expanduser(args.html_cache_dir))
        if args.html_cache_dir
        else semibench_dir / ".html_cache"
    )

    # -----------------------------------------------------------------------
    # Load raw data: triples + archive.org URLs from SemiBench, HTML fetched
    # (and cached) from the Wayback Machine.
    # -----------------------------------------------------------------------
    raw_examples: List[Dict[str, Any]] = load_semibench_records(
        semibench_dir, html_cache_dir, args.request_delay, args.request_timeout
    )

    # -----------------------------------------------------------------------
    # Group pages by website group (remove last segment of page_id).
    # -----------------------------------------------------------------------
    def get_website_group(page_id: str) -> str:
        parts = page_id.split("-")
        return "-".join(parts[:-1]) if len(parts) > 1 else page_id

    page_groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in raw_examples:
        group_id = get_website_group(item.get("page_id", ""))
        page_groups.setdefault(group_id, []).append(item)

    # -----------------------------------------------------------------------
    # Split logic
    # -----------------------------------------------------------------------
    train_raw: List[Dict[str, Any]] = []
    validation_raw: List[Dict[str, Any]] = []
    # For pairing mode we will populate these instead of *_raw
    train_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    val_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    if args.domain_split_json:
        # Domain-by-domain split using a predefined JSON mapping:
        # All pages whose (sub)domain matches any domain in side A → train
        # All pages whose (sub)domain matches any domain in side B → test
        # Pages not listed in either side are excluded.
        domain_json_path = Path(os.path.expanduser(args.domain_split_json))
        if not domain_json_path.exists():
            raise FileNotFoundError(f"Could not find domain split JSON at {domain_json_path}")

        with domain_json_path.open() as fp:
            domain_spec: Dict[str, Any] = json.load(fp)

        # Identify side A / side B keys flexibly
        side_a_key = None
        side_b_key = None
        for k in domain_spec.keys():
            lk = k.lower()
            if side_a_key is None and lk.startswith("side_a"):
                side_a_key = k
            if side_b_key is None and lk.startswith("side_b"):
                side_b_key = k
        if side_a_key is None or side_b_key is None:
            raise ValueError("Domain split JSON must contain top-level keys starting with 'side_a' and 'side_b'.")

        def flatten_domains(section: Dict[str, Any]) -> List[str]:
            domains: List[str] = []
            for v in section.values():
                if isinstance(v, list):
                    domains.extend(v)
            # Normalize: lowercase and trim
            normed = []
            for d in domains:
                if not isinstance(d, str):
                    continue
                dd = d.strip().lower()
                normed.append(dd)
            return normed

        side_a_domains = set(flatten_domains(domain_spec.get(side_a_key, {})))
        side_b_domains = set(flatten_domains(domain_spec.get(side_b_key, {})))

        def extract_domain_from_example(example: Dict[str, Any]) -> str:
            # Use the 'website' field directly for exact matching against JSON lists.
            site = example.get("website", "") or ""
            return site.strip().lower()

        included_train = 0
        included_test = 0
        excluded = 0
        for item in raw_examples:
            d = extract_domain_from_example(item)
            if d in side_a_domains:
                train_raw.append(item)
                included_train += 1
            elif d in side_b_domains:
                validation_raw.append(item)
                included_test += 1
            else:
                excluded += 1
                print(f"Excluded {d} not in mapping.")
        print(f"Domain split included {included_train} train, {included_test} test; excluded {excluded} not in mapping.")
    elif args.alt_preprocess:
        # Alternative preprocessing mode as requested by user.
        # 1) Keep only groups of size 3 or 13; ignore groups of size 1.
        # 2) Split groups 70/30 into train/val (entire groups, no within-group split).
        # 3) Train: include exactly 1 sample per group; remaining members listed in other_members.
        # 4) Val: include all samples; other members listed in other_members.

        # Determine train/val groups using a hash-based 70/30 split for determinism within a run.
        train_group_ids = set()
        val_group_ids = set()

        for group_id, pages in page_groups.items():
            n_pages = len(pages)
            if n_pages not in (3, 13):
                # Skip groups that are not of size 3 or 13.
                continue

            if hash(group_id) % 10 < 7:
                train_group_ids.add(group_id)
            else:
                val_group_ids.add(group_id)

        # Build raw example lists according to the rules above.
        for group_id in train_group_ids:
            pages = page_groups[group_id]
            if not pages:
                continue
            # Include only one example for training; keep others for other_members attachment later.
            train_raw.append(pages[0])

        for group_id in val_group_ids:
            pages = page_groups[group_id]
            validation_raw.extend(pages)
    elif args.train_all_webpages:
        # Use all webpages per group for training, mirroring validation behavior.
        # Keep only groups of size 3 or 13 and split groups 70/30 deterministically.
        train_group_ids = set()
        val_group_ids = set()

        for group_id, pages in page_groups.items():
            n_pages = len(pages)
            if n_pages not in (3, 13):
                continue
            if hash(group_id) % 10 < 7:
                train_group_ids.add(group_id)
            else:
                val_group_ids.add(group_id)

        for group_id in train_group_ids:
            pages = page_groups[group_id]
            train_raw.extend(pages)

        for group_id in val_group_ids:
            pages = page_groups[group_id]
            validation_raw.extend(pages)
    elif args.reuse_test_groups_dir:
        # Reuse test groups from an existing dataset's test.parquet.
        # Build training from all webpages in remaining groups, keeping only groups of size 3 or 13.
        reuse_dir = Path(os.path.expanduser(args.reuse_test_groups_dir))
        test_parquet_path = reuse_dir / "test.parquet"
        if not test_parquet_path.exists():
            raise FileNotFoundError(f"Could not find test.parquet at {test_parquet_path}")

        # Load the existing test parquet and collect its group ids
        reused = datasets.load_dataset("parquet", data_files={"test": str(test_parquet_path)})
        reused_test = reused["test"]
        reused_test_group_ids = set()
        for rec in reused_test:
            pid = rec.get("extra_info", {}).get("page_id", "")
            if pid:
                reused_test_group_ids.add(get_website_group(pid))

        # Now partition current raw examples by whether their group is in the reused test set
        for group_id, pages in page_groups.items():
            n_pages = len(pages)
            if n_pages not in (3, 13):
                continue
            if group_id in reused_test_group_ids:
                validation_raw.extend(pages)
            else:
                train_raw.extend(pages)
    elif args.pair_two_webpages:
        # Pairing mode: one example per train group (two webpages per example),
        # and n examples per val group by pairing consecutive pages with wrap-around.
        train_group_ids = set()
        val_group_ids = set()

        # Keep only groups of size 3 or 13 for consistency with other modes and ensure n>=2.
        for group_id, pages in page_groups.items():
            n_pages = len(pages)
            if n_pages not in (3, 13):
                continue
            if n_pages < 2:
                continue
            if hash(group_id) % 10 < 7:
                train_group_ids.add(group_id)
            else:
                val_group_ids.add(group_id)

        # Deterministically iterate groups
        for group_id in sorted(train_group_ids):
            pages = list(page_groups[group_id])
            # Sort for stable ordering before sampling indices
            pages.sort(key=lambda x: x.get("page_id", ""))
            if len(pages) < 2:
                continue
            # Sample two distinct indices deterministically based on global seed and group_id
            # Derive a per-group RNG to avoid cross-group dependency on ordering
            seed_material = f"{args.pair_random_seed}:{group_id}".encode("utf-8")
            stable_seed = int(hashlib.md5(seed_material).hexdigest(), 16) % (2**32)
            group_rng = random.Random(stable_seed)
            idxs = list(range(len(pages)))
            group_rng.shuffle(idxs)
            a_idx, b_idx = sorted(idxs[:2])
            train_pairs.append((pages[a_idx], pages[b_idx]))

        for group_id in sorted(val_group_ids):
            pages = list(page_groups[group_id])
            pages.sort(key=lambda x: x.get("page_id", ""))
            n = len(pages)
            if n < 2:
                continue
            # Create n pairs with wrap-around: (i, i+1 mod n)
            for i in range(n):
                a = pages[i]
                b = pages[(i + 1) % n]
                val_pairs.append((a, b))
    else:
        # Original splitting heuristics (kept for backward compatibility).
        for group_id, pages in page_groups.items():
            n_pages = len(pages)

            if n_pages == 1:
                # 70% chance of going to train
                if hash(group_id) % 10 < 7:
                    train_raw.append(pages[0])
                else:
                    validation_raw.append(pages[0])
            elif n_pages == 3:
                # 2 for train, 1 for validation
                train_raw.extend(pages[:2])
                validation_raw.append(pages[2])
            elif n_pages == 13:
                # ~80% for train, ~20% for validation
                split_point = int(n_pages * 0.8)
                train_raw.extend(pages[:split_point])
                validation_raw.extend(pages[split_point:])
            else:
                raise ValueError(f"Unexpected number of pages: {n_pages}")

    # -----------------------------------------------------------------------
    # Instantiate the HTML compressor.
    # -----------------------------------------------------------------------
    html_compressor = HTMLCompressor()

    # -----------------------------------------------------------------------
    # Helper to transform a raw example into the final schema.
    # -----------------------------------------------------------------------
    def transform(example: Dict[str, Any], split: str, idx: int) -> Dict[str, Any]:
        # -------------------------------------------------------------------
        # Extract fields & basic validation.
        # -------------------------------------------------------------------
        html_content = example.get("html_content")
        if not html_content:
            raise ValueError(f"HTML content missing for page_id={example.get('page_id')}")

        compressed_html = html_compressor.compress_html(html_content)

        # Parse triples_annotation → gold_triple.
        gold_triple = example.get("triples_annotation", [])
        if isinstance(gold_triple, str):
            try:
                gold_triple = json.loads(gold_triple)
            except json.JSONDecodeError:
                gold_triple = []

        # Ensure list of list[str] with length≥3.
        processed_triples: List[List[str]] = []
        if isinstance(gold_triple, list):
            for triple in gold_triple:
                if isinstance(triple, list) and len(triple) >= 3:
                    processed_triples.append([str(x) for x in triple[:3]])

        return {
            "data_source": DATA_SOURCE,
            "prompt": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": compressed_html},
            ],
            "ability": "triple_extraction",
            "reward_model": {"style": "rule", "ground_truth": processed_triples},
            "extra_info": {
                "split": split,
                "index": idx,
                "page_id": example.get("page_id", ""),
                "page_url": example.get("page_url", ""),
                "website": example.get("website", ""),
                "html_content": html_content,
            },
        }

    def extract_processed_triples(example: Dict[str, Any]) -> List[List[str]]:
        raw_triples = example.get("triples_annotation", [])
        if isinstance(raw_triples, str):
            try:
                raw_triples = json.loads(raw_triples)
            except json.JSONDecodeError:
                raw_triples = []

        processed: List[List[str]] = []
        if isinstance(raw_triples, list):
            for triple in raw_triples:
                if isinstance(triple, list) and len(triple) >= 3:
                    processed.append([str(x) for x in triple[:3]])
        return processed

    def transform_pair(example_a: Dict[str, Any], example_b: Dict[str, Any], split: str, idx: int) -> Dict[str, Any]:
        html_a = example_a.get("html_content")
        html_b = example_b.get("html_content")
        if not html_a or not html_b:
            raise ValueError(
                f"HTML content missing for pages: A={example_a.get('page_id')} B={example_b.get('page_id')}"
            )

        comp_a = html_compressor.compress_html(html_a)
        comp_b = html_compressor.compress_html(html_b)
        combined_user_content = (
            "<!-- PAGE_A_START -->\n" + comp_a + "\n<!-- PAGE_A_END -->\n\n" +
            "<!-- PAGE_B_START -->\n" + comp_b + "\n<!-- PAGE_B_END -->"
        )

        gt_a = extract_processed_triples(example_a)
        gt_b = extract_processed_triples(example_b)
        combined_gt = list(gt_a) + list(gt_b)

        page_id_a = example_a.get("page_id", "")
        page_id_b = example_b.get("page_id", "")

        return {
            "data_source": DATA_SOURCE,
            "prompt": [
                {"role": "system", "content": SYSTEM_INSTRUCTION + " You will be shown two example webpages in the user message (Page A and Page B), separated by explicit markers. Use both examples to design generalizable logic. Your script should work when applied to any single webpage."},
                {"role": "user", "content": combined_user_content},
            ],
            "ability": "triple_extraction",
            "reward_model": {"style": "rule", "ground_truth": combined_gt},
            "extra_info": {
                "split": split,
                "index": idx,
                # Keep page_id for group counting compatibility; use first page
                "page_id": page_id_a,
                "page_id_a": page_id_a,
                "page_id_b": page_id_b,
                "page_url_a": example_a.get("page_url", ""),
                "page_url_b": example_b.get("page_url", ""),
                "website_a": example_a.get("website", ""),
                "website_b": example_b.get("website", ""),
                # Store combined and individual html contents
                "html_content": f"<!-- PAGE_A_START -->\n{html_a}\n<!-- PAGE_A_END -->\n\n<!-- PAGE_B_START -->\n{html_b}\n<!-- PAGE_B_END -->",
                "html_content_a": html_a,
                "html_content_b": html_b,
                "paired": True,
            },
        }

    # -----------------------------------------------------------------------
    # Transform and build Dataset objects.
    # -----------------------------------------------------------------------
    if args.pair_two_webpages:
        train_records = [
            transform_pair(ex_a, ex_b, "train", i) for i, (ex_a, ex_b) in enumerate(train_pairs)
        ]
        validation_records = [
            transform_pair(ex_a, ex_b, "test", i) for i, (ex_a, ex_b) in enumerate(val_pairs)
        ]
    else:
        train_records = [transform(ex, "train", i) for i, ex in enumerate(train_raw)]
        validation_records = [transform(ex, "test", i) for i, ex in enumerate(validation_raw)]

    # -----------------------------------------------------------------------
    # Attach information about other members in the same website group.
    # -----------------------------------------------------------------------

    def attach_other_members(records: List[Dict[str, Any]]) -> None:
        """Populate each record's extra_info with details of other pages in the
        same website group that belong to the same split.

        The information is stored under the key `other_members` and contains a
        list of dictionaries with the following structure::

            {
                "page_id": str,
                "html_content": str,
                "ground_truth": List[List[str]],
            }
        """
        # Build a mapping from group_id to indices of records belonging to that group.
        group_to_indices: Dict[str, List[int]] = {}
        for idx, rec in enumerate(records):
            group_id = get_website_group(rec["extra_info"]["page_id"])
            group_to_indices.setdefault(group_id, []).append(idx)

        # Iterate over each record and attach info about its peers within the group.
        for idx, rec in enumerate(records):
            group_id = get_website_group(rec["extra_info"]["page_id"])
            peer_info: List[Dict[str, Any]] = []
            for peer_idx in group_to_indices[group_id]:
                if peer_idx == idx:
                    continue  # Skip self
                peer_record = records[peer_idx]
                peer_info.append(
                    {
                        "page_id": peer_record["extra_info"]["page_id"],
                        "html_content": peer_record["extra_info"]["html_content"],
                        "ground_truth": peer_record["reward_model"]["ground_truth"],
                    }
                )
            # Store the list (may be empty) in the record's extra_info.
            rec["extra_info"]["other_members"] = peer_info
            print("Attached {} other members to {}".format(len(peer_info), rec["extra_info"]["page_id"]))

    if args.alt_preprocess:
        # In alternative mode, other_members should include peers from the same
        # website group even if they were not included as separate records (e.g.,
        # train split keeps only one record per group).

        def attach_other_members_from_page_groups(records: List[Dict[str, Any]]) -> None:
            for rec in records:
                rec_page_id = rec["extra_info"]["page_id"]
                group_id = get_website_group(rec_page_id)
                peers = []
                for ex in page_groups.get(group_id, []):
                    if ex.get("page_id") == rec_page_id:
                        continue
                    peers.append(
                        {
                            "page_id": ex.get("page_id", ""),
                            "html_content": ex.get("html_content", ""),
                            "ground_truth": extract_processed_triples(ex),
                        }
                    )
                rec["extra_info"]["other_members"] = peers
                print(
                    "Attached {} other members to {}".format(
                        len(peers), rec["extra_info"]["page_id"]
                    )
                )

        attach_other_members_from_page_groups(train_records)
        attach_other_members_from_page_groups(validation_records)
    elif args.train_all_webpages or bool(args.reuse_test_groups_dir):
        # All members of each selected group are present; attach peers from within the split.
        attach_other_members(train_records)
        attach_other_members(validation_records)
    elif args.pair_two_webpages:
        # In pairing mode, attach other_members as all pages from the same group
        # excluding the two webpages selected in the pair.
        def attach_other_members_for_pairs(records: List[Dict[str, Any]]) -> None:
            for rec in records:
                page_id_a = rec["extra_info"].get("page_id_a", rec["extra_info"].get("page_id", ""))
                page_id_b = rec["extra_info"].get("page_id_b", "")
                group_id = get_website_group(page_id_a)
                peers: List[Dict[str, Any]] = []
                for ex in page_groups.get(group_id, []):
                    pid = ex.get("page_id", "")
                    if pid == page_id_a or pid == page_id_b:
                        continue
                    peers.append(
                        {
                            "page_id": pid,
                            "html_content": ex.get("html_content", ""),
                            "ground_truth": extract_processed_triples(ex),
                        }
                    )
                rec["extra_info"]["other_members"] = peers
                print(
                    "Attached {} other members to pair ({}, {})".format(
                        len(peers), page_id_a, page_id_b
                    )
                )

        attach_other_members_for_pairs(train_records)
        attach_other_members_for_pairs(validation_records)
    else:
        # Populate the other_members field for both splits using the original logic.
        attach_other_members(train_records)
        attach_other_members(validation_records)

    # -----------------------------------------------------------------------
    # Transform and build Dataset objects.
    # -----------------------------------------------------------------------
    train_dataset = datasets.Dataset.from_list(train_records)
    val_dataset = datasets.Dataset.from_list(validation_records)

    # -----------------------------------------------------------------------
    # Persist to Parquet.
    # -----------------------------------------------------------------------
    local_dir = Path(os.path.expanduser(args.local_dir))
    local_dir.mkdir(parents=True, exist_ok=True)

    train_dataset.to_parquet(str(local_dir / "train.parquet"))
    val_dataset.to_parquet(str(local_dir / "test.parquet"))

    # -----------------------------------------------------------------------
    # Print summary statistics
    # -----------------------------------------------------------------------
    def count_unique_groups(records: List[Dict[str, Any]]) -> int:
        group_ids = set()
        for rec in records:
            group_ids.add(get_website_group(rec["extra_info"]["page_id"]))
        return len(group_ids)

    train_num_examples = len(train_records)
    val_num_examples = len(validation_records)
    train_num_groups = count_unique_groups(train_records)
    val_num_groups = count_unique_groups(validation_records)

    print(
        f"Train: {train_num_examples} examples across {train_num_groups} groups | "
        f"Validation: {val_num_examples} examples across {val_num_groups} groups"
    )