# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Triple Accuracy Evaluator

A refactored evaluation module that combines LLM-based and fuzzy matching approaches
for evaluating triple extraction accuracy. Designed for single question evaluation.
"""

import json

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from fuzzywuzzy import fuzz
from munkres import Munkres

from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TripleAccuracyEvaluator:
    """
    Evaluates triple extraction accuracy using multiple metrics including
    fuzzy matching, exact matching, and optional LLM-based semantic comparison.
    """

    def __init__(
        self,
        use_llm: bool = True,
        model_name: str = "gpt-4o",
        cache_file: Optional[str] = None,
        prompt_dir: Optional[str] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            use_llm: Whether to use LLM for semantic comparison
            model_name: LLM model name to use
            cache_file: Path to cache file for LLM responses
            prompt_dir: Directory containing prompt templates
        """
        self.use_llm = use_llm
        self.fuzzy_cache = {}
        self.llm_cache = {}
        self.model_name = model_name
        self.cache_file = cache_file
        # Set default prompt directory
        if prompt_dir is None:
            self.prompt_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts"
            )
        else:
            self.prompt_dir = prompt_dir

        # Load existing cache if available
        if cache_file:
            self._load_llm_cache()

    def _load_llm_cache(self):
        """Load LLM cache from file."""
        try:
            with open(self.cache_file, "r", encoding="utf8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        for key, value in data.items():
                            self.llm_cache[key] = value
        except FileNotFoundError:
            pass

    def _save_to_cache(self, key: str, value: str):
        """Save LLM response to cache file."""
        if self.cache_file:
            with open(self.cache_file, "a", encoding="utf8") as f:
                f.write(json.dumps({key: value}) + "\n")

    def _get_llm_response(self, tx: str, ty: str) -> str:
        """Get LLM response with caching."""
        from llm_modules import process_prompt_with_llm
        cache_key = f"{tx}$$$${ty}"

        if cache_key in self.llm_cache:
            return self.llm_cache[cache_key]

        # Use the process_prompt_with_llm function
        variables = {"tx": tx, "ty": ty}

        response = process_prompt_with_llm(
            prompt_file="triple_comparison.prompt",
            prompt_dir=self.prompt_dir,
            variables=variables,
            model=self.model_name,
            temperature=0.0,
        )

        self.llm_cache[cache_key] = response
        self._save_to_cache(cache_key, response)

        return response

    def _llm_semantic_match(self, triple1: List[str], triple2: List[str]) -> float:
        """
        Use LLM to determine if two triples are semantically equivalent.

        Args:
            triple1: First triple as 3-element list [subject, predicate, object]
            triple2: Second triple as 3-element list [subject, predicate, object]

        Returns:
            1.0 if semantically equivalent, 0.0 otherwise
        """
        if not self.use_llm:
            return 0.0

        # Format triples for LLM
        t1 = "(" + ", ".join(triple1) + ")"
        t2 = "(" + ", ".join(triple2) + ")"

        # Clean up formatting
        if t1.startswith("(("):
            t1 = t1[1:]
        if t1.endswith("))"):
            t1 = t1[:-1]
        if t2.startswith("(("):
            t2 = t2[1:]
        if t2.endswith("))"):
            t2 = t2[:-1]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._get_llm_response(t1, t2)
                response_lower = response.lower().strip()

                if response_lower == "yes" or "yes" in response_lower:
                    return 1.0
                else:
                    return 0.0

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    print(f"LLM call failed after {max_retries} attempts: {e}")
                    return 0.0

        return 0.0

    def _get_fuzzy_match(self, text1: str, text2: str) -> float:
        """Get fuzzy match score with caching."""
        if text1 in self.fuzzy_cache and text2 in self.fuzzy_cache[text1]:
            return self.fuzzy_cache[text1][text2]

        if text1 not in self.fuzzy_cache:
            self.fuzzy_cache[text1] = {}

        score = fuzz.ratio(text1, text2) / 100.0
        self.fuzzy_cache[text1][text2] = score
        return score

    def _clean_text(self, text: str) -> str:
        """Clean text by removing HTML comments and non-alphanumeric characters."""
        # Remove HTML comments
        text = re.sub(r"<--.*?-->", "", text)
        # Remove non-alphanumeric characters and join
        text = "".join(re.sub(r"[^\w\s]", "", text).split())
        return text.lower()

    def _maximum_weight_matching_fuzzy(
        self,
        gt_triples: List[List[str]],
        pred_triples: List[List[str]],
        use_greedy: bool = False,
        use_scipy: Optional[bool] = True,
        top_k_edges: Optional[int] = None,
        deadline: Optional[float] = None,
    ) -> Tuple[float, float, float, List[Tuple[List[str], List[str]]]]:
        """
        Perform maximum weight bipartite matching using fuzzy matching.

        Args:
            gt_triples: List of ground truth triples (each as 3-element list)
            pred_triples: List of predicted triples (each as 3-element list)
            top_k_edges: If provided, for each vertex only the best `top_k_edges` edges
                (ranked by fuzzy score) are retained in the bipartite graph. A value
                of ``None`` keeps the full graph (default behaviour).
            use_greedy: If True, use greedy matching instead of optimal Hungarian algorithm
            use_scipy: If True, use scipy's linear_sum_assignment (rectangular matrix).
                If None, defaults to True when scipy is available, False otherwise.

        Returns:
            Tuple of (precision, recall, f1_score, matched_pairs) based on the (possibly
            pruned) bipartite graph.
        """
        fc = lambda x: re.sub(r"<--.*?-->", "", x)
        fs = lambda x: "".join(re.sub(r"[^\w\s]", "", x).split())

        a = []
        b = []
        amap = {}
        bmap = {}

        for i in range(len(gt_triples)):
            x = "\t".join(gt_triples[i])
            x = fc(fs(x.lower()))
            if x not in amap:
                amap[x] = i
                a += [x]

        for i in range(len(pred_triples)):
            x = "\t".join(pred_triples[i])
            x = fc(fs(x.lower()))
            if x not in bmap:
                bmap[x] = i
                b += [x]

        if len(a) == 0 or len(b) == 0:
            # No vertices on either side
            return 0.0, 0.0, 0.0, []

        # Helper to obtain or compute a fuzzy score
        def _score(i: int, j: int) -> float:
            return self._get_fuzzy_match(a[i], b[j])

        if use_greedy:
            # === Greedy matching algorithm ===
            # Create all possible pairs with their scores
            pairs_with_scores = []
            timed_out = False
            for i in range(len(a)):
                for j in range(len(b)):
                    score = _score(i, j)
                    pairs_with_scores.append((score, i, j))
                    if deadline is not None and time.perf_counter() > deadline:
                        timed_out = True
                        break
                if timed_out:
                    break
            
            # Sort by score in descending order (best matches first)
            pairs_with_scores.sort(reverse=True)
            
            # Greedily select non-conflicting pairs
            used_a = set()
            used_b = set()
            matched_pairs = []
            total_score = 0
            
            for score, i, j in pairs_with_scores:
                if i not in used_a and j not in used_b:
                    used_a.add(i)
                    used_b.add(j)
                    total_score += score
                    
                    # Collect matched pairs for LLM evaluation
                    gt_original = gt_triples[amap[a[i]]]
                    pred_original = pred_triples[bmap[b[j]]]
                    matched_pairs.append((gt_original, pred_original))
            # If timed out while building pairs, project remaining score using observed average
            if timed_out:
                observed_matches = len(matched_pairs)
                if observed_matches > 0:
                    avg_similarity = total_score / observed_matches
                    remaining_capacity = min(len(a) - len(used_a), len(b) - len(used_b))
                    projected_additional = avg_similarity * max(0, remaining_capacity)
                    total_score += projected_additional
        elif use_scipy:
            # === SciPy rectangular matrix implementation ===
            m, n = len(a), len(b)
            
            if top_k_edges is None or top_k_edges <= 0:
                # === Full rectangular cost matrix ===
                cost_matrix = []
                for i in range(m):
                    row = []
                    for j in range(n):
                        similarity = _score(i, j)
                        cost = 1.0 - similarity  # Convert similarity to cost
                        row.append(cost)
                    cost_matrix.append(row)
            else:
                # === Pruned rectangular matrix ===
                import numpy as np
                
                # Start with high-cost matrix (low similarity)
                cost_matrix = np.ones((m, n))
                
                # For every GT triple, keep top-K outgoing edges
                for i in range(m):
                    scores = [(_score(i, j), j) for j in range(n)]
                    scores.sort(reverse=True)
                    for similarity, j in scores[:top_k_edges]:
                        cost_matrix[i, j] = 1.0 - similarity
                
                # For every predicted triple, ensure its top-K edges are kept
                for j in range(n):
                    scores = [(_score(i, j), i) for i in range(m)]
                    scores.sort(reverse=True)
                    for similarity, i in scores[:top_k_edges]:
                        cost_matrix[i, j] = 1.0 - similarity
            
            # Solve rectangular assignment problem
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Calculate total score and collect matched pairs
            total_score = 0
            matched_pairs = []
            
            for i, j in zip(row_indices, col_indices):
                if i < len(a) and j < len(b):
                    similarity = 1.0 - cost_matrix[i][j]
                    total_score += similarity
                    
                    # Collect matched pairs for LLM evaluation
                    gt_original = gt_triples[amap[a[i]]]
                    pred_original = pred_triples[bmap[b[j]]]
                    matched_pairs.append((gt_original, pred_original))
        else:
            # === Original optimal matching algorithm ===
            weight = {}
            n = max(len(a), len(b))

            # --- Build the cost matrix with optional pruning ---
            cost: List[List[float]] = []

            if top_k_edges is None or top_k_edges <= 0:
                # === Original exhaustive computation ===
                for i in range(n):
                    row_cost: List[float] = []
                    for j in range(n):
                        if i >= len(a) or j >= len(b):
                            weight[(i, j)] = 0.0
                        else:
                            weight[(i, j)] = _score(i, j)
                        row_cost.append(1 - weight[(i, j)])
                    cost.append(row_cost)
            else:
                # === Pruned graph: keep only the best `top_k_edges` edges per vertex ===
                # Pre-initialise all weights to 0 so missing edges are treated as 0-weight.
                for i in range(n):
                    for j in range(n):
                        weight[(i, j)] = 0.0

                # For every vertex on the left (ground truth side), keep top-K outgoing edges.
                for i in range(len(a)):
                    scores = [(_score(i, j), j) for j in range(len(b))]
                    scores.sort(reverse=True)
                    for s, j in scores[: top_k_edges]:
                        weight[(i, j)] = s

                # For every vertex on the right (prediction side), also ensure its top-K edges are kept.
                for j in range(len(b)):
                    scores = [(_score(i, j), i) for i in range(len(a))]
                    scores.sort(reverse=True)
                    for s, i in scores[: top_k_edges]:
                        weight[(i, j)] = s

                # Build the cost matrix from the (sparse) weight dict.
                for i in range(n):
                    row_cost = []
                    for j in range(n):
                        row_cost.append(1 - weight[(i, j)])
                    cost.append(row_cost)

            m = Munkres()
            indexes = m.compute(cost)
            total_score = 0
            matched_pairs = []

            for row, column in indexes:
                total_score += weight[(row, column)]
                # Collect matched pairs for LLM evaluation
                if row < len(a) and column < len(b):
                    gt_original = gt_triples[amap[a[row]]]
                    pred_original = pred_triples[bmap[b[column]]]
                    matched_pairs.append((gt_original, pred_original))

        p = total_score / len(b) if len(b) > 0 else 1
        r = total_score / len(a) if len(a) > 0 else 0
        f = 2 * p * r / (p + r) if p * r != 0 else 0

        return p, r, f, matched_pairs

    def _maximum_weight_matching_llm(
        self,
        matched_pairs: List[Tuple[List[str], List[str]]],
        num_gold: int,
        num_prediction: int,
    ) -> Tuple[float, float, float]:
        """
        Evaluate matched pairs using LLM and calculate metrics.

        Args:
            matched_pairs: List of (ground_truth, predicted) triple pairs (each as 3-element lists)

        Returns:
            Tuple of (precision, recall, f1_score) based on LLM evaluation
        """
        if not matched_pairs:
            return 1.0, 0.0, 0.0

        # Parallelize LLM calls with max 20 workers
        with ThreadPoolExecutor(max_workers=20) as executor:
            # Submit all LLM tasks
            future_to_pair = {
                executor.submit(self._llm_semantic_match, gt_triple, pred_triple): (
                    gt_triple,
                    pred_triple,
                )
                for gt_triple, pred_triple in matched_pairs
            }

            # Collect results
            llm_total = 0
            for future in future_to_pair:
                llm_score = future.result()
                llm_total += llm_score

        p = llm_total / num_prediction if num_prediction > 0 else 1
        r = llm_total / num_gold if num_gold > 0 else 0
        f = 2 * p * r / (p + r) if p * r != 0 else 0

        return p, r, f

    def _calculate_soft_metrics(
        self, gt_triples: List[List[str]], pred_triples: List[List[str]]
    ) -> Tuple[float, float]:
        """
        Calculate soft precision and recall based on subject-object containment.

        Args:
            gt_triples: List of ground truth triples (each as 3-element list)
            pred_triples: List of predicted triples (each as 3-element list)

        Returns:
            Tuple of (soft_recall, soft_precision)
        """
        if not gt_triples and not pred_triples:
            return 1.0, 1.0
        if not gt_triples:
            return 1.0, 0.0
        if not pred_triples:
            return 0.0, 1.0

        # Clean triples
        gt_clean = [self._clean_text("\t".join(triple)) for triple in gt_triples]
        pred_clean = [self._clean_text("\t".join(triple)) for triple in pred_triples]

        # Calculate soft recall
        soft_recall_count = 0
        for i, gt_triple in enumerate(gt_clean):
            subject = self._clean_text(gt_triples[i][0])
            obj = self._clean_text(gt_triples[i][2])
            for pred_triple in pred_clean:
                if subject in pred_triple and obj in pred_triple:
                    soft_recall_count += 1
                    break

        # Calculate soft precision
        soft_precision_count = 0
        for i, pred_triple in enumerate(pred_clean):
            for j, gt_triple in enumerate(gt_clean):
                subject = self._clean_text(gt_triples[j][0])
                obj = self._clean_text(gt_triples[j][2])
                if subject in pred_triple and obj in pred_triple:
                    soft_precision_count += 1
                    break

        soft_recall = soft_recall_count / len(gt_clean) if gt_clean else 0.0
        soft_precision = soft_precision_count / len(pred_clean) if pred_clean else 1.0

        return soft_recall, soft_precision

    def evaluate_triples(
        self,
        ground_truth_triples: List[List[str]],
        predicted_triples: List[List[str]],
        type_check: bool = False,
        top_k_edges: Optional[int] = None,
        use_greedy: bool = False,
        use_scipy: Optional[bool] = None,
        max_eval_seconds: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Evaluate predicted triples against ground truth using multiple metrics.

        Args:
            ground_truth_triples: List of ground truth triples (each as 3-element list)
            predicted_triples: List of predicted triples (each as 3-element list)
            type_check: Whether to perform type checking on predicted_triples
            top_k_edges: If provided (positive integer), activates the *pruned* maximum
                matching mode described in the paper. In this mode, only the fuzzy
                precision/recall/F1 computed on the pruned graph are returned – all
                other metrics are skipped for efficiency.
            use_greedy: If True, use greedy matching instead of optimal Hungarian algorithm
            use_scipy: If True, use scipy's linear_sum_assignment (rectangular matrix).
                If None, defaults to True when scipy is available, False otherwise.

        Returns:
            Dictionary containing various accuracy metrics. If ``top_k_edges`` is
            supplied, only ``fuzzy_precision``, ``fuzzy_recall`` and ``fuzzy_f1`` are
            included; the other metrics will be ``None`` to signal they were not
            computed.
        """
        if type_check:
            # Type checking for predicted_triples
            if not isinstance(predicted_triples, list):
                return {
                    "exact_match": 0.0,
                    "fuzzy_edit_distance": 0.0,
                    "fuzzy_precision": 0.0,
                    "fuzzy_recall": 0.0,
                    "fuzzy_f1": 0.0,
                    "soft_recall": 0.0,
                    "soft_precision": 0.0,
                    "llm_precision": 0.0,
                    "llm_recall": 0.0,
                    "llm_f1": 0.0,
                }
            
            # Check if all elements are lists of strings
            for i, triple in enumerate(predicted_triples):
                if not(isinstance(triple, list) or isinstance(triple, tuple)):
                    return {
                        "exact_match": 0.0,
                        "fuzzy_edit_distance": 0.0,
                        "fuzzy_precision": 0.0,
                        "fuzzy_recall": 0.0,
                        "fuzzy_f1": 0.0,
                        "soft_recall": 0.0,
                        "soft_precision": 0.0,
                        "llm_precision": 0.0,
                        "llm_recall": 0.0,
                        "llm_f1": 0.0,
                    }
                for j, element in enumerate(triple):
                    if not isinstance(element, str):
                        return {
                            "exact_match": 0.0,
                            "fuzzy_edit_distance": 0.0,
                            "fuzzy_precision": 0.0,
                            "fuzzy_recall": 0.0,
                            "fuzzy_f1": 0.0,
                            "soft_recall": 0.0,
                            "soft_precision": 0.0,
                            "llm_precision": 0.0,
                            "llm_recall": 0.0,
                            "llm_f1": 0.0,
                        }
        
        results = {}

        # Fast-path: if we are using a pruned sub-graph then only compute the fuzzy-
        # based maximum matching and return.
        if top_k_edges is not None and top_k_edges > 0:
            deadline = None
            if max_eval_seconds is not None:
                deadline = time.perf_counter() + float(max_eval_seconds)
            fp, fr, ff, _ = self._maximum_weight_matching_fuzzy(
                ground_truth_triples, predicted_triples, top_k_edges=top_k_edges, use_greedy=use_greedy, use_scipy=use_scipy, deadline=deadline
            )

            # Return only the pruned F1 score under a dedicated key.
            return {
                "pruned_fuzzy_precision": fp,
                "pruned_fuzzy_recall": fr,
                "pruned_fuzzy_f1": ff,
            }

        # --- Standard evaluation pipeline (full graph) ---

        # Convert to strings for processing
        gt_text = "\n".join(
            ["\t".join(triple) for triple in ground_truth_triples]
        ).strip()
        pred_text = "\n".join(
            ["\t".join(triple) for triple in predicted_triples]
        ).strip()

        # Exact Match (set-based)
        gt_clean_set = set(
            self._clean_text("\t".join(triple)) for triple in ground_truth_triples
        )
        pred_clean_set = set(
            self._clean_text("\t".join(triple)) for triple in predicted_triples
        )

        if gt_clean_set or pred_clean_set:
            exact_match = len(gt_clean_set & pred_clean_set) / max(
                len(gt_clean_set), len(pred_clean_set)
            )
        else:
            exact_match = 1.0
        results["exact_match"] = exact_match

        # Fuzzy Edit Distance (overall text similarity)
        fuzzy_similarity = fuzz.ratio(
            self._clean_text(gt_text), self._clean_text(pred_text)
        )
        results["fuzzy_edit_distance"] = fuzzy_similarity

        # Step 1: Fuzzy Matching with Maximum Weight Bipartite Matching
        deadline = None
        if max_eval_seconds is not None:
            deadline = time.perf_counter() + float(max_eval_seconds)
        fuzzy_precision, fuzzy_recall, fuzzy_f1, matched_pairs = self._maximum_weight_matching_fuzzy(
            ground_truth_triples, predicted_triples, top_k_edges=None, use_greedy=use_greedy, use_scipy=use_scipy, deadline=deadline
        )
        results["fuzzy_precision"] = fuzzy_precision
        results["fuzzy_recall"] = fuzzy_recall
        results["fuzzy_f1"] = fuzzy_f1

        # Soft Metrics (subject-object containment)
        soft_recall, soft_precision = self._calculate_soft_metrics(
            ground_truth_triples, predicted_triples
        )
        results["soft_recall"] = soft_recall
        results["soft_precision"] = soft_precision

        # Step 2: LLM-based Semantic Matching on matched pairs (if enabled)
        if self.use_llm and matched_pairs:
            llm_precision, llm_recall, llm_f1 = self._maximum_weight_matching_llm(
                matched_pairs,
                len(ground_truth_triples),
                len(predicted_triples),
            )
            results["llm_precision"] = llm_precision
            results["llm_recall"] = llm_recall
            results["llm_f1"] = llm_f1
        else:
            results["llm_precision"] = None
            results["llm_recall"] = None
            results["llm_f1"] = None

        return results


def main(
    ground_truth_triples: List[List[str]],
    predicted_triples: List[List[str]],
    use_llm: bool = True,
    model_name: str = "llama3.3-70b-instruct",
    type_check: bool = False,
    top_k_edges: Optional[int] = None,
    use_greedy: bool = False,
    use_scipy: Optional[bool] = None,
    max_eval_seconds: Optional[float] = None,
) -> Dict[str, float]:
    """
    Main function to evaluate triple extraction accuracy.

    Args:
        ground_truth_triples: List of ground truth triples (each as 3-element list)
        predicted_triples: List of predicted triples (each as 3-element list)
        use_llm: Whether to use LLM for semantic comparison
        model_name: LLM model name to use
        type_check: Whether to perform type checking on predicted_triples
        top_k_edges: If provided, use pruned matching with only top-k edges
        use_greedy: If True, use greedy matching instead of optimal Hungarian algorithm
        use_scipy: If True, use scipy's linear_sum_assignment (rectangular matrix).
            If None, defaults to True when scipy is available, False otherwise.

    Returns:
        Dictionary containing various accuracy metrics
    """
    if model_name != "llama3.3-70b-instruct":
        print(f"Warning: Only llama3.3-70b-instruct is supported for now for triple evaluation.")
        raise ValueError("Only llama3.3-70b-instruct is supported for now")
    evaluator = TripleAccuracyEvaluator(use_llm=use_llm, model_name=model_name)

    # Run the evaluation
    return evaluator.evaluate_triples(
        ground_truth_triples,
        predicted_triples,
        type_check=type_check,
        top_k_edges=top_k_edges,
        use_greedy=use_greedy,
        use_scipy=use_scipy,
        max_eval_seconds=max_eval_seconds,
    )


# Example usage
if __name__ == "__main__":
    # Example triples (3-element lists: [subject, predicate, object])
    gt_triples = [
        ["Apple", "is a", "company"],
        ["iPhone", "manufactured by", "Apple"],
        ["Tim Cook", "CEO of", "Apple"],
    ]

    pred_triples = [
        ["Apple Inc", "is a", "tech company"],
        ["iPhone", "made by", "Apple"],
    ]

    # Evaluate without LLM using SciPy's optimized rectangular assignment
    print("=== SciPy Rectangular Assignment (Default) ===")
    use_llm = False
    results = main(gt_triples, pred_triples, use_llm=use_llm, use_scipy=True)

    print("Triple Evaluation Results:")
    print(f"Exact Match: {results['exact_match']:.3f}")
    print(f"Fuzzy Edit Distance: {results['fuzzy_edit_distance']:.3f}")
    print(f"Fuzzy Precision: {results['fuzzy_precision']:.3f}")
    print(f"Fuzzy Recall: {results['fuzzy_recall']:.3f}")
    print(f"Fuzzy F1: {results['fuzzy_f1']:.3f}")
    print(f"Soft Precision: {results['soft_precision']:.3f}")
    print(f"Soft Recall: {results['soft_recall']:.3f}")
    if use_llm:
        print(f"LLM Precision: {results['llm_precision']:.3f}")
        print(f"LLM Recall: {results['llm_recall']:.3f}")
        print(f"LLM F1: {results['llm_f1']:.3f}")

    # Evaluate without LLM using greedy algorithm
    print("\n=== Greedy Algorithm ===")
    results_greedy = main(gt_triples, pred_triples, use_llm=use_llm, use_greedy=True)

    print("Triple Evaluation Results (Greedy):")
    print(f"Exact Match: {results_greedy['exact_match']:.3f}")
    print(f"Fuzzy Edit Distance: {results_greedy['fuzzy_edit_distance']:.3f}")
    print(f"Fuzzy Precision: {results_greedy['fuzzy_precision']:.3f}")
    print(f"Fuzzy Recall: {results_greedy['fuzzy_recall']:.3f}")
    print(f"Fuzzy F1: {results_greedy['fuzzy_f1']:.3f}")
    print(f"Soft Precision: {results_greedy['soft_precision']:.3f}")
    print(f"Soft Recall: {results_greedy['soft_recall']:.3f}")
    if use_llm:
        print(f"LLM Precision: {results_greedy['llm_precision']:.3f}")
        print(f"LLM Recall: {results_greedy['llm_recall']:.3f}")
        print(f"LLM F1: {results_greedy['llm_f1']:.3f}")

    # Evaluate using legacy Munkres for comparison
    print("\n=== Legacy Munkres Algorithm ===")
    results_munkres = main(gt_triples, pred_triples, use_llm=use_llm, use_scipy=False, use_greedy=False)
    print("Triple Evaluation Results (Munkres):")
    print(f"Fuzzy F1: {results_munkres['fuzzy_f1']:.3f}")

    # Compare results
    print(f"\n=== Algorithm Comparison ===")
    print(f"Fuzzy F1 SciPy:   {results['fuzzy_f1']:.3f}")
    print(f"Fuzzy F1 Greedy:  {results_greedy['fuzzy_f1']:.3f}")
    print(f"Fuzzy F1 Munkres: {results_munkres['fuzzy_f1']:.3f}")
    
