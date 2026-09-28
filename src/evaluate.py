"""
evaluate.py — Evaluation metrics for entity resolution.

Implements:
- F0.5 macro-averaged per S1 entity (exact competition scoring)
- Blocking recall measurement
- Blocking statistics
"""


def compute_f05_per_entity(true_set: set, pred_set: set) -> float:
    """
    Compute F0.5 for a single S1 entity.
    
    Rules (from competition):
    - Singleton with correct empty prediction → 1.0
    - Singleton with any false match → 0.0
    - Non-singleton with empty prediction → 0.0
    """
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0   # correct singleton
    if len(true_set) == 0 and len(pred_set) > 0:
        return 0.0   # false merge on singleton
    if len(pred_set) == 0:
        return 0.0   # missed all matches
    
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    
    if precision + recall == 0:
        return 0.0
    
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def macro_f05(predictions: dict, ground_truth: dict) -> float:
    """
    Compute macro-averaged F0.5 across all S1 entities.
    
    Args:
        predictions: {s1_id: set(predicted_match_ids)}
        ground_truth: {s1_id: set(true_match_ids)}  — empty set for singletons
    
    Returns:
        Macro-averaged F0.5 score.
    """
    if not ground_truth:
        return 0.0
    
    total = 0.0
    for s1_id in ground_truth:
        true = ground_truth[s1_id]
        pred = predictions.get(s1_id, set())
        total += compute_f05_per_entity(true, pred)
    
    return total / len(ground_truth)


def blocking_recall(candidates: dict, ground_truth: dict) -> float:
    """
    Measure blocking recall: what fraction of true matches appear in candidates.
    
    Only counts non-singleton entities (singletons have no matches to recall).
    
    Args:
        candidates: {s1_id: set(candidate_ids)}
        ground_truth: {s1_id: set(true_match_ids)}
    
    Returns:
        Average per-entity recall across non-singleton entities.
    """
    recalls = []
    for s1_id, true_matches in ground_truth.items():
        if not true_matches:  # singleton — skip
            continue
        cands = candidates.get(s1_id, set())
        recall = len(true_matches & cands) / len(true_matches)
        recalls.append(recall)
    
    return sum(recalls) / len(recalls) if recalls else 0.0


def blocking_stats(candidates: dict) -> dict:
    """
    Compute statistics about the blocking candidate set.
    
    Returns dict with: total_candidates, avg_per_entity, median, max, 
    pct_empty (entities with zero candidates).
    """
    counts = [len(cands) for cands in candidates.values()]
    if not counts:
        return {"total": 0, "avg": 0, "median": 0, "max": 0, "pct_empty": 0}
    
    counts_sorted = sorted(counts)
    n = len(counts_sorted)
    
    return {
        "total_candidates": sum(counts),
        "num_entities": n,
        "avg_per_entity": sum(counts) / n,
        "median_per_entity": counts_sorted[n // 2],
        "max_per_entity": counts_sorted[-1],
        "pct_empty": sum(1 for c in counts if c == 0) / n * 100,
    }


def load_ground_truth(gt_path: str) -> dict:
    """
    Load ground truth TSV into {s1_id: set(matched_ids)}.
    """
    import csv
    gt = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            s1_id = row[0]
            if len(row) < 2 or not row[1].strip():
                gt[s1_id] = set()
            else:
                gt[s1_id] = set(row[1].split(","))
    return gt


def evaluate_predictions_from_file(matching_path: str, gt_path: str) -> float:
    """
    Load matching_results.tsv and ground truth, compute F0.5.
    """
    import csv
    
    gt = load_ground_truth(gt_path)
    
    preds = {}
    with open(matching_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id = row[0]
            if len(row) < 2 or not row[1].strip():
                preds[s1_id] = set()
            else:
                preds[s1_id] = set(row[1].split(","))
    
    return macro_f05(preds, gt)
