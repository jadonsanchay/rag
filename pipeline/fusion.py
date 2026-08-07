from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

# Conventional RRF constant. Larger values flatten the contribution of top
# ranks, making the fusion less sensitive to any single list's ordering.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    weights: Optional[Sequence[float]] = None,
    k: int = DEFAULT_RRF_K,
) -> List[Tuple[str, float, Dict[int, int]]]:
    """Merge ranked id lists by position rather than score.

    Cosine similarity and BM25 are on incomparable scales, and normalising them
    requires assumptions that break whenever the corpus changes. RRF sidesteps
    that entirely by using only rank: each list contributes weight/(k + rank).

    Returns (id, fused_score, {list_index: rank}) sorted best-first, where the
    rank map is what a retrieval-trace UI needs to explain *why* something ranked.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match the number of ranked lists")

    scores: Dict[str, float] = defaultdict(float)
    ranks: Dict[str, Dict[int, int]] = defaultdict(dict)

    for list_index, (ids, weight) in enumerate(zip(ranked_lists, weights)):
        for rank, item_id in enumerate(ids, start=1):
            # Keep the best rank if an id somehow repeats within one list.
            if list_index in ranks[item_id]:
                continue
            scores[item_id] += weight / (k + rank)
            ranks[item_id][list_index] = rank

    fused = [(item_id, score, ranks[item_id]) for item_id, score in scores.items()]
    # Tie-break on best single rank so ordering is deterministic.
    fused.sort(key=lambda row: (-row[1], min(row[2].values())))
    return fused
