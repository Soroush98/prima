"""Anomaly-detection metrics shared by the benchmarks.

We deliberately report a *strict* and a *lenient* view of every detector, per
the mle-practices rule "always pair an oracle/optimistic metric with a strict
one and show the gap":

  * AUC-PR (strict)              — point-wise ranking quality. Its no-skill floor
    is the positive prevalence, NOT 0.5, so judge it against the anomaly rate.
  * point-adjusted best-F1       — the OmniAnomaly paper's headline metric
    (§ point-adjust): a ground-truth anomaly *segment* counts as fully detected if
    ANY point in it is alerted. Oracle threshold + segment credit ⇒ optimistic.
  * point-adjusted F1 @ threshold — the same segment credit but at a *fixed*
    (deployed) threshold, so it reflects what actually ships, not an oracle.
"""
from __future__ import annotations

import numpy as np


def auc_pr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Strict point-wise area under the precision-recall curve."""
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    total_pos = labels.sum()
    if total_pos == 0:
        return float("nan")
    precision = tp / (tp + fp)
    recall = tp / total_pos
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum((recall[1:] - recall[:-1]) * (precision[1:] + precision[:-1]) / 2))


def _segments(labels: np.ndarray) -> list[tuple[int, int]]:
    n = len(labels)
    segs, i = [], 0
    while i < n:
        if labels[i] == 1:
            j = i
            while j + 1 < n and labels[j + 1] == 1:
                j += 1
            segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs


def best_f1_adjusted(scores: np.ndarray, labels: np.ndarray) -> float:
    """Point-adjusted best F-score (papers' §point-adjust). A ground-truth anomaly
    segment counts as fully detected if ANY point in it is alerted; points outside
    segments are scored as usual. Sweep all thresholds, take best F1. Best F1 can
    only occur at a segment-max threshold (lowering it between segment maxes only
    adds false positives), so we enumerate those."""
    segs = _segments(labels)
    P = sum(e - s + 1 for s, e in segs)
    if P == 0:
        return float("nan")
    seg_max = np.array([scores[s : e + 1].max() for s, e in segs])
    seg_len = np.array([e - s + 1 for s, e in segs], dtype=np.float64)
    normal = np.sort(scores[labels == 0])            # ascending
    order = np.argsort(-seg_max)
    sm, tp_cum = seg_max[order], np.cumsum(seg_len[order])
    best = 0.0
    for k in range(len(sm)):
        tp = tp_cum[k]
        fp = len(normal) - np.searchsorted(normal, sm[k], side="left")
        f1 = 2 * tp / (2 * tp + fp + (P - tp))       # FN = P - tp
        best = max(best, float(f1))
    return best


def adjusted_f1_at(scores: np.ndarray, labels: np.ndarray, thr: float) -> float:
    """Point-adjusted F1 at a *fixed* threshold — the deployed-cutoff view."""
    pred = scores > thr
    adj = pred.copy()
    for s, e in _segments(labels):
        if pred[s : e + 1].any():
            adj[s : e + 1] = True
    tp = int((adj & (labels == 1)).sum())
    fp = int((adj & (labels == 0)).sum())
    fn = int((~adj & (labels == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0
