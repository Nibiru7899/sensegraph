"""Multi-label metrics with tied-score AP and chart-cluster paired intervals."""
import numpy as np

from .synthetic import RELATIONS


def average_precision(labels, scores):
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores)
    positives = labels.sum()
    if not positives:
        return None
    order = np.argsort(-scores, kind="stable")
    y, p = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(p[:-1] != p[1:]), len(p) - 1]
    tp = np.cumsum(y)[ends]
    recall = tp / positives
    precision = tp / (ends + 1)
    return float(np.sum(np.diff(np.r_[0., recall]) * precision))


def confusion(labels, scores, thresholds):
    y = np.asarray(labels, dtype=bool)
    predicted = np.asarray(scores) >= np.asarray(thresholds)
    return np.stack(((predicted & y).sum(0), (predicted & ~y).sum(0), (~predicted & y).sum(0), (~predicted & ~y).sum(0)), axis=1)


def f1_from_counts(counts):
    counts = np.asarray(counts)
    tp, fp, fn = counts[:, 0], counts[:, 1], counts[:, 2]
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(tp), dtype=float), where=(2 * tp + fp + fn) != 0)
    micro = float(2 * tp.sum() / max(1, 2 * tp.sum() + fp.sum() + fn.sum()))
    return float(f1.mean()), micro


def metrics(labels, scores, thresholds):
    labels, scores = np.asarray(labels), np.asarray(scores)
    counts = confusion(labels, scores, thresholds)
    macro, micro = f1_from_counts(counts)
    classes = {}
    aps = []
    for i, name in enumerate(RELATIONS):
        tp, fp, fn, tn = [int(x) for x in counts[i]]
        ap = average_precision(labels[:, i], scores[:, i])
        if ap is not None:
            aps.append(ap)
        classes[name] = {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                         "f1": 2 * tp / max(1, 2 * tp + fp + fn), "ap": ap,
                         "tp": tp, "fp": fp, "fn": fn, "tn": tn, "positives": tp + fn, "negatives": fp + tn,
                         "threshold": float(thresholds[i])}
    return {"macro_f1": macro, "micro_f1": micro, "macro_ap": float(np.mean(aps)) if aps else None,
            "ap_defined_classes": len(aps), "per_class": classes, "edge_candidates": len(labels)}


def calibrate(labels, scores, grid):
    labels, scores = np.asarray(labels), np.asarray(scores)
    chosen = []
    for i in range(labels.shape[1]):
        if not labels[:, i].sum():
            chosen.append(0.5)
            continue
        ranked = []
        for threshold in grid:
            count = confusion(labels[:, i:i + 1], scores[:, i:i + 1], [threshold])
            ranked.append((f1_from_counts(count)[0], threshold))
        chosen.append(max(ranked)[1])
    return chosen


def paired_bootstrap(gat_counts, mlp_counts, samples=2000, seed=20261103):
    """Inputs [seed, chart, class, TP/FP/FN/TN]; resample paired chart clusters."""
    a, b = np.asarray(gat_counts), np.asarray(mlp_counts)
    if a.shape != b.shape or a.ndim != 4:
        raise ValueError("Paired bootstrap requires matching seed/chart/class counts")
    rng = np.random.default_rng(seed)
    point = np.mean([np.subtract(f1_from_counts(x.sum(0)), f1_from_counts(y.sum(0))) for x, y in zip(a, b)], axis=0)
    differences = []
    for _ in range(samples):
        indices = rng.integers(0, a.shape[1], size=a.shape[1])
        differences.append(np.mean([np.subtract(f1_from_counts(x[indices].sum(0)), f1_from_counts(y[indices].sum(0))) for x, y in zip(a, b)], axis=0))
    intervals = np.quantile(differences, [0.025, 0.975], axis=0)
    return {name: {"gat_minus_mlp": float(point[i]), "ci95": intervals[:, i].tolist()} for i, name in enumerate(("macro_f1", "micro_f1"))}
