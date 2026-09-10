"""
Saliency evaluation metrics.

WHY SO MANY, AND WHY ONE DECIDES
    These metrics disagree by construction. A map tuned to maximise KLD is not
    the map that maximises NSS - one rewards matching the whole distribution,
    another rewards putting mass exactly on fixation points, and a third only
    cares about ranking. Averaging them produces a number that means nothing.

    So the bake-off names ONE decision metric - NSS on our own annotated ad
    frames - and reports the rest as context. A model that wins on NSS and loses
    on SIM is telling us something worth reading, not failing.

CENTRE BIAS IS THE TRAP
    Human fixations cluster heavily in the middle of a frame, so a plain
    Gaussian blob at the centre scores surprisingly well on AUC and NSS. Without
    that control in the table, a model that has learned nothing but "look at the
    middle" appears to work. It is why `centre_prior` lives in here.

Definitions follow Bylinskii et al., "What do different evaluation metrics tell
us about saliency models?" - the standard reference for these.
"""
from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-12


def _normalise(saliency: np.ndarray) -> np.ndarray:
    total = float(saliency.sum())
    return saliency / total if total > 0 else saliency


def _standardise(saliency: np.ndarray) -> np.ndarray:
    std = float(saliency.std())
    return (saliency - float(saliency.mean())) / (std if std > 0 else 1.0)


def _match(saliency: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if saliency.shape[:2] == shape:
        return saliency.astype(np.float32)
    return cv2.resize(saliency.astype(np.float32), (shape[1], shape[0]),
                      interpolation=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------- metrics
def nss(saliency: np.ndarray, fixations: np.ndarray) -> float:
    """Normalised Scanpath Saliency. THE DECISION METRIC.

    Standardise the prediction, then average its value at the fixation points.
    Zero means "no better than chance"; higher means the model puts its mass
    where people actually look. It is the metric that most directly answers the
    question Vision Lab asks, and it is not fooled by a model that hedges by
    spreading attention everywhere - spreading lowers the standardised value at
    every point.
    """
    saliency = _match(saliency, fixations.shape[:2])
    points = fixations > 0
    if not points.any():
        return float("nan")
    return float(_standardise(saliency)[points].mean())


def cc(saliency: np.ndarray, ground_truth: np.ndarray) -> float:
    """Pearson correlation between the two distributions. Symmetric, so it
    punishes false positives and false negatives equally."""
    saliency = _match(saliency, ground_truth.shape[:2])
    a = saliency - saliency.mean()
    b = ground_truth - ground_truth.mean()
    denom = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def similarity(saliency: np.ndarray, ground_truth: np.ndarray) -> float:
    """Histogram intersection of two normalised maps, 0 to 1."""
    saliency = _normalise(_match(saliency, ground_truth.shape[:2]))
    return float(np.minimum(saliency, _normalise(ground_truth)).sum())


def kld(saliency: np.ndarray, ground_truth: np.ndarray) -> float:
    """Kullback-Leibler divergence, ground truth to prediction. LOWER IS BETTER.

    Heavily punishes a model for putting no mass where the ground truth has
    some - a miss costs far more than a false alarm, which is the opposite of
    how SIM behaves. That asymmetry is why both are reported.
    """
    saliency = _normalise(_match(saliency, ground_truth.shape[:2]))
    truth = _normalise(ground_truth)
    return float((truth * np.log(truth / (saliency + EPS) + EPS)).sum())


def auc_judd(saliency: np.ndarray, fixations: np.ndarray) -> float:
    """Area under the ROC curve, thresholding the map at each fixation value.

    0.5 is chance. Saturates easily - most reasonable models land between 0.8
    and 0.9 - which is exactly why it is context rather than the decision.
    """
    saliency = _match(saliency, fixations.shape[:2])
    lo, hi = float(saliency.min()), float(saliency.max())
    if hi <= lo:
        return 0.5
    saliency = (saliency - lo) / (hi - lo)

    points = fixations > 0
    if not points.any():
        return float("nan")

    at_fixations = np.sort(saliency[points])[::-1]
    flat = np.sort(saliency.ravel())[::-1]
    n_fix, n_all = len(at_fixations), len(flat)

    tp = np.zeros(n_fix + 2)
    fp = np.zeros(n_fix + 2)
    tp[-1], fp[-1] = 1.0, 1.0
    for index, threshold in enumerate(at_fixations):
        above = np.searchsorted(-flat, -threshold, side="right")
        tp[index + 1] = (index + 1) / n_fix
        fp[index + 1] = (above - index - 1) / (n_all - n_fix)
    return float(np.trapezoid(tp, fp))


def auc_shuffled(saliency: np.ndarray, fixations: np.ndarray,
                 other_fixations: np.ndarray) -> float:
    """AUC scored against fixations from OTHER images as the negative set.

    This is the centre-bias-corrected version. Because the negatives carry the
    same centre bias as the positives, a model that only knows "look at the
    middle" scores ~0.5 here while still scoring ~0.8 on plain AUC. It is the
    single most informative number for telling a real model from a centre prior.
    """
    saliency = _match(saliency, fixations.shape[:2])
    lo, hi = float(saliency.min()), float(saliency.max())
    if hi <= lo:
        return 0.5
    saliency = (saliency - lo) / (hi - lo)

    positives = saliency[fixations > 0]
    negatives = saliency[other_fixations > 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")

    thresholds = np.sort(positives)[::-1]
    tp, fp = [0.0], [0.0]
    for threshold in thresholds:
        tp.append((positives >= threshold).mean())
        fp.append((negatives >= threshold).mean())
    tp.append(1.0)
    fp.append(1.0)
    return float(np.trapezoid(tp, fp))


# --------------------------------------------------------------------------- controls
def centre_prior(height: int, width: int, sigma_fraction: float = 0.28) -> np.ndarray:
    """A Gaussian in the middle of the frame. NOT a candidate - the control.

    Whatever a model scores, this is what the same score is worth for free. Any
    candidate that does not clearly beat it has learned nothing we did not
    already know.
    """
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    sy, sx = height * sigma_fraction, width * sigma_fraction
    blob = np.exp(-(((xs - width / 2) ** 2) / (2 * sx ** 2)
                    + ((ys - height / 2) ** 2) / (2 * sy ** 2)))
    return _normalise(blob.astype(np.float32))


def evaluate(saliency: np.ndarray, ground_truth: np.ndarray,
             fixations: np.ndarray,
             other_fixations: np.ndarray | None = None) -> dict:
    """Every metric for one frame. NaN where a metric cannot be computed, so a
    missing value is visible rather than silently averaged as zero."""
    scores = {
        "NSS": nss(saliency, fixations),
        "CC": cc(saliency, ground_truth),
        "SIM": similarity(saliency, ground_truth),
        "KLD": kld(saliency, ground_truth),
        "AUC": auc_judd(saliency, fixations),
    }
    if other_fixations is not None:
        scores["sAUC"] = auc_shuffled(saliency, fixations, other_fixations)
    return scores
