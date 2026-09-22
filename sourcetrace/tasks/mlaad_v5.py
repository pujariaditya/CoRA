"""MLAAD v5 open-set source-tracing evaluation (standalone).

Faithful port of the metric flow in the original research grader
(``mlaad_klein.compute_all_metrics`` L838-1000, PRIMARY method-score path
L919-959; the ``train_and_score`` orchestration L738-818), with all
Research-grader coupling removed: no subprocess ``embed_worker`` /
``mlaad_klein_worker``, no GPU file-locks, no outlier-exposure/augmentation
side-channels, no composite scoring. The split, label space, seed and metric
formulas are unchanged, so results reproduce.

Pipeline
--------
1. Build the family-level split (:func:`sourcetrace.datasets.mlaad.build_split`).
2. Load row-aligned cached features per split
   (:func:`sourcetrace.tasks._features.stack_features`).
3. ``method.fit(train_features, gen_labels=class_id, lang_labels=lang_id, seed)``
   on the ID-only train rows.
4. ``method.score_openset(eval_X)`` on dev + eval -> a per-sample score where
   *higher = more known* (the method contract). The OOD-positive score is
   ``-score`` (higher = more OOD), exactly as the original negates the method
   output before the repo metrics (L921-925, L932).
5. Metrics (see "Which scores feed which metric" below).

Which scores feed which metric
------------------------------
The three headline numbers deliberately read from *different* score pools:

* ``fpr95`` (:func:`sourcetrace.metrics.protocol.fpr95_panda`, original L936-938 --
  the paper's 3.36 metric) is **calibrated on the DEV-OOD scores and tested on
  the EVAL-ID scores**. Dev-ID scores and eval-OOD scores never enter it. Two of
  the four pools are unused by design; that split is what keeps the threshold
  free of eval information.
* ``ood_eer`` (:func:`sourcetrace.metrics.protocol.ood_eer`, original ``ms_eer_eval``
  L933/L941) uses the **full eval split** -- every eval row, ID and OOD alike,
  contributes a ``(label=1 if OOD, score=-method_score)`` pair. Dev is not
  involved.
* ``id_acc`` uses the **ID eval rows only**: each utterance is attributed to the
  known model with the nearest trained anchor (``method.attribute``, the paper's
  rule ``argmin_c delta_c(z)``), and the accuracy is the fraction attributed to the
  correct model. The original grader measured a 24-way softmax-head accuracy
  instead, so ``id_acc`` is not comparable to a published softmax-head number;
  ``fpr95`` and ``ood_eer`` are byte-identical ports and are comparable.

Leak-safety: ``fit`` sees ONLY train ID features + labels. Dev is used only to
calibrate the OOD threshold; no eval labels enter training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import PROTOCOL
from ..datasets.mlaad import MlaadSplit, MlaadTable, build_split
from ..metrics.protocol import fpr95_panda, ood_eer
from ._features import FeatureSource, stack_features


@dataclass(frozen=True)
class MlaadV5Result:
    """MLAAD v5 evaluation outcome. All three metrics are percentages."""

    fpr95: float           # FPR@95, percent (compare to SOTA 3.36)
    ood_eer: float         # eval OOD-EER, percent (asymmetric Klein convention)
    id_acc: float          # known-model attribution accuracy, percent (nearest anchor)
    n_known_classes: int   # number of ID ("known") generator classes
    n_eval: int            # eval rows in total
    n_eval_id: int         # eval rows whose class is known (ID)
    n_eval_ood: int        # eval rows whose class is unseen (OOD)

    def as_dict(self) -> dict[str, float]:
        """The three headline metrics only, for JSON reporting (counts omitted)."""
        return {
            "fpr95": self.fpr95,
            "ood_eer": self.ood_eer,
            "id_acc": self.id_acc,
        }


def known_model_accuracy(method: Any, eval_X: np.ndarray, eval_cid: np.ndarray,
                         n_known: int) -> float:
    """Top-1 known-model attribution accuracy over the ID eval rows, in percent.

    The paper's attribution rule (Sec. 2.4): an utterance is attributed to the known
    model with the nearest trained anchor, ``argmin_c delta_c(z)``, computed by
    ``method.attribute``. Returns ``nan`` if the eval split has no ID rows.
    """
    id_mask = eval_cid < n_known
    if not np.any(id_mask):
        return float("nan")
    pred = np.asarray(method.attribute(eval_X[id_mask]))
    return float(np.mean(pred == eval_cid[id_mask]) * 100.0)


def fit_mlaad_v5(
    method: Any,
    features: FeatureSource,
    split: MlaadSplit | None = None,
    seed: int | None = None,
) -> Any:
    """Fit a source-tracing method on the MLAAD v5 *train* split (no eval).

    Train entry point for the split train/inference workflow. Fits ONLY on the
    ID-only train rows (leak-safe: dev/eval never enter ``fit``). Returns the same
    (now-fitted) ``method`` so it can be checkpointed and later handed to
    :func:`eval_mlaad_v5`.

    Args:
        method: a method implementing ``fit(features, gen_labels, lang_labels, seed)``.
        features: a :data:`FeatureSource` resolving each table's cached features.
        split: prebuilt :class:`MlaadSplit`; built via
            :func:`sourcetrace.datasets.mlaad.build_split` when ``None``.
        seed: fit/split seed; defaults to ``config.PROTOCOL.panda_seed`` (42).

    Returns:
        The fitted ``method``.
    """
    seed = PROTOCOL.panda_seed if seed is None else int(seed)
    split = split if split is not None else build_split(seed=seed)
    train: MlaadTable = split.train
    train_X = stack_features(train, features)
    method.fit(train_X, train.class_id, train.lang_id, seed)
    return method


def eval_mlaad_v5(
    method: Any,
    features: FeatureSource,
    split: MlaadSplit | None = None,
    seed: int | None = None,
) -> MlaadV5Result:
    """Evaluate a PRE-FIT source-tracing method on the MLAAD v5 protocol.

    Inference entry point: assumes ``method`` is already fitted (via
    :func:`fit_mlaad_v5` or :meth:`sourcetrace.method.Method.load`) and does NOT
    call ``fit``. Computes the identical metrics as :func:`evaluate_mlaad_v5`.

    ``fpr95`` calibrates on DEV-OOD scores and tests on EVAL-ID scores;
    ``ood_eer`` uses the whole eval split; ``id_acc`` uses the ID eval rows. See
    the module docstring for why those pools differ.

    Args:
        method: a fitted method implementing ``embed(X)``, ``score_openset(X)``
            (higher = more known) and ``attribute(X)`` (nearest-anchor class).
        features: a :data:`FeatureSource` resolving each table's cached features.
        split: prebuilt :class:`MlaadSplit`; built when ``None`` (same seed).
        seed: split seed; defaults to ``config.PROTOCOL.panda_seed`` (42).

    Returns:
        An :class:`MlaadV5Result` with ``fpr95`` / ``ood_eer`` / ``id_acc``.
    """
    seed = PROTOCOL.panda_seed if seed is None else int(seed)
    split = split if split is not None else build_split(seed=seed)

    dev: MlaadTable = split.dev
    ev: MlaadTable = split.eval
    n_known = split.n_known_classes

    dev_emb = np.asarray(method.embed(stack_features(dev, features)))
    eval_emb = np.asarray(method.embed(stack_features(ev, features)))

    # method score: higher = more known -> OOD-positive score is its negative.
    dev_ood_score = -np.asarray(method.score_openset(dev_emb), dtype=np.float64)
    eval_ood_score = -np.asarray(method.score_openset(eval_emb), dtype=np.float64)

    dev_is_ood = dev.class_id >= n_known
    eval_is_ood = ev.class_id >= n_known
    eval_is_id = ~eval_is_ood

    # -- FPR95 ----------------------------------------------------------------- #
    # Calibrate the threshold on DEV-OOD scores, test it on EVAL-ID scores.
    # Dev-ID and eval-OOD scores are deliberately NOT used by this metric.
    fpr95 = fpr95_panda(dev_ood_score[dev_is_ood],
                        eval_ood_score[eval_is_id]) * 100.0

    # -- OOD-EER --------------------------------------------------------------- #
    # Uses the FULL eval split (ID rows and OOD rows together), unlike FPR95.
    # Labels 1=OOD; score higher = more OOD. Asymmetric fpr[eer_index] convention.
    eer_frac, _ = ood_eer(eval_is_ood.astype(np.int64), eval_ood_score)

    # -- Known-model attribution accuracy (ID eval rows only) ------------------ #
    id_acc = known_model_accuracy(method, eval_emb, ev.class_id, n_known)

    return MlaadV5Result(
        fpr95=fpr95,
        ood_eer=eer_frac * 100.0,
        id_acc=id_acc,
        n_known_classes=n_known,
        n_eval=len(ev),
        n_eval_id=int(eval_is_id.sum()),
        n_eval_ood=int(eval_is_ood.sum()),
    )


def evaluate_mlaad_v5(
    method: Any,
    features: FeatureSource,
    split: MlaadSplit | None = None,
    seed: int | None = None,
) -> MlaadV5Result:
    """Fit + evaluate a source-tracing method on the MLAAD v5 protocol.

    Back-compatible combined entry point: fits on train then evaluates on dev/eval
    in one call. Equivalent to ``eval_mlaad_v5(fit_mlaad_v5(method, features,
    split, seed), features, split, seed)``. The metric math is unchanged.

    Args:
        method: a method object implementing the contract
            ``fit(features, gen_labels, lang_labels, seed)`` /
            ``embed(X) -> (N, emb_dim)`` (L2-normed) /
            ``score_openset(X) -> (N,)`` (higher = more known) /
            ``attribute(X) -> (N,)`` (nearest-anchor class).
        features: a :data:`FeatureSource` resolving each table's per-group cached
            ``.npy`` features (see :mod:`sourcetrace.tasks._features`).
        split: a prebuilt :class:`MlaadSplit`; built via
            :func:`sourcetrace.datasets.mlaad.build_split` when ``None``.
        seed: fit/split seed; defaults to ``config.PROTOCOL.panda_seed`` (42).

    Returns:
        An :class:`MlaadV5Result` with ``fpr95`` / ``ood_eer`` / ``id_acc`` (all
        percentages).
    """
    seed = PROTOCOL.panda_seed if seed is None else int(seed)
    split = split if split is not None else build_split(seed=seed)
    fit_mlaad_v5(method, features, split, seed)
    return eval_mlaad_v5(method, features, split, seed)
