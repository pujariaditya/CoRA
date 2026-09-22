"""Cached per-group feature resolution for the standalone protocols.

The protocols consume per-group frozen-SSL feature matrices produced by
:mod:`sourcetrace.features`. A *group* is a ``(split, system)`` (MLAAD) or
``(split, attack)`` (STOPA) key; the split builders expose it per-row as
``table.group``. This helper turns a table + a feature source into row-aligned
stacked feature matrices, loading each group's ``.npy`` at most once.

A ``FeatureSource`` is any of:

* a ``dict[str, np.ndarray]`` mapping group key -> ``(n_group, feat_dim)`` array
  (in the SAME per-group row order the table enumerates), or
* a ``pathlib.Path`` / ``str`` directory containing ``<group>.npy`` files, or
* a callable ``group -> np.ndarray``.

All three are pure numpy on the grader side -- no torch, no subprocess.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np


class _HasGroup(Protocol):
    """Structural type for any split table exposing a per-row ``group`` list.

    Annotation-only (not ``runtime_checkable``): a ``Protocol`` carrying data
    members cannot be used with ``isinstance`` anyway, so marking it so would be
    misleading.
    """

    group: list[str]


FeatureSource = (
    dict[str, np.ndarray]
    | Path
    | str
    | Callable[[str], np.ndarray]
)


def _resolver(source: FeatureSource) -> Callable[[str], np.ndarray]:
    """Normalize a :data:`FeatureSource` to a ``group -> ndarray`` callable."""
    if isinstance(source, dict):
        return lambda g: np.asarray(source[g])
    if isinstance(source, (str, Path)):
        root = Path(source)
        return lambda g: np.load(root / f"{g}.npy")
    if callable(source):
        return lambda g: np.asarray(source(g))
    raise TypeError(f"unsupported feature source: {type(source)!r}")


def stack_features(table: _HasGroup, source: FeatureSource) -> np.ndarray:
    """Load and row-align a table's cached features into one ``(N, feat_dim)`` array.

    Rows are emitted in the table's own order. Consecutive (or interleaved) rows
    sharing a group are served from that group's cached array via a per-group
    cursor, so each ``.npy`` is read exactly once regardless of how the table
    interleaves groups.

    .. warning::

       **Row alignment between the cache and the table is ASSUMED, not verified.**
       The only consistency check performed here is a per-group **row COUNT**
       comparison. A cache whose rows were written in a different order than the
       table enumerates them -- same group, same number of rows, different
       permutation -- passes that check silently and yields feature vectors
       paired with the WRONG labels for every affected row. There is no checksum,
       no filename list, and no content hash to catch it.

       The invariant that makes this safe is upstream: the split builders and
       :mod:`sourcetrace.features` both enumerate a group's members
       deterministically in the same order, so a cache written by
       ``sourcetrace/extract.py`` for a given split is aligned by
       construction. That invariant breaks if a cache is regenerated with a
       different dataset revision, a different file-listing/sort order, or a
       different split seed while an older cache directory is left in place --
       the row counts can still match while the order does not. Regenerate the
       feature cache whenever the dataset, the extraction script's enumeration
       order, or the split definition changes; do not mix cache directories
       across those changes.

    Args:
        table: any split table exposing a per-row ``group`` list.
        source: a :data:`FeatureSource`.

    Returns:
        ``float32`` array of shape ``(len(table), feat_dim)``, or an empty
        ``(0, 0)`` ``float32`` array when the table has no rows.

    Raises:
        ValueError: if a group's cached row count does not match the number of
            rows the table lists for that group. Note this catches only COUNT
            mismatches -- see the warning above.
        TypeError: if ``source`` is not a supported :data:`FeatureSource`.
    """
    load = _resolver(source)
    groups = list(table.group)

    # Count table rows per group, preserving first-seen order so each cached
    # array is loaded once, in a deterministic order.
    order: list[str] = []
    counts: dict[str, int] = {}
    for g in groups:
        if g in counts:
            counts[g] += 1
        else:
            order.append(g)
            counts[g] = 1

    per_group: dict[str, np.ndarray] = {}
    for g in order:
        arr = np.asarray(load(g), dtype=np.float32)
        if arr.ndim == 1:            # a single-row group may be stored flat
            arr = arr[None, :]
        if len(arr) != counts[g]:
            raise ValueError(
                f"feature/label desync for group {g!r}: cache has {len(arr)} rows "
                f"but the table lists {counts[g]}")
        per_group[g] = arr

    if not groups:
        return np.empty((0, 0), np.float32)

    # Re-emit in table row order; per-group cursors respect interleaving.
    cursors = dict.fromkeys(order, 0)
    rows = []
    for g in groups:
        rows.append(per_group[g][cursors[g]])
        cursors[g] += 1
    return np.stack(rows).astype(np.float32, copy=False)
