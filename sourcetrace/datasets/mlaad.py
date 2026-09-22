"""MLAAD v5 open-set source-tracing split builder.

Faithful port of the original research grader ``protocols/mlaad_klein.py``
``build_tables()`` (source lines 251-365), together with its merge map
(``_MERGE_MAP`` / ``_family``, L217-239), manifest reader (``_read_manifest``,
L195-202), language parser (``_lang_of``, L205-209) and deterministic subsample
(``_subsample_indices`` / ``hash_seed``, L493-540). Only structure, naming,
typing and documentation change: the split logic, seed, ratios and label space
are byte-for-byte identical, so results reproduce.

Protocol (Neamtu et al. 2026, arXiv 2606.10758; SOTA FPR95 = 3.36)
-------------------------------------------------------------------
Every MLAAD v5 sample is pooled from the Mueller protocol manifests
(``train.csv`` / ``dev.csv`` / ``eval.csv``, header ``path,model_name``), then:

* **Family-level OOD holdout** (the 2026-06-30 fidelity fix, original L280-299):
  systems are grouped into *acoustic-model architecture families* via the
  merge map (``_family``); ``ceil``-ish ``max(6, round(0.15 * n_families))``
  families are held out **entirely** as OOD (``PANDA_OOD_FRACTION = 0.15``), the
  first half -> dev calibration, the second half -> eval test. Holding out at the
  family level (not the system level) guarantees no architecture spans ID and
  OOD, removing the within-family near-duplicate (e.g. one Bark variant OOD while
  its siblings stay ID) that was the entire v5-OOD-EER floor (~12.6 -> ~7.4).

* **Per-ID-system 70/15/15 sample split** (original L314-329): each ID system's
  samples are capped at ``PANDA_MAX_PER_MODEL = 2000`` *before* splitting
  (deterministic, seeded by ``hash_seed("panda", system)``), then split
  ``PANDA_TRAIN_RATIO = 0.70`` train / ``PANDA_VAL_RATIO = 0.15`` dev / remaining
  0.15 eval. The model thus *sees every ID system in training*; the open-set task
  is to reject the held-out OOD families while accepting unseen samples of known
  systems -- the regime that yields FPR95 ~3.36.

* **Label space** (original L302-312): class label = architecture family;
  ID families get dense ids ``0..K-1``, every OOD sample gets the sentinel id
  ``K``, so ``is_ID == (class_id < K)``.

All protocol constants are imported from :data:`sourcetrace.config.PROTOCOL`
(``panda_seed=42``, ``panda_ood_fraction=0.15``, ``panda_max_per_model=2000``,
``panda_train_ratio=0.70``, ``panda_val_ratio=0.15``) so the contract is single-
sourced. Audio paths resolve under :data:`sourcetrace.config.PATHS.mlaad_root`.

The bundled manifests (``assets/protocols/mlaad/{train,dev,eval}.csv``) and merge
map (``assets/protocols/panda_merge_map.json``) are the exact split definitions shipped in
the original grader. ``assets/provenance/v5_train_split_replicated.json`` is a provenance
dump of an earlier run; see ``assets/README.md``.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import PATHS, PROTOCOL
from . import ASSETS_DIR

# --------------------------------------------------------------------------- #
# Asset locations (bundled, version-controlled split definitions)
# --------------------------------------------------------------------------- #
PRIVATE_MLAAD_DIR: Path = ASSETS_DIR / "protocols" / "mlaad"
MERGE_MAP_PATH: Path = ASSETS_DIR / "protocols" / "panda_merge_map.json"


# --------------------------------------------------------------------------- #
# Merge map -> architecture family label (original _MERGE_MAP / _family)
# --------------------------------------------------------------------------- #
def _load_merge_map() -> dict[str, str]:
    """Load the version-merge map (``assets/protocols/panda_merge_map.json``).

    Faithful to the original ``_MERGE_MAP`` (mlaad_klein.py L217-229): a MILD map
    where only true version duplicates collapse (Llasa sizes, Parler variants,
    Bark variants, the two en/ljspeech vits, ...). Aggressive acoustic-family
    merging is deliberately NOT used -- it makes held-out same-family systems
    unrejectable and was the FPR95=62% bug. Keys are ``_safe``-normalized system
    names.
    """
    with MERGE_MAP_PATH.open(encoding="utf-8") as fh:
        return dict(json.load(fh))


_MERGE_MAP: dict[str, str] = _load_merge_map()


def _safe(name: str) -> str:
    """Normalize a system name to its merge-map key (original ``_family`` L238)."""
    return name.replace("/", "_").replace(" ", "_")


def _family(model_name: str) -> str:
    """Family class label for a system (faithful port of ``_family``, L232-239).

    Each system is its own class *except* true version duplicates, which merge per
    the merge map. This fine granularity makes held-out systems genuinely novel
    classes (rejectable) -- the source of the paper's low FPR95.
    """
    return _MERGE_MAP.get(_safe(model_name), model_name)


# --------------------------------------------------------------------------- #
# Manifest + path parsing (original _read_manifest / _lang_of)
# --------------------------------------------------------------------------- #
def _read_manifest(path: Path) -> list[dict[str, str]]:
    """Read a protocol CSV (header ``path,model_name``) -> rows in file order.

    Faithful port of ``_read_manifest`` (L195-202). File order is preserved
    because the downstream per-system split relies on a deterministic pool order.
    """
    with path.open(newline="", encoding="utf-8") as fh:
        return [{"path": r["path"], "model_name": r["model_name"]}
                for r in csv.DictReader(fh)]


def _lang_of(rel_path: str) -> str:
    """Language id from a manifest path ``./fake/<lang>/<system>/<file>.wav``.

    Faithful port of ``_lang_of`` (L205-209).
    """
    parts = rel_path.lstrip("./").split("/")
    return parts[1] if len(parts) >= 2 else "unk"


# --------------------------------------------------------------------------- #
# Deterministic seeding + subsample (original hash_seed / _subsample_indices)
# --------------------------------------------------------------------------- #
def hash_seed(*parts: object) -> int:
    """Stable 32-bit seed from arbitrary parts (faithful port, L538-540).

    ``int(md5("\\x1f".join(parts))[:8], 16)`` -- used to seed the per-system
    subsample so a (split, system) selection is reproducible run-to-run.
    """
    digest = hashlib.md5("\x1f".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:8], 16)


def _subsample_indices(n: int, cap: int | None, seed: int) -> np.ndarray:
    """Deterministic subsample of ``<=cap`` indices out of ``n``.

    Faithful port of ``_subsample_indices`` (L493-500): shuffle ``arange(n)`` with
    a fixed-seed ``np.random.default_rng``, truncate to ``cap``, then *sort* so the
    resulting order is stable and append-friendly.
    """
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    if cap is not None and cap > 0:
        idx = idx[:cap]
    return np.sort(idx)


# --------------------------------------------------------------------------- #
# Split table container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MlaadTable:
    """One split (train / dev / eval) of the MLAAD v5 source-tracing task.

    All list/array fields are row-aligned. ``group`` is the per-row cached-feature
    group key (``<split>__<safe(system)>``) used to load precomputed ``.npy``
    features; ``class_id`` is the architecture-family id (ID) or the sentinel
    ``K`` (OOD); ``is_id == class_id < K``.
    """

    split: str
    rel: list[str]                 # manifest-relative paths (``./fake/...``)
    path: list[str]                # absolute wav paths under PATHS.mlaad_root
    model_name: list[str]          # raw MLAAD system name
    lang: list[str]                # language string
    lang_id: np.ndarray            # int64, dense language id
    class_id: np.ndarray           # int64, family id (ID) or sentinel K (OOD)
    is_id: np.ndarray              # bool, class_id < K
    group: list[str]               # per-row cache group key

    def __len__(self) -> int:
        return len(self.rel)


@dataclass(frozen=True)
class MlaadSplit:
    """The full MLAAD v5 split: train / dev / eval tables + label maps."""

    train: MlaadTable
    dev: MlaadTable
    eval: MlaadTable
    n_known_classes: int                    # K = number of ID (known) families
    train_class_names: list[str]            # id -> family name, index == class id
    name2id: dict[str, int]                 # system -> class id (OOD -> K)
    roles: dict[str, str]                   # system -> "ID" | "OOD"
    lang2id: dict[str, int]                 # language -> dense id


def _cache_group(split: str, model_name: str) -> str:
    """Cached-feature group key for a (split, system) -> ``<split>__<safe(name)>``.

    Mirrors the original per-(split, model) cache filename stem
    (mlaad_klein.py L530) so precomputed feature ``.npy`` files line up.
    """
    return f"{split}__{_safe(model_name)}"


def build_split(
    mlaad_root: Path | str | None = None,
    seed: int | None = None,
) -> MlaadSplit:
    """Build the MLAAD v5 family-level open-set split.

    Faithful port of ``build_tables`` (mlaad_klein.py L251-365). Pools all samples
    from the bundled protocol manifests, holds out ``0.15`` of architecture
    families entirely as OOD (family level; first half -> dev, second half ->
    eval), splits each ID system's (<=2000) samples 70/15/15, and labels by
    architecture family.

    Args:
        mlaad_root: MLAAD audio root; defaults to ``config.PATHS.mlaad_root``.
        seed: OOD-family permutation seed; defaults to ``PROTOCOL.panda_seed``
            (42). Overriding it changes the split (do not, to reproduce results).

    Returns:
        An :class:`MlaadSplit` with train/dev/eval :class:`MlaadTable`s and the
        label maps.
    """
    root = Path(mlaad_root) if mlaad_root is not None else PATHS.mlaad_root
    panda_seed = PROTOCOL.panda_seed if seed is None else int(seed)

    # -- pool every unique sample from the three manifests (original L267-273) -- #
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for name in ("train.csv", "dev.csv", "eval.csv"):
        for row in _read_manifest(PRIVATE_MLAAD_DIR / name):
            if row["path"] in seen:
                continue
            seen.add(row["path"])
            rows.append(row)

    by_sys: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_sys.setdefault(row["model_name"], []).append(row)
    systems = sorted(by_sys)

    # -- family-level OOD holdout (original L287-303; the 2026-06-30 fix) ------- #
    fams = sorted({_family(s) for s in systems})
    rng = np.random.default_rng(panda_seed)
    perm = rng.permutation(len(fams))
    n_ood_fam = max(6, int(round(PROTOCOL.panda_ood_fraction * len(fams))))
    ood_fam_order = [fams[i] for i in perm[:n_ood_fam]]
    ood_fams = set(ood_fam_order)
    half = len(ood_fam_order) // 2
    ood_cal_fams = set(ood_fam_order[:half])       # -> dev calibration
    # remaining ood families (ood_fam_order[half:]) -> eval test, handled below as
    # "OOD system not in ood_cal" (identical to the original L295/L328 logic).

    id_sys = [s for s in systems if _family(s) not in ood_fams]
    ood_sys = [s for s in systems if _family(s) in ood_fams]
    ood_cal = {s for s in ood_sys if _family(s) in ood_cal_fams}
    id_set = set(id_sys)

    id_fams = sorted({_family(s) for s in id_sys})  # disjoint from ood_fams
    fam2id = {f: i for i, f in enumerate(id_fams)}
    n_known = len(id_fams)                          # K = number of ID classes

    # system -> class id (ID family id; OOD -> sentinel K) and role
    name2id: dict[str, int] = {}
    roles: dict[str, str] = {}
    for s in systems:
        if s in id_set:
            name2id[s] = fam2id[_family(s)]
            roles[s] = "ID"
        else:
            name2id[s] = n_known
            roles[s] = "OOD"

    # -- per-system partition into train/dev/eval (original L314-329) ---------- #
    part: dict[str, list[dict[str, str]]] = {"train": [], "dev": [], "eval": []}
    for s in systems:
        srows = list(by_sys[s])
        sel = _subsample_indices(
            len(srows), PROTOCOL.panda_max_per_model, seed=hash_seed("panda", s))
        srows = [srows[i] for i in sel]
        if s in id_set:
            n = len(srows)
            n_tr = int(PROTOCOL.panda_train_ratio * n)
            n_val = int(PROTOCOL.panda_val_ratio * n)
            for j, row in enumerate(srows):
                dst = "train" if j < n_tr else ("dev" if j < n_tr + n_val else "eval")
                part[dst].append(row)
        else:
            part["dev" if s in ood_cal else "eval"].extend(srows)

    # -- language nuisance ids: fit on TRAIN, extend for dev/eval (L333-339) --- #
    train_langs = sorted({_lang_of(r["path"]) for r in part["train"]})
    lang2id = {lang: i for i, lang in enumerate(train_langs)}
    for row in part["dev"] + part["eval"]:
        lang = _lang_of(row["path"])
        if lang not in lang2id:
            lang2id[lang] = len(lang2id)

    def _table(split: str, prows: Sequence[dict[str, str]]) -> MlaadTable:
        rel = [r["path"] for r in prows]
        model = [r["model_name"] for r in prows]
        lang = [_lang_of(p) for p in rel]
        cid = np.array([name2id[m] for m in model], dtype=np.int64)
        return MlaadTable(
            split=split,
            rel=rel,
            path=[str(root / p[2:]) for p in rel],   # drop leading './'
            model_name=model,
            lang=lang,
            lang_id=np.array([lang2id[lg] for lg in lang], dtype=np.int64),
            class_id=cid,
            is_id=cid < n_known,
            group=[_cache_group(split, m) for m in model],
        )

    return MlaadSplit(
        train=_table("train", part["train"]),
        dev=_table("dev", part["dev"]),
        eval=_table("eval", part["eval"]),
        n_known_classes=n_known,
        train_class_names=list(id_fams),
        name2id=name2id,
        roles=roles,
        lang2id=lang2id,
    )
