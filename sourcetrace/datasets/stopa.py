"""STOPA open-set source-tracing verification table builder.

Faithful port of the original research grader's STOPA table code:
``protocols/stopa_verif.py`` ``build_tables`` (L106-165), ``_cond_of`` (L97-103)
and the fixed EET/TEE/KNOWN/UNKNOWN partition (L84-88); plus ``stopa_audio.py``
``load_attacks_meta`` (L50-54), ``_speaker_from_name`` (L57-61) and
``list_attack_wavs`` (L64-69). Only structure, naming, typing and documentation
change: the partition, filename parsing and attack grid are identical, so results
reproduce.

Protocol (Firc et al. IS-2025; lowest published unknown-attack EER 16.43, zero-shot,
Chhibber et al. Odyssey-26 -- see the setting caveat in :mod:`sourcetrace.tasks.stopa`)
-----------------------------------------------------------------------------------
Each STOPA attack ``AAxx`` is an acoustic-model x vocoder pair, read from
``attacks.json`` (bundled at ``assets/protocols/stopa/attacks.json``):

    AA01=am01xvm01  AA02=am01xvm02  AA03=am02xvm03  AA04=am02xvm04
    AA05=am03xvm03  AA06=am03xvm04  AA07=am04xvm05  AA08=am04xvm02
    AA09=am05xvm01  AA10=am05xvm02  AA11=am06xvm06  AA12=am07xvm06
    AA13=am08xvm03

Disjoint-attack, disjoint-speaker partition (drawn from three on-disk set dirs).
This repository trains nothing on STOPA: the MLAAD-trained method is applied as is,
so the EET partition is listed here for completeness and never read.

* **EET**  (STOPA's embedding-extractor train set; unused here): ``AA11, AA12, AA13``
  -> ``EET/wav/LA_T_*``
* **TEE**  (enroll / fingerprint, KNOWN): ``AA01, 03, 05, 07, 10`` -> ``TEE/wav/LA_D_*``
* **Trials** (eval probes): 5 KNOWN (``AA01,03,05,07,10``) + 5 UNKNOWN
  (``AA02,04,06,08,09``) -> ``Trials/wav/LA_E_*``

Filenames are ``LA_{T,D,E}_<attack>_<speaker>_<cond><idx>.wav``: field 3 (0-based)
is the speaker; the trailing field's leading letter is the enrollment/eval
condition (``c`` or ``s`` on disk -- *not* the spec's co/nc/r; the official pooled
metric pools across conditions, so ``cond_filter=None`` pools all by default).

Partition constants are imported from :data:`sourcetrace.config.PROTOCOL`
(``stopa_eet_attacks``, ``stopa_tee_attacks``, ``stopa_unknown_attacks``) so the
contract is single-sourced. Audio resolves under
:data:`sourcetrace.config.PATHS.stopa_root`.
"""
from __future__ import annotations

import glob
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..config import PATHS, PROTOCOL
from . import ASSETS_DIR

# Bundled attack grid (AM x VM). The verified on-disk copy lives under
# ``$STOPA_ROOT/metadata/attacks.json``; the bundled copy makes the split
# definition self-contained and version-controlled.
ATTACKS_JSON_ASSET: Path = ASSETS_DIR / "protocols" / "stopa" / "attacks.json"

# set key -> (subdirectory, filename prefix letter) -- original SET_DIRS
# (stopa_audio.py L43-47).
_SET_DIRS: dict[str, tuple[str, str]] = {
    "trn": ("EET", "T"),        # EET  -> embedding-extractor train
    "dev": ("TEE", "D"),        # TEE  -> enroll / fingerprint (KNOWN)
    "eval": ("Trials", "E"),    # Trials -> eval probes
}

# Fixed verification partition (config-sourced; original L84-88).
EET_ATTACKS: tuple[str, ...] = PROTOCOL.stopa_eet_attacks
TEE_ATTACKS: tuple[str, ...] = PROTOCOL.stopa_tee_attacks
KNOWN_ATTACKS: tuple[str, ...] = PROTOCOL.stopa_tee_attacks       # == TEE attacks
UNKNOWN_ATTACKS: tuple[str, ...] = PROTOCOL.stopa_unknown_attacks
TRIALS_ATTACKS: tuple[str, ...] = KNOWN_ATTACKS + UNKNOWN_ATTACKS  # all probes


def load_attacks_meta(attacks_json: Path | str | None = None) -> dict[str, dict[str, str]]:
    """Load the attack grid -> ``{AAxx: {'am': am_id, 'vm': vm_id}}``.

    Faithful port of ``stopa_audio.load_attacks_meta`` (L50-54). Reads the bundled
    ``assets/protocols/stopa/attacks.json`` by default; pass a path to use the on-disk copy.
    """
    path = Path(attacks_json) if attacks_json is not None else ATTACKS_JSON_ASSET
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return {a: {"am": d["acoustic_model_id"], "vm": d["vocoder_model_id"]}
            for a, d in raw.items()}


def _speaker_from_name(fname: str) -> str:
    """``LA_<X>_<attack>_<spk>_<...>.wav`` -> ``<spk>`` (filename field 3).

    Faithful port of ``stopa_audio._speaker_from_name`` (L57-61).
    """
    parts = os.path.basename(fname).split("_")
    return parts[3] if len(parts) > 3 else "unknown"


def _cond_of(fname: str) -> str:
    """Enrollment/eval condition letter from the last filename field.

    Faithful port of ``stopa_verif._cond_of`` (L97-103):
    ``LA_<X>_<atk>_<spk>_<cond><idx>.wav`` -> ``<cond>`` (single lowercase letter,
    e.g. ``c`` or ``s``); returns ``'u'`` if the field does not start with a
    letter.
    """
    last = os.path.basename(fname).rsplit("_", 1)[-1]   # '<cond><idx>.wav'
    c = last[:1]
    return c if c.isalpha() else "u"


def list_attack_wavs(
    attack: str,
    which_set: str,
    stopa_root: Path | str | None = None,
) -> list[str]:
    """Sorted absolute wav paths for one attack from one set directory.

    Faithful port of ``stopa_audio.list_attack_wavs`` (L64-69):
    ``glob($STOPA_ROOT/<sub>/wav/LA_<prefix>_<attack>_*.wav)`` then ``sorted``.

    Args:
        attack: attack id (e.g. ``"AA01"``).
        which_set: one of ``"trn"`` (EET), ``"dev"`` (TEE), ``"eval"`` (Trials).
        stopa_root: STOPA audio root; defaults to ``config.PATHS.stopa_root``.
    """
    root = Path(stopa_root) if stopa_root is not None else PATHS.stopa_root
    sub, prefix = _SET_DIRS[which_set]
    pat = str(root / sub / "wav" / f"LA_{prefix}_{attack}_*.wav")
    return sorted(glob.glob(pat))


@dataclass(frozen=True)
class StopaTable:
    """One STOPA split (EET / TEE / Trials). All list fields are row-aligned.

    ``group`` is the per-row cached-feature group key
    (``<split>__<attack>``) used to load precomputed ``.npy`` features; the loader
    enumerates rows attack-by-attack in ``sorted(attack)`` order so embeddings and
    labels stay aligned.
    """

    split: str                     # "eet" | "tee" | "trials"
    path: list[str]                # absolute wav paths
    attack: list[str]              # AAxx
    am: list[str]                  # acoustic-model id
    vm: list[str]                  # vocoder-model id
    speaker: list[str]
    cond: list[str]                # condition letter
    set: list[str]                 # on-disk set key (trn/dev/eval)
    group: list[str]               # per-row cache group key

    def __len__(self) -> int:
        return len(self.path)


@dataclass(frozen=True)
class StopaTables:
    """The full STOPA verification split: EET / TEE / Trials + label maps."""

    eet: StopaTable
    tee: StopaTable
    trials: StopaTable
    attacks_meta: dict[str, dict[str, str]]
    atk2id: dict[str, int]         # dense over the full attack universe
    am2id: dict[str, int]
    vm2id: dict[str, int]


def _cache_group(split: str, attack: str) -> str:
    """Cached-feature group key for a (split, attack) -> ``<split>__<attack>``.

    Mirrors the original per-(split, attack) cache filename stem
    (stopa_verif.py L443) so precomputed feature ``.npy`` files line up.
    """
    return f"{split}__{attack}"


def build_tables(
    cond_filter: Iterable[str] | None = None,
    attacks_meta: dict[str, dict[str, str]] | None = None,
    stopa_root: Path | str | None = None,
) -> StopaTables:
    """Build per-split STOPA tables (EET / TEE / Trials) from the wav dirs + grid.

    Faithful port of ``stopa_verif.build_tables`` (L106-165). Each table is drawn
    from *its own* set directory (EET->trn, TEE->dev, Trials->eval), so for the
    KNOWN attacks enroll (dev) and probe (eval) pools stay disjoint. Label id
    spaces are dense over the full attack universe.

    Args:
        cond_filter: ``None`` pools all conditions (the repo default); an iterable
            of condition letters (e.g. ``{"s"}``) restricts every split's pools.
        attacks_meta: attack grid; defaults to :func:`load_attacks_meta`.
        stopa_root: STOPA audio root; defaults to ``config.PATHS.stopa_root``.

    Returns:
        A :class:`StopaTables` with EET / TEE / Trials :class:`StopaTable`s and
        the ``atk2id`` / ``am2id`` / ``vm2id`` maps.
    """
    meta = attacks_meta or load_attacks_meta()
    cf = None if cond_filter is None else set(cond_filter)

    splitmap: dict[str, tuple[tuple[str, ...], str]] = {
        "eet": (EET_ATTACKS, "trn"),
        "tee": (TEE_ATTACKS, "dev"),
        "trials": (TRIALS_ATTACKS, "eval"),
    }
    built: dict[str, StopaTable] = {}
    for split, (attacks, wset) in splitmap.items():
        cols: dict[str, list] = {k: [] for k in
                                 ("path", "attack", "am", "vm", "speaker",
                                  "cond", "set", "group")}
        for atk in attacks:
            am, vm = meta[atk]["am"], meta[atk]["vm"]
            for wav in list_attack_wavs(atk, wset, stopa_root=stopa_root):
                cond = _cond_of(wav)
                if cf is not None and cond not in cf:
                    continue
                cols["path"].append(wav)
                cols["attack"].append(atk)
                cols["am"].append(am)
                cols["vm"].append(vm)
                cols["speaker"].append(_speaker_from_name(wav))
                cols["cond"].append(cond)
                cols["set"].append(wset)
                cols["group"].append(_cache_group(split, atk))
        built[split] = StopaTable(split=split, **cols)

    # -- dense label id spaces over the full attack universe (original L150-155) - #
    all_atk = sorted(meta.keys())
    atk2id = {a: i for i, a in enumerate(all_atk)}
    am2id = {m: i for i, m in enumerate(sorted({meta[a]["am"] for a in all_atk}))}
    vm2id = {v: i for i, v in enumerate(sorted({meta[a]["vm"] for a in all_atk}))}

    return StopaTables(
        eet=built["eet"],
        tee=built["tee"],
        trials=built["trials"],
        attacks_meta=meta,
        atk2id=atk2id,
        am2id=am2id,
        vm2id=vm2id,
    )
