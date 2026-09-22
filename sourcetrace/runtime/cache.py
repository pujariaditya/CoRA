"""Diagnose the two caches an entry point reads before it does any real work.

The model cache holds the frozen front-ends; the feature cache holds what
extraction wrote. Both are environment-driven, so the common failure is pointing at
the wrong one rather than a corrupt one -- and both predicates below say which path
they looked at, because that is the fact that resolves it.

Pure predicates first (:func:`model_cache_problem`), then the ``require_*`` wrappers
that turn a problem into a :class:`SystemExit` carrying the fix.
"""

from __future__ import annotations

from pathlib import Path


def model_cache_problem(cache_dir: Path, offline: bool) -> str | None:
    """Describe why ``cache_dir`` cannot serve the frozen front-end, or ``None``.

    Only an *empty or absent* cache is reported, and only when the process cannot
    reach the hub to fill it. A populated cache is never second-guessed here: deciding
    whether a particular snapshot is complete is the hub client's job, and guessing at
    it would produce false alarms on a perfectly good cache.
    """
    if cache_dir.is_dir() and any(cache_dir.iterdir()):
        return None
    where = "is empty" if cache_dir.is_dir() else "does not exist"
    if not offline:
        return None  # it will simply be downloaded on first use
    return f"the model cache {cache_dir} {where}, and offline mode is set"


def cache_help(problem: str, cache_dir: Path) -> str:
    """An actionable message for a model cache that cannot be filled."""
    return (
        f"{problem}.\n\n"
        "--- sourcetrace: no cached front-end models -----------------------------\n"
        f"Expected WavLM-Large and EnCodec-24kHz under:\n    {cache_dir}\n\n"
        "Fetch them once (a few GB, resumable):\n\n"
        "    python scripts/download_models.py\n\n"
        "Or point ST_MODEL_CACHE at a directory that already holds them. Unset\n"
        "HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE if you meant to download now.\n"
        "-----------------------------------------------------------------------"
    )


def require_model_cache() -> None:
    """Exit with guidance if the frozen front-end cannot be loaded or fetched.

    The failure this prevents is the offline one: on a machine with no hub access an
    empty cache surfaces as a connection error from inside ``transformers``, which
    names neither the cache directory nor the script that fills it.
    """
    import os

    from sourcetrace.config import PATHS

    offline = any(
        os.environ.get(v, "").strip().lower() in ("1", "true", "yes")
        for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )
    problem = model_cache_problem(PATHS.model_cache, offline)
    if problem is not None:
        raise SystemExit(cache_help(problem, PATHS.model_cache))


def require_feature_cache(task: str, override: str | None = None) -> Path:
    """Resolve the feature cache for ``task``, or exit saying how to build it.

    Two failures collapse into one message here. Reading ``os.environ["ST_FEATURE_CACHE"]``
    directly raises a bare ``KeyError`` when the variable is unset -- which is the
    *documented default* case, since :class:`~sourcetrace.config.Paths` falls back to
    ``<ST_DATA_ROOT>/features`` -- and an unbuilt cache surfaces far downstream as a
    missing ``.npy``. Resolve through the config default, then say which command fills it.
    """
    from sourcetrace.config import PATHS

    cache = Path(override) if override else PATHS.feature_cache / task
    if not cache.is_dir():
        raise SystemExit(
            f"no {task} feature cache at {cache}.\n\n"
            "--- sourcetrace: features not extracted yet ----------------------------\n"
            f"Build it first:\n\n    python -m sourcetrace.extract --dataset {task}\n\n"
            "Or pass --cache-dir to point at an existing one. ST_FEATURE_CACHE moves the\n"
            "default location; unset, it is <ST_DATA_ROOT>/features.\n"
            "-----------------------------------------------------------------------"
        )
    return cache
