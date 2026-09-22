"""sourcetrace: open-set audio deepfake attribution with neural codec residuals (CoRA).

Attribute a synthetic utterance to its synthesis model and reject unseen models,
using a frozen self-supervised speech frontend, a spectral signature and an EnCodec
reconstruction residual, fused by a light trainable head into one 288-d embedding
that carries the open-set score, the known-model label and the cross-corpus STOPA
protocol alike.

Public API. Two distinct laziness steps, which are easy to conflate:

1. Importing this package loads no torch-backed module at all. Touching one of the
   names below imports it (and therefore torch, transformers, s3prl).
2. Importing those modules still builds no model. WavLM and EnCodec are constructed on
   the first extraction call -- ``WavLMFrontend._ensure_model`` / the EnCodec equivalent
   -- so ``Extractor(...)`` is cheap and the first ``extract`` is not.

* :class:`Extractor` / :func:`extract_features` -- frozen-SSL feature extractor
  (audio -> 2133-d), :mod:`sourcetrace.features`.
* :class:`Method` -- the trainable head + open-set score, with checkpoint
  ``save`` / ``load``, :mod:`sourcetrace.method`.
* :func:`compute_eer` / :func:`compute_fpr95` / :func:`compute_stopa_eer`
  -- the exact protocol metrics (lightweight, eager), :mod:`sourcetrace.metrics.protocol`.

* :func:`require_torch` / :func:`require_frontend` / :func:`require_packages` -- exit
  with an actionable message instead of a traceback when the interpreter has no usable
  PyTorch, or when a needed package is missing, :mod:`sourcetrace.runtime`. Every
  entry point that imports a heavy dependency calls the one that matches what it
  needs -- in ``scripts/``, in ``analysis/``, and the ``python -m`` modules below.

Entry points, all driven through ``python -m``:

``sourcetrace.extract``
    Audio -> the 2133-d feature cache, one ``<group>.npy`` per split/system.
``sourcetrace.train``
    Fit the head on MLAAD v5 and write the one checkpoint both protocols use.
``sourcetrace.evaluate``
    Score that checkpoint on MLAAD v5 and/or STOPA (no STOPA training), or fit and
    score in one shot.

The shell wrappers in ``scripts/`` are the documented way in --
``scripts/smoke.py`` to check an install, ``scripts/verify_results.sh`` for the
reported table, ``scripts/run_ablations.sh`` for the ablation sweep.

The evaluation protocols themselves live in :mod:`sourcetrace.tasks` (``fit_mlaad_v5``
/ ``eval_mlaad_v5`` / ``evaluate_mlaad_v5`` and ``eval_stopa``); the split builders in
:mod:`sourcetrace.datasets`; the frozen hyperparameters and the ``configs/*.yaml``
overlay in :mod:`sourcetrace.config`.
"""
from .runtime import require_frontend, require_packages, require_torch

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("sourcetrace")
except (ImportError, PackageNotFoundError):  # not installed, e.g. a source checkout
    __version__ = "unknown"

# Stable public aliases for the metric functions.
#
# ``compute_eer`` and ``compute_stopa_eer`` are NOT the same statistic, and the generic
# name is the trap: this literature carries two incompatible EER conventions (Sec. 3.1 of
# the paper says so, and reports both). ``compute_eer`` is ``ood_eer``, the ASYMMETRIC
# Klein-repository convention -- fpr at the vertex minimising |fpr - (1 - tpr)|.
# ``compute_stopa_eer`` is ``stopa_eer``, the SYMMETRIC (fpr + fnr) / 2 used by the STOPA
# benchmark. Numbers from the two are not comparable; prefer the specific names in new
# code, and if you use ``compute_eer`` say which convention you mean when you report it.
__all__ = [
    "__version__",
    "Extractor", "extract_features", "Method",
    "compute_eer", "compute_fpr95", "compute_stopa_eer",
    "fpr95_panda", "ood_eer", "stopa_eer",
    "require_torch", "require_frontend", "require_packages",
]

# name -> (submodule, attribute); imported on first access so a bare
# ``import sourcetrace`` never pulls in torch or builds a model.
_LAZY: dict[str, tuple[str, str]] = {
    "Extractor": ("features", "Extractor"),
    "extract_features": ("features", "extract_features"),
    "Method": ("method", "Method"),
    # The metrics were imported EAGERLY here until eval 19, which pulled scikit-learn on
    # any ``import sourcetrace.*`` -- including ``sourcetrace.runtime``, the module whose
    # whole job is to diagnose missing dependencies. The result was that every entry
    # point's ``--help`` died with a ModuleNotFoundError traceback on exactly the
    # incomplete interpreter the guards exist to explain. They are lazy now, like the
    # rest.
    "fpr95_panda": ("metrics.protocol", "fpr95_panda"),
    "ood_eer": ("metrics.protocol", "ood_eer"),
    "stopa_eer": ("metrics.protocol", "stopa_eer"),
    # Stable aliases; see the convention note above.
    "compute_fpr95": ("metrics.protocol", "fpr95_panda"),
    "compute_eer": ("metrics.protocol", "ood_eer"),
    "compute_stopa_eer": ("metrics.protocol", "stopa_eer"),
}


def __getattr__(name: str):  # PEP 562 lazy attribute loading
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module("." + mod, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(list(globals()) + __all__))
