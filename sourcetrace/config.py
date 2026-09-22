"""Central configuration: paths, model/dataset identifiers, and the frozen
hyperparameters of the published codec-residual source-tracing method.

Every filesystem location resolves from an environment variable with a portable
default, so the repository runs unchanged on any machine. No path is hardcoded to
a specific research drive.

Environment variables (all optional; defaults shown):
    ST_DATA_ROOT      base for datasets/caches     (default: ``./data``)
    ST_MLAAD_ROOT     MLAAD v5 audio root          (default: ``$ST_DATA_ROOT/MLAAD``)
    ST_STOPA_ROOT     STOPA audio root             (default: ``$ST_DATA_ROOT/STOPA``)
    ST_FEATURE_CACHE  extracted-feature cache      (default: ``$ST_DATA_ROOT/features``)
    ST_MODEL_CACHE    Hugging Face / torch cache   (default: ``$ST_DATA_ROOT/models``)
    ST_CPU            set to ``1`` to force CPU

Importing this module points ``HF_HOME``/``TORCH_HOME`` at ``ST_MODEL_CACHE`` (via
``setdefault``) before transformers/torch are imported, so downloaded weights land
in one portable place.

The hyperparameters are the exact values of the published champion (commit
``fa876f7d``); changing them changes the results.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from pathlib import Path

# --------------------------------------------------------------------------- #
# Repository anchor and the shipped assets
# --------------------------------------------------------------------------- #

#: The repository root, resolved from this file rather than the working
#: directory, so an entry point run from anywhere finds the same assets.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: Frozen protocol definitions that ship with the repository: the MLAAD manifests
#: and merge map, the STOPA attack grid, the realized family holdout. They live at
#: the top level rather than inside the package because they are data a reader is
#: meant to open and diff, not an implementation detail -- and because the paper
#: and the docs cite them by repository path.
#:
#: This makes an editable install (``pip install -e .``) the supported one: a
#: wheel would not carry them. That is the documented install.
ASSETS_DIR: Path = REPO_ROOT / "assets"

# --------------------------------------------------------------------------- #
# Paths (environment-driven, portable)
# --------------------------------------------------------------------------- #


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(os.path.expandvars(os.path.expanduser(raw))) if raw else default


@dataclass(frozen=True)
class Paths:
    """Filesystem locations, all overridable via ``ST_*`` environment variables."""

    data_root: Path = field(default_factory=lambda: _env_path("ST_DATA_ROOT", Path("data")))

    @property
    def mlaad_root(self) -> Path:
        return _env_path("ST_MLAAD_ROOT", self.data_root / "MLAAD")

    @property
    def stopa_root(self) -> Path:
        return _env_path("ST_STOPA_ROOT", self.data_root / "STOPA")

    @property
    def feature_cache(self) -> Path:
        return _env_path("ST_FEATURE_CACHE", self.data_root / "features")

    @property
    def model_cache(self) -> Path:
        return _env_path("ST_MODEL_CACHE", self.data_root / "models")


PATHS = Paths()

# Point HF/torch downloads at the portable model cache before they are imported.
os.environ.setdefault("HF_HOME", str(PATHS.model_cache / "hf"))
os.environ.setdefault("TORCH_HOME", str(PATHS.model_cache / "torch"))


def device() -> str:
    """Resolve the compute device. ``ST_CPU=1`` forces CPU."""
    if os.environ.get("ST_CPU") == "1":
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------- #
# Pretrained models + public dataset sources
# --------------------------------------------------------------------------- #

WAVLM_MODEL_ID = "microsoft/wavlm-large"        # frozen SSL frontend (~316M params)
ENCODEC_MODEL_ID = "facebook/encodec_24khz"     # neural codec for the residual channel
MLAAD_HF_REPO = "mueller91/MLAAD"               # Hugging Face dataset
STOPA_ZENODO_DOI = "10.5281/zenodo.15606628"    # Zenodo record

# Published SOTA anchors for the results table (verified against the papers).
SOTA_MLAAD_V5_FPR95 = 3.36                       # Neamtu et al. 2026 (arXiv 2606.10758)
SOTA_STOPA_UNKNOWN_EER = 16.43                   # Chhibber et al., Odyssey 2026 (arXiv 2509.24674)


# --------------------------------------------------------------------------- #
# Frozen method hyperparameters (champion fa876f7d)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureConfig:
    """Front-end feature extraction (audio -> 2133-d pooled vector)."""

    sample_rate: int = 16_000
    clip_seconds: float = 4.0
    # WavLM layers 1-4, the shallowest band. Selected on the development split (dev OOD-EER
    # 1.95 for 1-4, 2.11 for 4-7, 3.16 for 8-11) and confirmed on evaluation (FPR95 0.50 / 0.91
    # / 1.58): see
    # results/layer_band_summary.json. Every file in results/ and both checkpoints use this band.
    wavlm_layers: tuple[int, ...] = (1, 2, 3, 4)
    pool_stats: tuple[str, ...] = ("mean", "std")   # per-layer temporal stats -> 2*1024
    ssl_dim: int = 2048
    sig_nfft: int = 512
    sig_hop: int = 160
    sig_bands: int = 24                             # group-delay bands
    sig_mod: int = 16                              # low-freq modulation bins
    sig_codec: int = 12                            # codec-grid fingerprint
    sig_dim: int = 52
    codec_sr: int = 24_000
    codec_bandwidths: tuple[float, ...] = (1.5, 6.0, 12.0)
    codec_bands: int = 8
    codec_dim: int = 33                            # 3*(1+8) + 3 softmax(-e) + 3 pairwise
    feat_dim: int = 2133                           # 2048 + 52 + 33


@dataclass(frozen=True)
class ModelConfig:
    """Factorized gated embedding head (and the auxiliary head trained beside it)."""

    fact_am_dim: int = 192
    fact_voc_dim: int = 64
    fact_codec_dim: int = 32
    base_emb_dim: int = 288                         # 192 + 64 + 32: the fused embedding z
    voc_weight: float = 0.1
    codec_weight: float = 0.5
    voc_gate_beta: float = 0.30                     # AM->vocoder FiLM depth
    codec_gate_beta: float = 0.30                   # AM->codec FiLM depth
    tmp_gate_beta: float = 0.030                    # mean->std FiLM depth (bounded)
    tmp_gate_hidden: int = 64
    # Gate weights are drawn N(0, gate_init_std^2) with zero bias, so every gate starts
    # near identity and the head begins training from a clean concatenation. Named here
    # rather than left literal in models/head.py because the paper states it (Sec. 2.2)
    # and every constant the paper states must have one central definition.
    gate_init_std: float = 0.05
    # The 33->32 residual projection and its FiLM gate are trained with the rest of the head.
    # Setting this False leaves them at their random initialisation (the earlier design):
    # FPR95 0.78 against 0.50 and OOD-EER 3.36 against 2.45 on MLAAD v5, the "fixed
    # projection" row of the paper's Table 2.
    train_codec_path: bool = True
    # One embedding for every task: the open-set score, the known-model label and the STOPA
    # protocol all read the 288-d z. (An earlier design appended a 512-d PCA-whitened SSL
    # channel for STOPA; it is gone.) Must equal base_emb_dim.
    emb_dim: int = 288


@dataclass(frozen=True)
class TrainConfig:
    """Head training objective and optimizer."""

    lr: float = 3e-4
    # Adam is used with NO weight decay. This was implicit -- the original call was
    # ``torch.optim.Adam(params, lr=LR)`` and relied on the library default -- but the
    # paper reports it (Sec. 2.3), so it is stated here instead of inferred from an
    # absent argument. 0.0 is exactly the default, so the fit is unchanged.
    weight_decay: float = 0.0
    # Per-class anchors are initialised ``randn(C, D) * anchor_init_std``. Named here for
    # the same reason as MODEL.gate_init_std: the paper states it (Sec. 2.3), so it must
    # have one definition rather than two literals in method.py.
    anchor_init_std: float = 0.01
    epochs: int = 300
    batch_size: int = 512
    logit_scale: float = 12.0
    # The auxiliary head (paper Sec. 2.3: "used only during training"): its own cosine
    # cross-entropy at scale 14, added to the loss with weight 1.
    xfer_logit_scale: float = 14.0
    xfer_loss_weight: float = 1.0
    supcon_weight: float = 0.05
    supcon_tau: float = 0.10
    hamos_weight: float = 0.03
    hamos_n_ood: int = 256
    hamos_tau: float = 0.07


@dataclass(frozen=True)
class ScoringConfig:
    """Open-set scoring (conformal margin + relative-Mahalanobis), small-open-set (v5)."""

    cal_holdout: float = 0.15
    conf_std_k: float = 3.5                         # per-class margin offset k (v5, C<80)
    cal_shrink_n0: float = 20.0
    conf_lambda: float = 0.0                        # split-conformal blend (0 = margin only)
    conf_alpha: float = 0.05
    cos_weight: float = 1.0                         # conformal-margin blend weight (ablation knob)
    dens_weight: float = 1.0                        # relative-Mahalanobis blend weight
    lang_infl_k: int = 3                            # within-class language-PC inflation (v5)
    lang_infl_alpha: float = 16.0
    large_os_threshold: int = 80
    # Shrinkage-weight clamp for the covariance estimator. ADDED, not moved: these were
    # literals inside scoring/mahalanobis.py, so the formula the paper prints --
    # gamma = clip(d/(d+n), 0.05, 0.9) -- was unauditable from config.py. Values are the
    # ones results/ was measured with.
    shrink_gamma_min: float = 0.05
    shrink_gamma_max: float = 0.9


@dataclass(frozen=True)
class ProtocolConfig:
    """Evaluation protocols (MLAAD v5 family-level split + STOPA verification).

    The ``panda_*`` names are this repository's label for the MLAAD v5 protocol of
    Neamtu et al. (2026); that paper names no acronym.
    """

    panda_seed: int = 42
    panda_ood_fraction: float = 0.15
    panda_max_per_model: int = 2000
    panda_train_ratio: float = 0.70
    panda_val_ratio: float = 0.15
    # STOPA's released partition. The EET attacks are listed for completeness only: no
    # STOPA utterance is used for training. The MLAAD-trained model is applied as is,
    # TEE forms the enrolment fingerprints and Trials are the probes.
    stopa_eet_attacks: tuple[str, ...] = ("AA11", "AA12", "AA13")
    stopa_tee_attacks: tuple[str, ...] = ("AA01", "AA03", "AA05", "AA07", "AA10")
    stopa_unknown_attacks: tuple[str, ...] = ("AA02", "AA04", "AA06", "AA08", "AA09")


# --------------------------------------------------------------------------- #
# Ablation overrides (env-driven; UNSET => champion values, byte-identical)
# --------------------------------------------------------------------------- #
# These let scripts run leave-one-out ablations without editing this file. With no
# ST_ABL_* variable set, every value below is exactly the frozen champion default.
#   ST_ABL_CODEC_WEIGHT   MODEL.codec_weight   (0 => drop codec-residual subspace)
#   ST_ABL_VOC_WEIGHT     MODEL.voc_weight     (0 => drop signature/vocoder subspace)
#   ST_ABL_NO_GATING=1    zero all FiLM gate betas (naive concat, no learned gating)
#   ST_ABL_COS_WEIGHT     SCORING.cos_weight   (0 => relative-Mahalanobis only)
#   ST_ABL_DENS_WEIGHT    SCORING.dens_weight  (0 => conformal margin only)


def _env_float(var: str, default: float) -> float:
    raw = os.environ.get(var)
    return float(raw) if raw not in (None, "") else default


def _env_flag(var: str) -> bool:
    return os.environ.get(var, "").lower() in ("1", "true", "yes")


# --------------------------------------------------------------------------- #
# The YAML overlay (configs/*.yaml)
# --------------------------------------------------------------------------- #
#
# Precedence, lowest to highest:  dataclass defaults -> YAML -> ST_ABL_* env.
#
# The YAML is read HERE, at import time, and not from a ``--config`` flag handled
# in ``main()``. That is not a stylistic choice. Every consumer binds its config
# at module scope -- ``from ..config import MODEL`` in models/head.py,
# scoring/*.py, losses/objectives.py -- so the object is captured the moment that
# module is imported. Rebinding ``sourcetrace.config.MODEL`` after argparse has
# run would change nothing those modules can see, and would do it silently: the
# run would report the config it was asked for and fit the one it already had.
#
# So the path is resolved before any dataclass is instantiated, from ST_CONFIG or
# by reading ``--config`` out of sys.argv directly. Entry points still declare
# ``--config`` in their parser, for ``--help`` and for rejecting a bad path, but
# the value has already been consumed by then.


def _config_path() -> Path | None:
    """The overlay to apply: a ``--config`` in ``sys.argv``, else ``ST_CONFIG``.

    The command line wins over the environment, which is the opposite of the
    ST_ABL_* ordering below and is deliberate. ST_ABL_* is last there because it
    IS the command line -- ``ST_ABL_CODEC_WEIGHT=0 python -m ...`` is what the
    user typed. ST_CONFIG is ambient: exported once in a shell, or inherited by a
    subprocess. Something ambient must not silently replace the file named in the
    invocation.
    """
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    raw = os.environ.get("ST_CONFIG")
    if raw:
        return Path(os.path.expandvars(os.path.expanduser(raw)))
    return None


#: YAML section name -> the dataclass it overlays.
_SECTIONS = {
    "features": FeatureConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "scoring": ScoringConfig,
    "protocol": ProtocolConfig,
}


def _overlay(path: Path | None) -> dict[str, dict]:
    """Read a config YAML into ``{section: {field: value}}``, or ``{}`` for none.

    Unknown sections and unknown fields are errors rather than no-ops. A typo in
    a hyperparameter name is the failure this has to catch: silently ignoring it
    produces a run that claims to be an ablation and is in fact the baseline.
    """
    if path is None:
        return {}
    if not path.is_file():
        raise SystemExit(f"config not found: {path}")

    import yaml

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: top level must be a mapping of sections")

    out: dict[str, dict] = {}
    for section, values in raw.items():
        if section in ("name", "description"):
            continue  # documentation keys, not hyperparameters
        if section not in _SECTIONS:
            raise SystemExit(
                f"{path}: unknown section {section!r}; "
                f"expected one of {', '.join(sorted(_SECTIONS))}"
            )
        if not isinstance(values, dict):
            raise SystemExit(f"{path}: section {section!r} must be a mapping")
        fields = {f.name: f for f in dc_fields(_SECTIONS[section])}
        for key, value in values.items():
            if key not in fields:
                raise SystemExit(f"{path}: {section}.{key} is not a field of "
                                 f"{_SECTIONS[section].__name__}")
            # YAML has no tuples, and these fields are compared and hashed as
            # tuples everywhere downstream.
            if isinstance(value, list):
                value = tuple(value)
            out.setdefault(section, {})[key] = value
    return out


_CONFIG_PATH = _config_path()
_OVERLAY = _overlay(_CONFIG_PATH)

#: The overlay actually in force, for provenance in a result JSON. ``None`` means
#: the frozen champion defaults, which is what ``configs/base.yaml`` also encodes.
CONFIG_NAME: str | None = _CONFIG_PATH.name if _CONFIG_PATH else None

_NO_GATING = _env_flag("ST_ABL_NO_GATING")

FEATURES = FeatureConfig(**_OVERLAY.get("features", {}))
MODEL = ModelConfig(**{
    **dict(
        voc_weight=0.1,
        codec_weight=0.5,
        voc_gate_beta=0.30,
        codec_gate_beta=0.30,
        tmp_gate_beta=0.030,
    ),
    **_OVERLAY.get("model", {}),
    # ST_ABL_* stays the highest-precedence layer: run_ablations.sh and
    # the paper both document it, and a config file must not silently
    # override an override the user typed on the command line.
    **({"voc_gate_beta": 0.0, "codec_gate_beta": 0.0, "tmp_gate_beta": 0.0}
       if _NO_GATING else {}),
    **({"voc_weight": _env_float("ST_ABL_VOC_WEIGHT", 0.0)}
       if os.environ.get("ST_ABL_VOC_WEIGHT") else {}),
    **({"codec_weight": _env_float("ST_ABL_CODEC_WEIGHT", 0.0)}
       if os.environ.get("ST_ABL_CODEC_WEIGHT") else {}),
})
TRAIN = TrainConfig(**_OVERLAY.get("train", {}))
SCORING = ScoringConfig(**{
    **dict(cos_weight=1.0, dens_weight=1.0),
    **_OVERLAY.get("scoring", {}),
    **({"cos_weight": _env_float("ST_ABL_COS_WEIGHT", 1.0)}
       if os.environ.get("ST_ABL_COS_WEIGHT") else {}),
    **({"dens_weight": _env_float("ST_ABL_DENS_WEIGHT", 1.0)}
       if os.environ.get("ST_ABL_DENS_WEIGHT") else {}),
})
PROTOCOL = ProtocolConfig(**_OVERLAY.get("protocol", {}))
