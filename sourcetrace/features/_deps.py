"""Import diagnostics for the pinned ``transformers`` / ``huggingface_hub`` pair.

``transformers`` is pinned to ``>=4.38,<4.40`` (see ``pyproject.toml``) because
EnCodec-24kHz's ``encode()`` return unpacking changed after that line.  That pin
transitively requires ``huggingface_hub < 1.0``.

The failure this module exists for is not a broken environment.  It is a *user
site* shadow: a newer ``huggingface_hub`` installed into
``~/.local/lib/pythonX.Y/site-packages`` takes precedence over the correct one in
the active env, and ``transformers`` then refuses to import.  Its own error text
suggests ``pip install transformers -U``, which would break the EnCodec pin --
exactly the wrong move.  We intercept and say so.

The fix is to keep the user site out of the interpreter, via ``PYTHONNOUSERSITE=1``.
"""
from __future__ import annotations

import os


def is_user_site_shadowed(module_file: str | None, user_site: str | None) -> bool:
    """Was ``module_file`` loaded out of the ``user_site`` directory?

    Both arguments are tolerated as ``None`` (the module may be a namespace
    package, and ``site.getusersitepackages()`` is not always available), in which
    case the answer is "no" -- we only ever claim a shadow we can actually see.
    """
    if not module_file or not user_site:
        return False
    try:
        return os.path.commonpath([
            os.path.realpath(module_file),
            os.path.realpath(user_site),
        ]) == os.path.realpath(user_site)
    except (ValueError, OSError):  # different drives / unresolvable paths
        return False


def encodec_import_help(
    exc: BaseException,
    hub_file: str | None,
    user_site: str | None,
) -> str:
    """Build an actionable message for a failed EnCodec/``transformers`` import.

    The original error text is always preserved -- it carries the actual version
    diagnosis.  Guidance is appended only when ``huggingface_hub`` is demonstrably
    being loaded from the user site, so an unrelated ImportError is never dressed
    up as a shadowing problem.
    """
    original = str(exc)
    if not is_user_site_shadowed(hub_file, user_site):
        return original

    return (
        f"{original}\n\n"
        "--- sourcetrace diagnosis ---------------------------------------------\n"
        f"huggingface_hub is being loaded from your USER site-packages:\n"
        f"    {hub_file}\n"
        f"which shadows the copy in the active environment.\n\n"
        "Do NOT follow the 'pip install transformers -U' suggestion above: this\n"
        "project pins transformers >=4.38,<4.40 because EnCodec-24kHz's encode()\n"
        "return unpacking changed after 4.40, and upgrading silently breaks the\n"
        "codec-residual channel.\n\n"
        "Fix by keeping the user site out of the interpreter:\n"
        "    export PYTHONNOUSERSITE=1\n"
        f"(or remove the shadowing install from {user_site})\n"
        "-----------------------------------------------------------------------"
    )


def load_encodec_cls():
    """Import ``transformers.EncodecModel``, re-raising with guidance on failure."""
    try:
        from transformers import EncodecModel
    except ImportError as exc:
        raise ImportError(encodec_import_help(exc, _hub_file(), _user_site())) from exc
    return EncodecModel


def _hub_file() -> str | None:
    try:
        import huggingface_hub

        return getattr(huggingface_hub, "__file__", None)
    except Exception:
        return None


def _user_site() -> str | None:
    try:
        import site

        return site.getusersitepackages()
    except Exception:
        return None
