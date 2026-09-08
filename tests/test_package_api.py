"""Package-level API surface: nmr/__init__.py must mirror every submodule's __all__."""

from __future__ import annotations

import importlib
import pathlib


def test_nmr_package_reexports_all_module_public_symbols() -> None:
    """Every public symbol (module ``__all__``) must be importable from the
    ``nmr`` package top level — ``from nmr import DataConfig`` is part of the
    documented public surface. Underscore-prefixed private modules
    (``_transforms``, ``_gpu``, ``_atomicio``) are exempt by design: they are
    embedded by value in deploy artifacts and never public API."""
    import nmr

    missing: dict[str, list[str]] = {}
    for path in sorted(pathlib.Path(nmr.__file__).parent.glob("*.py")):  # type: ignore[union-attr]
        if path.name.startswith("_"):
            continue
        mod = importlib.import_module(f"nmr.{path.stem}")
        for name in getattr(mod, "__all__", []):
            if not hasattr(nmr, name):
                missing.setdefault(mod.__name__, []).append(name)
    assert missing == {}, (
        "nmr/__init__.py is missing re-exports for public module symbols; "
        f"add them (imports AND __all__): {missing}"
    )


def test_nmr_package_exports_model_backend_protocol_types_only() -> None:
    import nmr

    for name in ("BackendAdapter", "BackendCapabilities", "BackendIdentity"):
        assert name in nmr.__all__
        assert getattr(nmr, name) is not None

    assert "canonical_json_bytes" not in nmr.__all__
    assert "normalize_identity_value" not in nmr.__all__


def test_nmr_package_exports_backend_registry() -> None:
    import nmr

    assert "BackendRegistry" in nmr.__all__
    assert getattr(nmr, "BackendRegistry") is not None
