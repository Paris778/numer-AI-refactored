"""Feature-set resolution and stability screening for research campaigns.

Pure functions over ``features.json`` and the train frame; no model logic and
no file state beyond the explicit ``features_json`` argument. Derived subsets
must remain pure functions of their inputs so the run_id fingerprint (config +
data_version + ``nmr/*.py`` + env) is unchanged by subset selection.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["resolve_feature_sets"]


def resolve_feature_sets(features_json: Path) -> dict[str, list[str]]:
    """Return every named feature set in ``features.json``, deterministically ordered.

    Includes the canonical sets (small/medium/all) and the obfuscated family
    sets (intelligence, charisma, sunshine, ...) exactly as declared. Pure
    function of the file contents; values are defensive copies.
    """
    path = Path(features_json)
    raw = json.loads(path.read_text(encoding="utf-8"))
    sets = raw.get("feature_sets")
    if not isinstance(sets, dict) or not sets:
        raise ValueError(f"{path}: 'feature_sets' must be a non-empty mapping")
    result: dict[str, list[str]] = {}
    for name, values in sorted(sets.items()):
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(
                f"{path}: feature set {name!r} must be a list of strings"
            )
        result[name] = list(values)
    return result
