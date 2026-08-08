# nmr/campaign.py
"""Campaign orchestration: deterministic trial-lineage logs for research fleets.

A campaign is a named batch of experiment configs whose runs share a
hypothesis. The registry stores per-run state but not per-hypothesis lineage;
this module provides that attribution schema. All writes are atomic
(temp + fsync + os.replace) per AGENTS.md §9. No wall-clock fields are stored
in the log (canonical-determinism friendly; file mtime carries chronology).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from nmr._atomicio import atomic_write_text

__all__ = [
    "CampaignConfig",
    "CampaignRun",
    "CampaignLog",
    "campaign_id",
    "build_campaign_log",
    "write_campaign_log",
]

_VALID_STATUSES = ("recorded", "skipped", "error")


@dataclass(frozen=True)
class CampaignConfig:
    path: str
    sha256: str


@dataclass(frozen=True)
class CampaignRun:
    config_path: str
    run_id: str | None
    status: str
    error: str | None = None


@dataclass(frozen=True)
class CampaignLog:
    campaign_id: str
    name: str
    configs: tuple[CampaignConfig, ...]
    runs: tuple[CampaignRun, ...]

    def to_payload(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "name": self.name,
            "configs": [
                {"path": c.path, "sha256": c.sha256} for c in self.configs
            ],
            "runs": [
                {
                    "config_path": r.config_path,
                    "run_id": r.run_id,
                    "status": r.status,
                    "error": r.error,
                }
                for r in self.runs
            ],
        }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def campaign_id(name: str, config_paths: Sequence[str | Path]) -> str:
    """Deterministic, path-independent campaign id (64-char hex).

    Hashes the name plus the per-file content SHA256 digests in the given
    order (order-sensitive), so moving or renaming config files does not
    change the campaign identity.
    """
    if not name:
        raise ValueError("campaign name must be non-empty")
    if not config_paths:
        raise ValueError("campaign requires at least one config path")
    digests = [
        _file_sha256(Path(path)) for path in config_paths
    ]
    payload = json.dumps(
        {"name": name, "configs": digests}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_campaign_log(
    name: str,
    config_paths: Sequence[str | Path],
    runs: Sequence[CampaignRun],
) -> CampaignLog:
    """Validate and assemble a :class:`CampaignLog`."""
    if not name:
        raise ValueError("campaign name must be non-empty")
    if not config_paths:
        raise ValueError("config_paths must contain at least one path")
    configs: list[CampaignConfig] = []
    for path in config_paths:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"config file not found: {resolved}")
        configs.append(
            CampaignConfig(path=str(path), sha256=_file_sha256(resolved))
        )
    for run in runs:
        if run.status not in _VALID_STATUSES:
            raise ValueError(
                f"run status {run.status!r} not in {_VALID_STATUSES}"
            )
        if run.status != "error" and run.run_id is None:
            raise ValueError(
                "non-error campaign runs must carry a run_id"
            )
    return CampaignLog(
        campaign_id=campaign_id(name, config_paths),
        name=name,
        configs=tuple(configs),
        runs=tuple(runs),
    )


def write_campaign_log(
    log: CampaignLog, campaigns_dir: str | Path
) -> Path:
    """Write ``log`` atomically to ``campaigns_dir/{campaign_id}.json``."""
    out_dir = Path(campaigns_dir)
    target = out_dir / f"{log.campaign_id}.json"
    atomic_write_text(
        target,
        json.dumps(log.to_payload(), indent=2, sort_keys=True),
    )
    return target
