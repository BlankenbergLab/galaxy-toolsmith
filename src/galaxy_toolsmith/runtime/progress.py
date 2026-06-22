from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


@dataclass(frozen=True)
class ProgressSnapshot:
    started_at: str
    now_at: str
    completed_units: int
    total_units: int | None
    elapsed_seconds: float
    units_per_second: float
    eta_seconds: float | None
    eta_timestamp: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def make_progress_snapshot(
    *,
    started_at: str,
    completed_units: int,
    total_units: int | None,
    now_at: str | None = None,
) -> ProgressSnapshot:
    now_iso = now_at or utc_now_iso()
    start = parse_iso(started_at)
    now = parse_iso(now_iso)
    elapsed = max(0.0, (now - start).total_seconds())
    rate = 0.0 if elapsed <= 0 else float(completed_units) / elapsed
    eta_s: float | None = None
    eta_ts: str | None = None
    if total_units is not None and total_units >= 0 and completed_units < total_units and rate > 0:
        remaining = max(0, int(total_units - completed_units))
        eta_s = float(remaining) / rate
        eta_ts = (now + timedelta(seconds=eta_s)).isoformat()
    return ProgressSnapshot(
        started_at=started_at,
        now_at=now_iso,
        completed_units=int(completed_units),
        total_units=int(total_units) if total_units is not None else None,
        elapsed_seconds=elapsed,
        units_per_second=rate,
        eta_seconds=eta_s,
        eta_timestamp=eta_ts,
    )
