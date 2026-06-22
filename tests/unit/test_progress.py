from __future__ import annotations

from galaxy_toolsmith.runtime.progress import make_progress_snapshot


def test_progress_snapshot_with_eta() -> None:
    snapshot = make_progress_snapshot(
        started_at="2026-01-01T00:00:00+00:00",
        now_at="2026-01-01T00:01:00+00:00",
        completed_units=30,
        total_units=60,
    )
    assert snapshot.elapsed_seconds == 60.0
    assert snapshot.units_per_second > 0
    assert snapshot.eta_seconds is not None
    assert snapshot.eta_timestamp is not None


def test_progress_snapshot_unknown_total_omits_eta() -> None:
    snapshot = make_progress_snapshot(
        started_at="2026-01-01T00:00:00+00:00",
        now_at="2026-01-01T00:01:00+00:00",
        completed_units=30,
        total_units=None,
    )
    assert snapshot.eta_seconds is None
    assert snapshot.eta_timestamp is None


def test_progress_snapshot_zero_rate_omits_eta() -> None:
    snapshot = make_progress_snapshot(
        started_at="2026-01-01T00:00:00+00:00",
        now_at="2026-01-01T00:00:00+00:00",
        completed_units=0,
        total_units=10,
    )
    assert snapshot.units_per_second == 0.0
    assert snapshot.eta_seconds is None
