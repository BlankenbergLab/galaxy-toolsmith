from __future__ import annotations

import json
from pathlib import Path

from galaxy_toolsmith.orchestration.promotion import PromotionPolicy, decide_promotion


def test_promotion_can_require_planemo_test_pass(tmp_path: Path) -> None:
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "total_wrappers": 1,
                "xml_well_formed_count": 1,
                "wrappers_with_unknown_datatypes": 0,
                "xsd_status": "passed",
                "planemo_status": "passed",
                "planemo_test_status": "failed",
            }
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "benchmark.json"
    summary_path.write_text(
        json.dumps(
            {
                "attempted": 1,
                "succeeded": 1,
                "failed": 0,
                "evaluation_report_path": str(evaluation_path),
            }
        ),
        encoding="utf-8",
    )

    decision = decide_promotion(
        candidate_summary_path=summary_path,
        policy=PromotionPolicy(require_planemo_test_pass=True),
    )

    assert decision.promote is False
    assert decision.metrics["candidate"]["planemo_test_status"] == "failed"
    assert any("Planemo test pass is required" in reason for reason in decision.reasons)
