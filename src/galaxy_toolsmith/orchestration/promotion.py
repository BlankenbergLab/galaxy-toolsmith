from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class PromotionPolicy:
    min_generation_success_rate: float = 0.95
    min_xml_well_formed_rate: float = 0.95
    max_unknown_datatype_rate: float = 0.10
    require_xsd_pass: bool = False
    require_planemo_pass: bool = False
    require_planemo_test_pass: bool = False
    baseline_tolerance: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_PROMOTION_POLICIES = {
    "development": PromotionPolicy(
        min_generation_success_rate=0.85,
        min_xml_well_formed_rate=0.90,
        max_unknown_datatype_rate=0.25,
        require_xsd_pass=False,
        require_planemo_pass=False,
        require_planemo_test_pass=False,
        baseline_tolerance=0.05,
    ),
    "staging": PromotionPolicy(
        min_generation_success_rate=0.95,
        min_xml_well_formed_rate=0.95,
        max_unknown_datatype_rate=0.10,
        require_xsd_pass=False,
        require_planemo_pass=False,
        require_planemo_test_pass=False,
        baseline_tolerance=0.02,
    ),
    "production": PromotionPolicy(
        min_generation_success_rate=0.98,
        min_xml_well_formed_rate=0.99,
        max_unknown_datatype_rate=0.02,
        require_xsd_pass=True,
        require_planemo_pass=True,
        require_planemo_test_pass=False,
        baseline_tolerance=0.0,
    ),
}


@dataclass(frozen=True)
class PromotionDecision:
    created_at: str
    promote: bool
    candidate_summary_path: str
    candidate_eval_path: str
    baseline_summary_path: str
    baseline_eval_path: str
    metrics: dict
    policy: dict
    reasons: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_default_promotion_policies(path: Path) -> Path:
    payload = {
        "policies": {
            name: policy.to_dict()
            for name, policy in DEFAULT_PROMOTION_POLICIES.items()
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_promotion_policy(path: Path, name: str) -> PromotionPolicy:
    data = _load_json(path)
    policies = data.get("policies", {})
    if name not in policies:
        raise ValueError(f"Promotion policy '{name}' not found in {path}")
    selected = policies[name]
    return PromotionPolicy(
        min_generation_success_rate=float(selected.get("min_generation_success_rate", 0.95)),
        min_xml_well_formed_rate=float(selected.get("min_xml_well_formed_rate", 0.95)),
        max_unknown_datatype_rate=float(selected.get("max_unknown_datatype_rate", 0.10)),
        require_xsd_pass=bool(selected.get("require_xsd_pass", False)),
        require_planemo_pass=bool(selected.get("require_planemo_pass", False)),
        require_planemo_test_pass=bool(selected.get("require_planemo_test_pass", False)),
        baseline_tolerance=float(selected.get("baseline_tolerance", 0.0)),
    )


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _extract_metrics(summary: dict, evaluation: dict) -> dict:
    attempted = int(summary.get("attempted", 0))
    succeeded = int(summary.get("succeeded", 0))
    failed = int(summary.get("failed", 0))

    total_wrappers = int(evaluation.get("total_wrappers", 0))
    xml_well_formed = int(evaluation.get("xml_well_formed_count", 0))
    unknown_dtype_wrappers = int(evaluation.get("wrappers_with_unknown_datatypes", 0))

    return {
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
        "generation_success_rate": _rate(succeeded, attempted),
        "total_wrappers": total_wrappers,
        "xml_well_formed_count": xml_well_formed,
        "xml_well_formed_rate": _rate(xml_well_formed, total_wrappers),
        "wrappers_with_unknown_datatypes": unknown_dtype_wrappers,
        "unknown_datatype_rate": _rate(unknown_dtype_wrappers, total_wrappers),
        "xsd_status": str(evaluation.get("xsd_status", "not_run")),
        "planemo_status": str(evaluation.get("planemo_status", "not_run")),
        "planemo_test_status": str(evaluation.get("planemo_test_status", "not_run")),
    }


def decide_promotion(
    candidate_summary_path: Path,
    policy: PromotionPolicy,
    baseline_summary_path: Path | None = None,
) -> PromotionDecision:
    candidate_summary = _load_json(candidate_summary_path)
    candidate_eval_path = Path(candidate_summary["evaluation_report_path"]).resolve()
    candidate_eval = _load_json(candidate_eval_path)
    candidate_metrics = _extract_metrics(candidate_summary, candidate_eval)

    baseline_summary_ref = ""
    baseline_eval_ref = ""
    baseline_metrics = None
    if baseline_summary_path is not None:
        baseline_summary = _load_json(baseline_summary_path)
        baseline_eval_path = Path(baseline_summary["evaluation_report_path"]).resolve()
        baseline_eval = _load_json(baseline_eval_path)
        baseline_metrics = _extract_metrics(baseline_summary, baseline_eval)
        baseline_summary_ref = str(baseline_summary_path.resolve())
        baseline_eval_ref = str(baseline_eval_path)

    reasons: list[str] = []
    if candidate_metrics["generation_success_rate"] < policy.min_generation_success_rate:
        reasons.append(
            f"Generation success rate {candidate_metrics['generation_success_rate']:.3f} is below minimum {policy.min_generation_success_rate:.3f}."
        )
    if candidate_metrics["xml_well_formed_rate"] < policy.min_xml_well_formed_rate:
        reasons.append(
            f"XML well-formed rate {candidate_metrics['xml_well_formed_rate']:.3f} is below minimum {policy.min_xml_well_formed_rate:.3f}."
        )
    if candidate_metrics["unknown_datatype_rate"] > policy.max_unknown_datatype_rate:
        reasons.append(
            f"Unknown datatype rate {candidate_metrics['unknown_datatype_rate']:.3f} exceeds maximum {policy.max_unknown_datatype_rate:.3f}."
        )
    if policy.require_xsd_pass and candidate_metrics["xsd_status"] != "passed":
        reasons.append("XSD pass is required but candidate xsd_status is not 'passed'.")
    if policy.require_planemo_pass and candidate_metrics["planemo_status"] != "passed":
        reasons.append("Planemo pass is required but candidate planemo_status is not 'passed'.")
    if (
        policy.require_planemo_test_pass
        and candidate_metrics["planemo_test_status"] != "passed"
    ):
        reasons.append(
            "Planemo test pass is required but candidate planemo_test_status is not 'passed'."
        )

    if baseline_metrics is not None:
        if candidate_metrics["generation_success_rate"] + policy.baseline_tolerance < baseline_metrics["generation_success_rate"]:
            reasons.append("Generation success rate regressed versus baseline beyond tolerance.")
        if candidate_metrics["xml_well_formed_rate"] + policy.baseline_tolerance < baseline_metrics["xml_well_formed_rate"]:
            reasons.append("XML well-formed rate regressed versus baseline beyond tolerance.")
        if candidate_metrics["unknown_datatype_rate"] - policy.baseline_tolerance > baseline_metrics["unknown_datatype_rate"]:
            reasons.append("Unknown datatype rate worsened versus baseline beyond tolerance.")

    return PromotionDecision(
        created_at=utc_now_iso(),
        promote=len(reasons) == 0,
        candidate_summary_path=str(candidate_summary_path.resolve()),
        candidate_eval_path=str(candidate_eval_path),
        baseline_summary_path=baseline_summary_ref,
        baseline_eval_path=baseline_eval_ref,
        metrics={
            "candidate": candidate_metrics,
            "baseline": baseline_metrics,
        },
        policy=policy.to_dict(),
        reasons=reasons,
    )
