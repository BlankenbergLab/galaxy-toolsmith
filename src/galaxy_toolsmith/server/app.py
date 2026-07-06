from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import UTC, datetime
from pathlib import Path

from galaxy_toolsmith import __version__
from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.inference.artifacts import format_cli_value, normalize_artifact_format
from galaxy_toolsmith.inference.generation import generate_xml_from_content
from galaxy_toolsmith.inference.prompt_context import DEFAULT_MAX_PROMPT_HELP_CHARS
from galaxy_toolsmith.inference.repository import build_tool_shed_metadata
from galaxy_toolsmith.inference.suite import generate_suite_from_content, plan_suite_from_content
from galaxy_toolsmith.orchestration.distributed import (
    claim_task,
    complete_task,
    create_training_job,
    get_job,
    get_job_progress,
    get_tasks_for_job,
    heartbeat_task,
    list_job_artifacts,
    list_jobs,
    resolve_artifact_path,
)
from galaxy_toolsmith.orchestration.training import get_local_training_run, list_local_training_runs
from galaxy_toolsmith.runtime.run_registry import (
    create_monitor_run_tracker,
    get_monitor_run,
    list_monitor_runs,
)
from galaxy_toolsmith.runtime.status import emit_status, resolve_status_log_path


def datetime_now_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")


def _variant_ids(paths: WorkspacePaths) -> list[str]:
    variants_dir = paths.models_root / "variants"
    if not variants_dir.exists():
        return []
    return sorted(path.stem.removesuffix(".manifest") for path in variants_dir.glob("*.manifest.json"))


def create_app(
    repo_root: Path,
    provider: str,
    model: str,
    model_variant: str,
    temperature: float,
    max_tokens: int,
    auth_tokens: list[str],
    require_generate_auth: bool,
    ollama_context_tokens: int | None = None,
    allow_stub_local: bool = False,
    status_log_path: Path | None = None,
    max_prompt_help_chars: int = DEFAULT_MAX_PROMPT_HELP_CHARS,
):
    try:
        from fastapi import FastAPI, Header, HTTPException, Query
        from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
    except Exception as error:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "FastAPI server requires optional server deps. Install with: pip install -e '.[server]'"
        ) from error

    app = FastAPI(title="Galaxy Toolsmith Server", version=__version__)
    paths = WorkspacePaths.from_repo_root(repo_root)
    paths.create_directories()

    token_set = {token for token in auth_tokens if token}

    def _emit_status(payload: dict) -> None:
        emit_status(payload, status_log_path=status_log_path)

    def _authorize(
        authorization: str | None,
        *,
        required: bool,
    ) -> None:
        if not required:
            return
        if not token_set:
            return
        if not authorization:
            raise HTTPException(status_code=401, detail="unauthorized")
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="unauthorized")
        if authorization.split(" ", 1)[1] not in token_set:
            raise HTTPException(status_code=401, detail="unauthorized")

    def _bundle_artifacts(output_dir: Path) -> list[dict]:
        artifacts: list[dict] = []
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue
            try:
                relative_path = path.relative_to(output_dir).as_posix()
            except ValueError:
                continue
            if relative_path.startswith(".gtsm/raw/"):
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            artifacts.append(
                {
                    "relative_path": relative_path,
                    "content": content,
                    "size_bytes": len(content.encode("utf-8", errors="replace")),
                }
            )
        return artifacts

    @app.get("/")
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/monitor")

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "provider": provider,
            "model": model,
            "model_variant": model_variant,
            "auth_enabled": bool(token_set),
            "generate_auth_required": bool(token_set) and require_generate_auth,
        }

    @app.get("/variants")
    async def variants() -> dict:
        return {"variants": _variant_ids(paths)}

    @app.get("/runs")
    async def monitor_runs(
        authorization: str | None = Header(default=None, alias="Authorization"),
        kind: str = Query(default=""),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict:
        _authorize(authorization, required=bool(token_set))
        _emit_status({"status": "server-monitor-runs-requested", "kind": kind, "limit": limit})
        return list_monitor_runs(paths, kind=kind, limit=limit)

    @app.get("/runs/{run_id}")
    async def monitor_run_detail(
        run_id: str,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict:
        _authorize(authorization, required=bool(token_set))
        try:
            _emit_status({"status": "server-monitor-run-requested", "run_id": run_id})
            return get_monitor_run(paths, run_id)
        except Exception as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/monitor", response_class=HTMLResponse)
    async def monitor() -> str:
        return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Galaxy Toolsmith Monitor</title>
  <style>
    body { font-family: sans-serif; margin: 1rem 1.5rem; color: #111; }
    h1 { margin: 0 0 0.5rem; }
    .row { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; margin: 0.5rem 0 1rem; }
    input, button { font-size: 0.95rem; padding: 0.35rem 0.5rem; }
    table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
    th, td { border: 1px solid #ddd; text-align: left; padding: 0.35rem 0.45rem; }
    th { background: #f5f5f5; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  </style>
</head>
<body>
  <h1>Galaxy Toolsmith Monitor</h1>
  <div class="row">
    <label>Bearer token <input id="token" type="password" size="42" placeholder="optional" /></label>
    <label>Poll seconds <input id="poll" type="number" min="1" value="3" style="width:4.5rem;" /></label>
    <button id="refresh">Refresh now</button>
    <span id="status"></span>
  </div>
  <div class="row">
    <strong>Summary:</strong>
    <span id="summary">loading...</span>
  </div>
  <h2>Local Direct Runs</h2>
  <table>
    <thead>
      <tr>
        <th>Run ID</th>
        <th>Status</th>
        <th>Backend</th>
        <th>Profile</th>
        <th>Progress</th>
        <th>PID</th>
        <th>Process</th>
        <th>Updated</th>
      </tr>
    </thead>
    <tbody id="local-runs"></tbody>
  </table>
  <h2>Server Jobs</h2>
  <table>
    <thead>
      <tr>
        <th>Job ID</th>
        <th>Status</th>
        <th>Profile</th>
        <th>Progress</th>
        <th>Elapsed (s)</th>
        <th>Rate</th>
        <th>ETA (s)</th>
        <th>Updated</th>
      </tr>
    </thead>
    <tbody id="jobs"></tbody>
  </table>
  <h2>Benchmark Runs</h2>
  <table>
    <thead>
      <tr>
        <th>Run ID</th>
        <th>Status</th>
        <th>Progress</th>
        <th>Provider</th>
        <th>Model Variant</th>
        <th>Elapsed (s)</th>
        <th>ETA (s)</th>
        <th>Environment</th>
        <th>Updated</th>
      </tr>
    </thead>
    <tbody id="benchmark-runs"></tbody>
  </table>
  <h2>Inference Runs</h2>
  <table>
    <thead>
      <tr>
        <th>Run ID</th>
        <th>Status</th>
        <th>Tool</th>
        <th>Provider</th>
        <th>Model Variant</th>
        <th>Validation</th>
        <th>Environment</th>
        <th>Updated</th>
      </tr>
    </thead>
    <tbody id="inference-runs"></tbody>
  </table>
  <h2>Data And Diagnostics Runs</h2>
  <table>
    <thead>
      <tr>
        <th>Run ID</th>
        <th>Kind</th>
        <th>Status</th>
        <th>Summary</th>
        <th>Environment</th>
        <th>Updated</th>
      </tr>
    </thead>
    <tbody id="data-runs"></tbody>
  </table>
  <h2>Export Runs</h2>
  <table>
    <thead>
      <tr>
        <th>Run ID</th>
        <th>Status</th>
        <th>Variant</th>
        <th>Summary</th>
        <th>Environment</th>
        <th>Updated</th>
      </tr>
    </thead>
    <tbody id="export-runs"></tbody>
  </table>
  <h2>Server Runs</h2>
  <table>
    <thead>
      <tr>
        <th>Run ID</th>
        <th>Status</th>
        <th>Summary</th>
        <th>Environment</th>
        <th>Updated</th>
      </tr>
    </thead>
    <tbody id="server-runs"></tbody>
  </table>
  <script>
    const tokenInput = document.getElementById("token");
    const pollInput = document.getElementById("poll");
    const statusEl = document.getElementById("status");
    const summaryEl = document.getElementById("summary");
    const jobsEl = document.getElementById("jobs");
    const localRunsEl = document.getElementById("local-runs");
    const benchmarkRunsEl = document.getElementById("benchmark-runs");
    const inferenceRunsEl = document.getElementById("inference-runs");
    const dataRunsEl = document.getElementById("data-runs");
    const exportRunsEl = document.getElementById("export-runs");
    const serverRunsEl = document.getElementById("server-runs");
    const refreshBtn = document.getElementById("refresh");
    const params = new URLSearchParams(window.location.search);
    if (params.get("token")) tokenInput.value = params.get("token");

    function authHeaders() {
      const token = tokenInput.value.trim();
      return token ? { "Authorization": `Bearer ${token}` } : {};
    }

    function fmt(v, digits=2) {
      if (v === null || v === undefined || Number.isNaN(Number(v))) return "";
      return Number(v).toFixed(digits);
    }

    function renderJobs(jobs) {
      jobsEl.innerHTML = "";
      for (const item of jobs) {
        const tr = document.createElement("tr");
        const p = item.progress || {};
        const progressText = `${p.completed_units ?? ""}/${p.total_units ?? "?"}`;
        tr.innerHTML = `
          <td class="mono">${item.job?.job_id ?? ""}</td>
          <td>${item.job?.status ?? ""}</td>
          <td class="mono">${item.job?.payload?.profile_name ?? ""}</td>
          <td>${progressText}</td>
          <td>${fmt(p.elapsed_seconds, 1)}</td>
          <td>${fmt(p.units_per_second, 3)}</td>
          <td>${fmt(p.eta_seconds, 1)}</td>
          <td class="mono">${item.job?.updated_at ?? ""}</td>
        `;
        jobsEl.appendChild(tr);
      }
      if (!jobs.length) {
        const tr = document.createElement("tr");
        tr.innerHTML = '<td colspan="8">No jobs found.</td>';
        jobsEl.appendChild(tr);
      }
    }

    function renderLocalRuns(runs) {
      localRunsEl.innerHTML = "";
      for (const item of runs) {
        const tr = document.createElement("tr");
        const p = item.progress || {};
        const progressText = `${p.completed_units ?? ""}/${p.total_units ?? "?"}`;
        tr.innerHTML = `
          <td class="mono">${item.run?.run_id ?? ""}</td>
          <td>${item.status ?? ""}</td>
          <td class="mono">${item.metrics?.backend_impl ?? item.run?.backend ?? ""}</td>
          <td class="mono">${item.run?.profile_name ?? ""}</td>
          <td>${progressText}</td>
          <td class="mono">${item.process?.pid ?? ""}</td>
          <td>${item.process?.running ? "running" : ""}</td>
          <td class="mono">${item.run?.created_at ?? ""}</td>
        `;
        localRunsEl.appendChild(tr);
      }
      if (!runs.length) {
        const tr = document.createElement("tr");
        tr.innerHTML = '<td colspan="8">No local direct runs found.</td>';
        localRunsEl.appendChild(tr);
      }
    }

    function envText(item) {
      const env = item.environment || {};
      const versions = env.package_versions || {};
      const runtime = env.runtime_capabilities || {};
      const vars = env.environment_variables || {};
      const tags = [];
      if (runtime.recommended_backend) tags.push(runtime.recommended_backend);
      if (vars.CUDA_VISIBLE_DEVICES) tags.push(`gpu=${vars.CUDA_VISIBLE_DEVICES}`);
      if (versions.torch) tags.push(`torch=${versions.torch}`);
      if (versions.transformers) tags.push(`transformers=${versions.transformers}`);
      return tags.join(", ");
    }

    function validationText(item) {
      const validation = item.summary?.validation || {};
      if (!Object.keys(validation).length) return "";
      return `${validation.xml_well_formed ? "xml" : "bad-xml"}/${validation.root_is_tool ? "tool" : "non-tool"}`;
    }

    function shortSummary(item) {
      const s = item.summary || {};
      const parts = [];
      for (const key of ["attempted", "succeeded", "failed", "processed_now", "total_records", "status"]) {
        if (s[key] !== undefined && s[key] !== "") parts.push(`${key}=${s[key]}`);
      }
      return parts.join(", ");
    }

    function renderTrackedRows(tbody, runs, columns, emptyText) {
      tbody.innerHTML = "";
      for (const item of runs) {
        const tr = document.createElement("tr");
        tr.innerHTML = columns(item).map((value) => `<td class="mono">${value ?? ""}</td>`).join("");
        tbody.appendChild(tr);
      }
      if (!runs.length) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="9">${emptyText}</td>`;
        tbody.appendChild(tr);
      }
    }

    function renderTrackedRuns(runs) {
      const byKind = (kind) => runs.filter((item) => item.kind === kind);
      renderTrackedRows(
        benchmarkRunsEl,
        byKind("benchmark"),
        (item) => {
          const p = item.progress || {};
          return [
            item.run_id,
            item.status,
            `${p.completed_units ?? ""}/${p.total_units ?? "?"}`,
            item.inputs?.provider || "",
            item.inputs?.model_variant || "",
            fmt(p.elapsed_seconds, 1),
            fmt(p.eta_seconds, 1),
            envText(item),
            item.updated_at,
          ];
        },
        "No benchmark runs found.",
      );
      renderTrackedRows(
        inferenceRunsEl,
        byKind("inference"),
        (item) => [
          item.run_id,
          item.status,
          item.inputs?.tool_name || "",
          item.inputs?.provider || "",
          item.inputs?.model_variant || "",
          validationText(item),
          envText(item),
          item.updated_at,
        ],
        "No inference runs found.",
      );
      renderTrackedRows(
        dataRunsEl,
        runs.filter((item) => ["extract", "diagnostics", "evaluation", "promotion"].includes(item.kind)),
        (item) => [item.run_id, item.kind, item.status, shortSummary(item), envText(item), item.updated_at],
        "No data or diagnostics runs found.",
      );
      renderTrackedRows(
        exportRunsEl,
        byKind("export"),
        (item) => [
          item.run_id,
          item.status,
          item.inputs?.variant_id || "",
          shortSummary(item),
          envText(item),
          item.updated_at,
        ],
        "No export runs found.",
      );
      renderTrackedRows(
        serverRunsEl,
        byKind("server"),
        (item) => [item.run_id, item.status, shortSummary(item), envText(item), item.updated_at],
        "No server runs found.",
      );
    }

    async function refresh() {
      statusEl.textContent = "loading...";
      try {
        const resp = await fetch("/train/jobs?limit=50", { headers: authHeaders() });
        const payload = await resp.json();
        if (!resp.ok) {
          throw new Error(payload.detail || `HTTP ${resp.status}`);
        }
        const localResp = await fetch("/train/local-runs?limit=50", { headers: authHeaders() });
        const localPayload = await localResp.json();
        if (!localResp.ok) {
          throw new Error(localPayload.detail || `HTTP ${localResp.status}`);
        }
        const runsResp = await fetch("/runs?limit=50", { headers: authHeaders() });
        const runsPayload = await runsResp.json();
        if (!runsResp.ok) {
          throw new Error(runsPayload.detail || `HTTP ${runsResp.status}`);
        }
        const s = payload.summary || {};
        const l = localPayload.summary || {};
        const r = runsPayload.summary || {};
        summaryEl.textContent = `server total=${s.total ?? 0}, queued=${s.queued ?? 0}, running=${s.running ?? 0}, completed=${s.completed ?? 0}, failed=${s.failed ?? 0}; local total=${l.total ?? 0}, running=${l.running ?? 0}, completed=${l.completed ?? 0}, failed=${l.failed ?? 0}; tracked total=${r.total ?? 0}, running=${r.running ?? 0}, completed=${r.completed ?? 0}, failed=${r.failed ?? 0}`;
        renderLocalRuns(localPayload.runs || []);
        renderJobs(payload.jobs || []);
        renderTrackedRuns(runsPayload.runs || []);
        statusEl.textContent = `updated ${new Date().toLocaleTimeString()}`;
      } catch (error) {
        statusEl.textContent = `error: ${String(error.message || error)}`;
      }
    }

    refreshBtn.addEventListener("click", refresh);
    refresh();
    setInterval(() => {
      const seconds = Math.max(1, Number(pollInput.value || 3));
      const intervalMs = seconds * 1000;
      if (!window.__gtsmPollMs || window.__gtsmPollMs !== intervalMs) {
        if (window.__gtsmTimer) clearInterval(window.__gtsmTimer);
        window.__gtsmTimer = setInterval(refresh, intervalMs);
        window.__gtsmPollMs = intervalMs;
      }
    }, 500);
  </script>
</body>
</html>
"""

    @app.post("/generate")
    async def generate(
        payload: dict,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict:
        _authorize(
            authorization,
            required=bool(token_set) and require_generate_auth,
        )

        tool_name = str(payload.get("tool_name", "")).strip()
        tool_id = str(payload.get("tool_id", "")).strip()
        help_text = str(payload.get("help_text", ""))
        source_code = str(payload.get("source_code", ""))
        req_provider = str(payload.get("provider", provider))
        req_model = str(payload.get("model", model))
        req_model_variant = str(payload.get("model_variant", model_variant))
        req_skills_profile = str(payload.get("skills_profile", "default"))
        req_artifact_format = normalize_artifact_format(str(payload.get("artifact_format", "xml")))
        req_temperature = float(payload.get("temperature", temperature))
        req_max_tokens = int(payload.get("max_tokens", max_tokens))
        req_ollama_context_tokens = (
            int(payload["ollama_context_tokens"])
            if payload.get("ollama_context_tokens") is not None
            else ollama_context_tokens
        )
        req_max_prompt_help_chars = int(
            payload.get("max_prompt_help_chars", max_prompt_help_chars)
        )
        req_allow_stub_local = bool(payload.get("allow_stub_local", allow_stub_local))
        include_toolsmith_citation = bool(payload.get("include_toolsmith_citation", True))

        if not tool_name:
            raise HTTPException(status_code=400, detail="tool_name_required")

        tracker = create_monitor_run_tracker(
            paths,
            kind="inference",
            command=["server", "/generate"],
            inputs={
                "tool_name": tool_name,
                "tool_id": tool_id,
                "provider": req_provider,
                "model": req_model,
                "model_variant": req_model_variant,
                "skills_profile": req_skills_profile,
                "artifact_format": format_cli_value(req_artifact_format),
                "temperature": req_temperature,
                "max_tokens": req_max_tokens,
                "ollama_context_tokens": req_ollama_context_tokens,
                "max_prompt_help_chars": req_max_prompt_help_chars,
                "include_toolsmith_citation": include_toolsmith_citation,
            },
        )
        try:
            result = generate_xml_from_content(
                tool_name=tool_name,
                help_text=help_text,
                source_code=source_code,
                provider_name=req_provider,
                model_variant=req_model_variant,
                model=req_model,
                temperature=req_temperature,
                max_tokens=req_max_tokens,
                ollama_context_tokens=req_ollama_context_tokens,
                skills_profile=req_skills_profile,
                paths=paths,
                allow_stub_local=req_allow_stub_local,
                max_prompt_help_chars=req_max_prompt_help_chars,
                artifact_format=req_artifact_format,
                tool_id=tool_id,
                tool_display_name=tool_name,
                include_toolsmith_citation=include_toolsmith_citation,
            )
            tracker.complete(
                summary={
                    "provider": result.get("provider", req_provider),
                    "model_variant": result.get("model_variant", req_model_variant),
                    "validation": result.get("validation", {}),
                }
            )
            return result
        except Exception as error:
            tracker.fail(error)
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/suite/plan")
    async def suite_plan(
        payload: dict,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict:
        _authorize(
            authorization,
            required=bool(token_set) and require_generate_auth,
        )
        tool_name = str(payload.get("tool_name", "")).strip()
        if not tool_name:
            raise HTTPException(status_code=400, detail="tool_name_required")
        plan = plan_suite_from_content(
            tool_name=tool_name,
            help_text=str(payload.get("help_text", "")),
            source_code=str(payload.get("source_code", "")),
            max_suite_tools=int(payload.get("max_suite_tools", 8)),
            force_suite=bool(payload.get("force_suite", False)),
        )
        return plan.to_dict()

    @app.post("/generate-suite")
    async def generate_suite_endpoint(
        payload: dict,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict:
        _authorize(
            authorization,
            required=bool(token_set) and require_generate_auth,
        )
        tool_name = str(payload.get("tool_name", "")).strip()
        if not tool_name:
            raise HTTPException(status_code=400, detail="tool_name_required")
        req_provider = str(payload.get("provider", provider))
        req_model = str(payload.get("model", model))
        req_model_variant = str(payload.get("model_variant", model_variant))
        req_skills_profile = str(payload.get("skills_profile", "default"))
        req_temperature = float(payload.get("temperature", temperature))
        req_max_tokens = int(payload.get("max_tokens", max_tokens))
        req_ollama_context_tokens = (
            int(payload["ollama_context_tokens"])
            if payload.get("ollama_context_tokens") is not None
            else ollama_context_tokens
        )
        req_max_prompt_help_chars = int(
            payload.get("max_prompt_help_chars", max_prompt_help_chars)
        )
        req_allow_stub_local = bool(payload.get("allow_stub_local", allow_stub_local))
        include_toolsmith_citation = bool(payload.get("include_toolsmith_citation", True))
        datatype_scaffold = bool(payload.get("datatype_scaffold", True))
        request_id = f"suite-{datetime_now_compact()}"
        output_dir = paths.runs_root / "generation-suite" / request_id / "repository"
        shed = payload.get("shed_metadata", {}) if isinstance(payload.get("shed_metadata"), dict) else {}
        metadata = build_tool_shed_metadata(
            name=str(shed.get("name") or payload.get("shed_name") or f"suite_{tool_name}"),
            owner=str(shed.get("owner") or payload.get("shed_owner") or ""),
            description=str(
                shed.get("description")
                or payload.get("shed_description")
                or f"Generated Galaxy Toolsmith suite for {tool_name}"
            ),
            homepage_url=str(shed.get("homepage_url") or payload.get("shed_homepage_url") or ""),
            remote_repository_url=str(
                shed.get("remote_repository_url")
                or payload.get("shed_remote_repository_url")
                or ""
            ),
            categories=list(shed.get("categories") or payload.get("shed_categories") or []),
            suite=True,
            repositories=[],
        )
        tracker = create_monitor_run_tracker(
            paths,
            kind="inference",
            command=["server", "/generate-suite"],
            inputs={
                "tool_name": tool_name,
                "provider": req_provider,
                "model": req_model,
                "model_variant": req_model_variant,
                "skills_profile": req_skills_profile,
                "temperature": req_temperature,
                "max_tokens": req_max_tokens,
                "ollama_context_tokens": req_ollama_context_tokens,
                "max_prompt_help_chars": req_max_prompt_help_chars,
                "max_suite_tools": int(payload.get("max_suite_tools", 8)),
                "include_toolsmith_citation": include_toolsmith_citation,
                "datatype_scaffold": datatype_scaffold,
            },
            outputs={"output_dir": str(output_dir)},
        )
        try:
            record = generate_suite_from_content(
                paths=paths,
                tool_name=tool_name,
                help_text=str(payload.get("help_text", "")),
                source_code=str(payload.get("source_code", "")),
                output_dir=output_dir,
                provider_name=req_provider,
                model_variant=req_model_variant,
                model=req_model,
                temperature=req_temperature,
                max_tokens=req_max_tokens,
                ollama_context_tokens=req_ollama_context_tokens,
                skills_profile=req_skills_profile,
                allow_stub_local=req_allow_stub_local,
                max_prompt_help_chars=req_max_prompt_help_chars,
                max_suite_tools=int(payload.get("max_suite_tools", 8)),
                generate_sidecars=bool(payload.get("generate_sidecars", True)),
                raw_response_logs=bool(payload.get("raw_response_logs", False)),
                stream_output=bool(payload.get("stream_output", False)),
                repair_invalid_xml=bool(payload.get("repair_invalid_xml", True)),
                shed_metadata=metadata,
                write_shed=not bool(payload.get("no_shed_yml", False)),
                include_toolsmith_citation=include_toolsmith_citation,
                datatype_scaffold=datatype_scaffold,
            )
            result = record.to_dict()
            result["artifacts"] = _bundle_artifacts(output_dir)
            tracker.complete(
                outputs={
                    "output_dir": str(output_dir),
                    "shed_yml_path": result.get("shed_yml_path", ""),
                    "generated_files": result.get("generated_files", []),
                },
                summary={
                    "suite_plan": result.get("suite_plan", {}),
                    "generated_file_count": len(result.get("generated_files", [])),
                },
            )
            return result
        except Exception as error:
            tracker.fail(error)
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/train/jobs")
    async def submit_training_job(
        payload: dict,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict:
        _authorize(authorization, required=bool(token_set))
        profile_name = str(payload.get("profile_name", "")).strip()
        dataset_manifest_path = str(payload.get("dataset_manifest_path", "")).strip()
        corpus_jsonl_path = str(payload.get("corpus_jsonl_path", "")).strip()
        variant_id = str(payload.get("variant_id", "")).strip()
        trainer_command = payload.get("trainer_command", [])
        learning_rate = payload.get("learning_rate")
        training_method = str(payload.get("training_method", "") or "").strip() or None
        if not profile_name:
            raise HTTPException(status_code=400, detail="profile_name_required")
        if not dataset_manifest_path:
            raise HTTPException(status_code=400, detail="dataset_manifest_path_required")
        if not corpus_jsonl_path:
            raise HTTPException(status_code=400, detail="corpus_jsonl_path_required")
        if not isinstance(trainer_command, list):
            raise HTTPException(status_code=400, detail="trainer_command_must_be_list")
        job = create_training_job(
            paths=paths,
            profile_name=profile_name,
            dataset_manifest_path=dataset_manifest_path,
            corpus_jsonl_path=corpus_jsonl_path,
            variant_id=variant_id,
            trainer_command=[str(item) for item in trainer_command],
            learning_rate=float(learning_rate) if learning_rate is not None else None,
            training_method=training_method,
        )
        _emit_status(
            {
                "status": "server-train-job-submitted",
                "job_id": job.job_id,
                "profile_name": profile_name,
                "variant_id": variant_id,
            }
        )
        return json.loads(job.to_json())

    @app.get("/train/jobs/{job_id}")
    async def training_job_status(
        job_id: str,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict:
        _authorize(authorization, required=bool(token_set))
        try:
            _emit_status({"status": "server-train-job-status-requested", "job_id": job_id})
            return {
                "job": get_job(paths, job_id),
                "tasks": get_tasks_for_job(paths, job_id),
                "progress": get_job_progress(paths, job_id),
            }
        except Exception as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/train/jobs")
    async def training_jobs_status(
        authorization: str | None = Header(default=None, alias="Authorization"),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict:
        _authorize(authorization, required=bool(token_set))
        _emit_status({"status": "server-train-jobs-list-requested", "limit": limit})
        rows: list[dict] = []
        summary = {"total": 0, "queued": 0, "running": 0, "completed": 0, "failed": 0}
        for job in list_jobs(paths, limit=limit):
            job_id = str(job.get("job_id", "")).strip()
            status = str(job.get("status", "")).strip().lower()
            if status in summary:
                summary[status] += 1
            summary["total"] += 1
            progress = get_job_progress(paths, job_id) if job_id else {}
            tasks = get_tasks_for_job(paths, job_id) if job_id else []
            latest_task = tasks[-1] if tasks else None
            rows.append(
                {
                    "job": job,
                    "progress": progress,
                    "latest_task": latest_task,
                }
            )
        return {"summary": summary, "jobs": rows}

    @app.get("/train/local-runs")
    async def local_training_runs_status(
        authorization: str | None = Header(default=None, alias="Authorization"),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict:
        _authorize(authorization, required=bool(token_set))
        _emit_status({"status": "server-local-training-runs-requested", "limit": limit})
        return list_local_training_runs(paths, limit=limit)

    @app.get("/train/local-runs/{run_id}")
    async def local_training_run_status(
        run_id: str,
        authorization: str | None = Header(default=None, alias="Authorization"),
        tail: int = Query(default=80, ge=0, le=1000),
    ) -> dict:
        _authorize(authorization, required=bool(token_set))
        try:
            _emit_status({"status": "server-local-training-run-requested", "run_id": run_id})
            return get_local_training_run(paths, run_id, tail_lines=tail)
        except Exception as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/train/tasks/claim")
    async def training_task_claim(
        payload: dict,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict:
        _authorize(authorization, required=bool(token_set))
        worker_id = str(payload.get("worker_id", "")).strip() or socket.gethostname()
        lease_seconds = int(payload.get("lease_seconds", 180))
        task = claim_task(paths, worker_id=worker_id, lease_seconds=lease_seconds)
        _emit_status(
            {
                "status": "server-train-task-claimed",
                "worker_id": worker_id,
                "task_id": task.get("task_id") if task else "",
                "job_id": task.get("job_id") if task else "",
                "has_task": bool(task),
            }
        )
        return {"task": task}

    @app.post("/train/tasks/{task_id}/heartbeat")
    async def training_task_heartbeat(
        task_id: str,
        payload: dict,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict:
        _authorize(authorization, required=bool(token_set))
        worker_id = str(payload.get("worker_id", "")).strip() or socket.gethostname()
        lease_seconds = int(payload.get("lease_seconds", 180))
        try:
            task = heartbeat_task(
                paths,
                task_id=task_id,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                progress=payload.get("progress"),
            )
            _emit_status(
                {
                    "status": "server-train-task-heartbeat",
                    "worker_id": worker_id,
                    "task_id": task_id,
                    "job_id": task.get("job_id", ""),
                }
            )
            return {"task": task}
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/train/tasks/{task_id}/complete")
    async def training_task_complete(
        task_id: str,
        payload: dict,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict:
        _authorize(authorization, required=bool(token_set))
        worker_id = str(payload.get("worker_id", "")).strip() or socket.gethostname()
        success = bool(payload.get("success", False))
        result = payload.get("result", {})
        error = str(payload.get("error", "")).strip()
        if not isinstance(result, dict):
            raise HTTPException(status_code=400, detail="result_must_be_object")
        try:
            job = complete_task(
                paths,
                task_id=task_id,
                worker_id=worker_id,
                success=success,
                result=result,
                error=error,
            )
            _emit_status(
                {
                    "status": "server-train-task-completed",
                    "worker_id": worker_id,
                    "task_id": task_id,
                    "job_id": job.get("job_id", ""),
                    "success": success,
                    "job_status": job.get("status", ""),
                    "error": error,
                }
            )
            return {"job": job}
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/train/artifacts/{job_id}")
    async def training_artifacts_list(
        job_id: str,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict:
        _authorize(authorization, required=bool(token_set))
        try:
            return {"artifacts": list_job_artifacts(paths, job_id)}
        except Exception as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/train/artifacts/{job_id}/download/{artifact_id}")
    async def training_artifacts_download(
        job_id: str,
        artifact_id: str,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ):
        _authorize(authorization, required=bool(token_set))
        try:
            artifact_path = resolve_artifact_path(paths, job_id, artifact_id)
        except Exception as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return FileResponse(path=artifact_path)

    return app


def serve(
    host: str,
    port: int,
    provider: str,
    model: str,
    model_variant: str,
    temperature: float,
    max_tokens: int,
    auth_tokens: list[str],
    require_generate_auth: bool,
    repo_root: Path,
    allow_stub_local: bool = False,
    status_log_path: Path | None = None,
    max_prompt_help_chars: int = DEFAULT_MAX_PROMPT_HELP_CHARS,
    ollama_context_tokens: int | None = None,
) -> None:
    try:
        import uvicorn
    except Exception as error:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "Uvicorn is required for server mode. Install with: pip install -e '.[server]'"
        ) from error

    app = create_app(
        repo_root=repo_root,
        provider=provider,
        model=model,
        model_variant=model_variant,
        temperature=temperature,
        max_tokens=max_tokens,
        ollama_context_tokens=ollama_context_tokens,
        max_prompt_help_chars=max_prompt_help_chars,
        auth_tokens=auth_tokens,
        require_generate_auth=require_generate_auth,
        allow_stub_local=allow_stub_local,
        status_log_path=status_log_path,
    )
    emit_status(
        {
            "status": "starting",
            "server": "fastapi",
            "host": host,
            "port": port,
            "provider": provider,
            "model": model,
            "model_variant": model_variant,
            "ollama_context_tokens": ollama_context_tokens,
            "max_prompt_help_chars": max_prompt_help_chars,
            "auth_enabled": bool(auth_tokens),
            "generate_auth_required": bool(auth_tokens) and require_generate_auth,
            "allow_stub_local": allow_stub_local,
        },
        status_log_path=status_log_path,
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main() -> int:
    parser = argparse.ArgumentParser(description="Galaxy Toolsmith optional inference server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--provider",
        choices=["local", "openai", "anthropic", "copilot", "ollama"],
        default="local",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--model-variant", default="server-default")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--ollama-context-tokens",
        type=int,
        default=None,
        help=(
            "Ollama runtime context tokens (num_ctx). "
            "Unset uses GTSM_OLLAMA_CONTEXT_TOKENS or Ollama defaults; 0 disables num_ctx."
        ),
    )
    parser.add_argument("--max-prompt-help-chars", type=int, default=DEFAULT_MAX_PROMPT_HELP_CHARS)
    parser.add_argument(
        "--auth-token",
        action="append",
        default=[],
        help="Optional bearer token (repeatable). Empty token list disables auth.",
    )
    parser.add_argument(
        "--auth-tokens-file",
        help="Optional file with one token per line.",
    )
    parser.add_argument(
        "--require-generate-auth",
        action="store_true",
        help="Require auth tokens for /generate when tokens are configured.",
    )
    parser.add_argument(
        "--allow-stub-local",
        action="store_true",
        help="Allow /generate to return canned starter XML for local provider requests with no real local model.",
    )
    parser.add_argument(
        "--status-log",
        default="",
        help="Optional JSONL file for status events (disabled by default).",
    )
    parser.add_argument("--repo-root", default=os.getcwd(), help="Repository root for model variant discovery.")
    args = parser.parse_args()

    tokens = [token for token in args.auth_token if token]
    if args.auth_tokens_file:
        token_file = Path(args.auth_tokens_file).resolve()
        if token_file.exists():
            for line in token_file.read_text(encoding="utf-8").splitlines():
                value = line.strip()
                if value:
                    tokens.append(value)

    serve(
        host=args.host,
        port=args.port,
        provider=args.provider,
        model=args.model,
        model_variant=args.model_variant,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ollama_context_tokens=args.ollama_context_tokens,
        max_prompt_help_chars=args.max_prompt_help_chars,
        auth_tokens=tokens,
        require_generate_auth=args.require_generate_auth,
        allow_stub_local=args.allow_stub_local,
        repo_root=Path(args.repo_root).resolve(),
        status_log_path=resolve_status_log_path(args.status_log),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
