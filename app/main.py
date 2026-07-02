import base64
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlparse

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import google.auth
from google.auth.transport.requests import AuthorizedSession
from google.cloud import run_v2
from google.protobuf import duration_pb2

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
MANIFEST_PATH = DATA_DIR / "manifest.json"
STATE_PATH = DATA_DIR / "state.json"

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
GCP_REGION = os.environ.get("GCP_REGION", "us-central1")
ARTIFACT_REPOSITORY = os.environ.get("ARTIFACT_REPOSITORY", "model-images")
MODEL_LIMIT = int(os.environ.get("MODEL_LIMIT", "25"))
MAX_REPLICATE_PAGES = int(os.environ.get("MAX_REPLICATE_PAGES", "40"))
TARGET_MODEL_FAMILIES = [
    x.strip().lower()
    for x in os.environ.get("TARGET_MODEL_FAMILIES", "sdxl,small-llm").split(",")
    if x.strip()
]

GPU_TYPE = os.environ.get("GPU_TYPE", "nvidia-l4")
GPU_COUNT = int(os.environ.get("GPU_COUNT", "1"))
GPU_CPU = os.environ.get("GPU_CPU", "4")
GPU_MEMORY = os.environ.get("GPU_MEMORY", "16Gi")
GPU_TIMEOUT_SECONDS = int(os.environ.get("GPU_TIMEOUT_SECONDS", "300"))
GPU_MIN_INSTANCES = int(os.environ.get("GPU_MIN_INSTANCES", "0"))
GPU_MAX_INSTANCES = int(os.environ.get("GPU_MAX_INSTANCES", "1"))

app = FastAPI(title="Runner Control")


class BuildRequest(BaseModel):
    force: bool = False


class DeployRequest(BaseModel):
    force: bool = False


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def state() -> Dict[str, Any]:
    return load_json(STATE_PATH, {"models": {}, "updated_at": None})


def save_state(s: Dict[str, Any]) -> None:
    s["updated_at"] = now_iso()
    save_json(STATE_PATH, s)


def headers_replicate() -> Dict[str, str]:
    if not REPLICATE_API_TOKEN:
        raise HTTPException(500, "Missing REPLICATE_API_TOKEN env var.")
    return {
        "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
        "Accept": "application/json",
        "User-Agent": "runner-control",
    }


def headers_github() -> Dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "runner-control",
    }
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def http_get_json(url: str, headers: Dict[str, str]) -> Any:
    r = requests.get(url, headers=headers, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {url}: {r.text[:1000]}")
    return r.json()


def normalize_github_repo(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    url = url.strip()

    m = re.match(r"^git@github\.com:([^/]+)/(.+?)(?:\.git)?$", url)
    if m:
        return f"{m.group(1)}/{m.group(2)}".strip("/").lower()

    try:
        parsed = urlparse(url)
    except Exception:
        return None

    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        return None

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return None

    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return f"{owner}/{repo}".strip("/").lower()


def flatten_for_search(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False).lower()
    except Exception:
        return str(value).lower()


def classify_model(raw: Dict[str, Any], slim: Dict[str, Any]) -> Optional[Dict[str, str]]:
    text = " ".join([
        str(slim.get("model_id") or ""),
        str(slim.get("name") or ""),
        str(slim.get("description") or ""),
        str(slim.get("github_repo") or ""),
        flatten_for_search(slim.get("default_example")),
        flatten_for_search(slim.get("latest_version", {}).get("openapi_schema")),
    ]).lower()

    sdxl_terms = [
        "sdxl",
        "stable-diffusion-xl",
        "stable diffusion xl",
        "stable diffusion-xl",
        "xl-base",
        "xl base",
        "sdxl-base",
        "sdxl refiner",
        "sdxl lora",
    ]
    if "sdxl" in TARGET_MODEL_FAMILIES and any(term in text for term in sdxl_terms):
        return {"task_bucket": "image", "model_family": "sdxl"}

    llm_terms = [
        "llm",
        "text-generation",
        "text generation",
        "chat model",
        "mistral",
        "llama",
        "gemma",
        "qwen",
        "phi-",
        "phi ",
        "tinyllama",
    ]
    small_terms = ["0.5b", "1b", "1.5b", "2b", "3b", "4b", "7b", "tiny", "small", "mini"]
    if "small-llm" in TARGET_MODEL_FAMILIES and any(t in text for t in llm_terms) and any(t in text for t in small_terms):
        return {"task_bucket": "llm", "model_family": "small-llm"}

    return None


def slim_model(m: Dict[str, Any]) -> Dict[str, Any]:
    latest = m.get("latest_version") or {}
    default_example = m.get("default_example") or {}
    owner = m.get("owner")
    name = m.get("name")
    model_id = f"{owner}/{name}" if owner and name else None
    github_url = m.get("github_url")
    github_repo = normalize_github_repo(github_url)

    return {
        "model_id": model_id,
        "owner": owner,
        "name": name,
        "url": m.get("url"),
        "description": m.get("description"),
        "created_at": m.get("created_at"),
        "github_url": github_url,
        "github_repo": github_repo,
        "weights_url": m.get("weights_url"),
        "paper_url": m.get("paper_url"),
        "license_url": m.get("license_url"),
        "run_count": m.get("run_count"),
        "visibility": m.get("visibility"),
        "is_official": m.get("is_official"),
        "latest_version": {
            "id": latest.get("id"),
            "created_at": latest.get("created_at"),
            "cog_version": latest.get("cog_version"),
            "openapi_schema_present": bool(latest.get("openapi_schema")),
            "openapi_schema": latest.get("openapi_schema"),
        },
        "default_example": {
            "id": default_example.get("id"),
            "status": default_example.get("status"),
            "input": default_example.get("input"),
            "output": default_example.get("output"),
            "metrics": default_example.get("metrics"),
            "error": default_example.get("error"),
        } if default_example else None,
    }


def project_route_from_cog_path(cog_path: str) -> str:
    parent = str(Path(cog_path).parent).replace("\\", "/")
    return "" if parent == "." else parent


def path_is_inside_project_route(path: str, route: str) -> bool:
    if route == "":
        return True
    return path.startswith(route.rstrip("/") + "/")


def inspect_github_repo(repo: str) -> Dict[str, Any]:
    repo_data = http_get_json(f"https://api.github.com/repos/{repo}", headers_github())
    branch = repo_data.get("default_branch")
    if not branch:
        return {"ok": False, "repo": repo, "error": "missing_default_branch"}

    tree_url = f"https://api.github.com/repos/{repo}/git/trees/{quote(branch, safe='')}?recursive=1"
    tree_data = http_get_json(tree_url, headers_github())

    tree = tree_data.get("tree") or []
    file_paths = sorted([
        item.get("path")
        for item in tree
        if item.get("type") == "blob" and item.get("path")
    ])

    cog_paths = []
    project_routes = []
    for p in file_paths:
        if p.split("/")[-1].lower() in {"cog.yaml", "cog.yml"}:
            route = project_route_from_cog_path(p)
            project_file_paths = [fp for fp in file_paths if path_is_inside_project_route(fp, route)]
            cog_paths.append({"cog_file_path": p, "project_route": route})
            project_routes.append({
                "cog_file_path": p,
                "project_route": route,
                "project_file_count": len(project_file_paths),
                "project_file_paths": project_file_paths,
            })

    return {
        "ok": True,
        "repo": repo,
        "github_repo_full_name": repo_data.get("full_name"),
        "default_branch": branch,
        "tree_truncated": bool(tree_data.get("truncated")),
        "file_count": len(file_paths),
        "file_paths": file_paths,
        "cog_paths": cog_paths,
        "project_routes": project_routes,
    }


def safe_image_name(model_id: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", model_id.lower()).strip("-")[:120]


def service_name_for(model_id: str) -> str:
    base = safe_image_name(model_id)
    return ("model-" + base)[:60].strip("-")


def image_uri_for(model: Dict[str, Any]) -> str:
    if not GCP_PROJECT_ID:
        raise HTTPException(500, "Missing GCP_PROJECT_ID env var.")
    image = safe_image_name(model["model_id"])
    tag = model.get("latest_version", {}).get("id") or "latest"
    tag = re.sub(r"[^a-zA-Z0-9_.-]+", "-", tag)[:80]
    return f"{GCP_REGION}-docker.pkg.dev/{GCP_PROJECT_ID}/{ARTIFACT_REPOSITORY}/{image}:{tag}"


def build_target_metadata(model: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image_uri": image_uri_for(model),
        "cloud_run_service": service_name_for(model["model_id"]),
        "gpu_type": GPU_TYPE,
        "gpu_count": GPU_COUNT,
        "cpu": GPU_CPU,
        "memory": GPU_MEMORY,
        "min_instances": GPU_MIN_INSTANCES,
        "max_instances": GPU_MAX_INSTANCES,
        "region": GCP_REGION,
        "artifact_repository": ARTIFACT_REPOSITORY,
    }


def aggregate_target_manifest(limit: int = MODEL_LIMIT) -> Dict[str, Any]:
    kept: List[Dict[str, Any]] = []
    inspected_repos: Dict[str, Any] = {}
    skipped = {"no_github": 0, "not_target": 0, "repo_error": 0, "not_single_cog": 0}
    url = "https://api.replicate.com/v1/models"
    page = 0

    while url and len(kept) < limit and page < MAX_REPLICATE_PAGES:
        page += 1
        data = http_get_json(url, headers_replicate())
        for raw in data.get("results") or []:
            if len(kept) >= limit:
                break

            m = slim_model(raw)
            if not m.get("model_id"):
                continue

            family = classify_model(raw, m)
            if not family:
                skipped["not_target"] += 1
                continue

            repo = m.get("github_repo")
            if not repo:
                skipped["no_github"] += 1
                continue

            if repo not in inspected_repos:
                try:
                    inspected_repos[repo] = inspect_github_repo(repo)
                except Exception as e:
                    inspected_repos[repo] = {"ok": False, "repo": repo, "error": str(e)}
                time.sleep(0.15)

            inspection = inspected_repos[repo]
            if not inspection.get("ok"):
                skipped["repo_error"] += 1
                continue

            cog_paths = inspection.get("cog_paths") or []
            if len(cog_paths) != 1:
                skipped["not_single_cog"] += 1
                continue

            enriched = dict(m)
            enriched.update(family)
            enriched["source"] = "replicate"
            enriched["candidate_status"] = "target_single_cog_confident"
            enriched["repo"] = {
                "repo_key": repo,
                "github_repo_full_name": inspection.get("github_repo_full_name"),
                "default_branch": inspection.get("default_branch"),
                "tree_truncated": inspection.get("tree_truncated"),
                "file_count": inspection.get("file_count"),
                "cog_count": len(cog_paths),
                "cog_paths": cog_paths,
                "project_routes": inspection.get("project_routes") or [],
            }
            enriched["build_target"] = build_target_metadata(enriched)
            kept.append(enriched)

        url = data.get("next")
        time.sleep(0.15)

    manifest = {
        "schema_version": "runner-manifest-v1",
        "generated_at": now_iso(),
        "source": "replicate+github",
        "target_model_families": TARGET_MODEL_FAMILIES,
        "max_replicate_pages": MAX_REPLICATE_PAGES,
        "models_count": len(kept),
        "skipped_counts": skipped,
        "models": kept,
    }
    save_json(MANIFEST_PATH, manifest)
    return manifest


def default_manifest() -> Dict[str, Any]:
    return {
        "schema_version": "runner-manifest-v1",
        "generated_at": None,
        "source": "empty-seed",
        "target_model_families": TARGET_MODEL_FAMILIES,
        "models_count": 0,
        "models": [],
    }


def load_manifest() -> Dict[str, Any]:
    return load_json(MANIFEST_PATH, default_manifest())


def get_model(model_id: str) -> Dict[str, Any]:
    manifest = load_manifest()
    for m in manifest.get("models", []):
        if m.get("model_id") == model_id:
            return m
    raise HTTPException(404, f"Model not found: {model_id}")


def authorized_session() -> AuthorizedSession:
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return AuthorizedSession(credentials)


def cloud_build_logs_url(build_id: Optional[str]) -> Optional[str]:
    if not build_id:
        return None
    return f"https://console.cloud.google.com/cloud-build/builds/{build_id};region=global?project={GCP_PROJECT_ID}"


def cloud_build_payload(model: Dict[str, Any], image_uri: str) -> Dict[str, Any]:
    repo = model["github_repo"]
    project_route = model["repo"]["cog_paths"][0].get("project_route") or "."
    registry_host = f"{GCP_REGION}-docker.pkg.dev"

    auth_script = f"""
set -euxo pipefail
export DOCKER_CONFIG=/workspace/.docker
mkdir -p "$DOCKER_CONFIG"
TOKEN="$(gcloud auth print-access-token)"
AUTH="$(printf 'oauth2accesstoken:%s' "$TOKEN" | base64 | tr -d '\n')"
cat > "$DOCKER_CONFIG/config.json" <<EOF
{{"auths":{{"{registry_host}":{{"auth":"$AUTH"}}}}}}
EOF
"""

    cog_script = f"""
set -euxo pipefail
export DOCKER_CONFIG=/workspace/.docker
apt-get update
apt-get install -y --no-install-recommends curl git ca-certificates
curl -L -o /usr/local/bin/cog https://github.com/replicate/cog/releases/latest/download/cog_Linux_x86_64
chmod +x /usr/local/bin/cog
git clone --depth 1 https://github.com/{repo}.git /workspace/modelrepo
cd /workspace/modelrepo/{project_route}
cog --version
cog push {image_uri}
"""

    return {
        "steps": [
            {
                "name": "gcr.io/google.com/cloudsdktool/cloud-sdk:slim",
                "id": "Configure Docker auth for Artifact Registry",
                "entrypoint": "bash",
                "args": ["-lc", auth_script],
            },
            {
                "name": "gcr.io/cloud-builders/docker",
                "id": "Cog push model image",
                "entrypoint": "bash",
                "args": ["-lc", cog_script],
                "env": ["DOCKER_CONFIG=/workspace/.docker"],
            },
        ],
        "timeout": "7200s",
        "options": {
            "logging": "CLOUD_LOGGING_ONLY",
            "diskSizeGb": "200",
        },
        "tags": ["runner-control", "cog-model-build", safe_image_name(model["model_id"])],
    }


def submit_cloud_build(model: Dict[str, Any]) -> Dict[str, Any]:
    if not GCP_PROJECT_ID:
        raise HTTPException(500, "Missing GCP_PROJECT_ID env var.")
    image_uri = image_uri_for(model)
    payload = cloud_build_payload(model, image_uri)

    url = f"https://cloudbuild.googleapis.com/v1/projects/{GCP_PROJECT_ID}/builds"
    session = authorized_session()
    r = session.post(url, json=payload, timeout=60)
    if r.status_code >= 400:
        raise HTTPException(500, {
            "stage": "cloud_build_create",
            "status_code": r.status_code,
            "response": r.text[:4000],
            "payload": payload,
        })

    op = r.json()
    build = (op.get("metadata") or {}).get("build") or {}
    build_id = build.get("id")

    s = state()
    record = s["models"].setdefault(model["model_id"], {})
    record.update({
        "model_id": model["model_id"],
        "task_bucket": model.get("task_bucket"),
        "model_family": model.get("model_family"),
        "build_status": build.get("status") or "QUEUED",
        "build_id": build_id,
        "build_operation": op.get("name"),
        "build_logs_url": cloud_build_logs_url(build_id),
        "image_uri": image_uri,
        "build_submitted_at": now_iso(),
        "github_repo": model.get("github_repo"),
        "cog_path": model["repo"]["cog_paths"][0]["cog_file_path"],
        "project_route": model["repo"]["cog_paths"][0].get("project_route") or ".",
        "cloud_build_payload": payload,
    })
    save_state(s)
    return record


def refresh_build_status(model_id: str) -> Dict[str, Any]:
    s = state()
    rec = s["models"].get(model_id)
    if not rec:
        raise HTTPException(404, f"No state for model: {model_id}")
    build_id = rec.get("build_id")
    if not build_id:
        return rec

    url = f"https://cloudbuild.googleapis.com/v1/projects/{GCP_PROJECT_ID}/builds/{build_id}"
    r = authorized_session().get(url, timeout=60)
    if r.status_code >= 400:
        rec["build_status_error"] = {"status_code": r.status_code, "response": r.text[:2000]}
    else:
        build = r.json()
        rec["build_status"] = build.get("status")
        rec["build_log_url"] = build.get("logUrl") or rec.get("build_logs_url")
        rec["build_finish_time"] = build.get("finishTime")
        rec["build_timing"] = build.get("timing")
    s["models"][model_id] = rec
    save_state(s)
    return rec


def deploy_cloud_run_gpu(model: Dict[str, Any]) -> Dict[str, Any]:
    if not GCP_PROJECT_ID:
        raise HTTPException(500, "Missing GCP_PROJECT_ID env var.")

    s = state()
    rec = s["models"].get(model["model_id"], {})
    if rec.get("build_id"):
        rec = refresh_build_status(model["model_id"])
    if rec.get("build_status") and rec.get("build_status") != "SUCCESS":
        raise HTTPException(409, {
            "stage": "deploy_guard",
            "message": "Build is not marked SUCCESS yet.",
            "build_status": rec.get("build_status"),
            "build_id": rec.get("build_id"),
            "build_logs_url": rec.get("build_logs_url"),
        })

    image_uri = rec.get("image_uri") or image_uri_for(model)
    service_name = service_name_for(model["model_id"])
    parent = f"projects/{GCP_PROJECT_ID}/locations/{GCP_REGION}"
    service_path = f"{parent}/services/{service_name}"

    client = run_v2.ServicesClient()
    container = run_v2.Container(
        image=image_uri,
        ports=[run_v2.ContainerPort(container_port=8080)],
        resources=run_v2.ResourceRequirements(
            limits={
                "cpu": GPU_CPU,
                "memory": GPU_MEMORY,
                "nvidia.com/gpu": str(GPU_COUNT),
            }
        ),
    )

    template = run_v2.RevisionTemplate(
        containers=[container],
        timeout=duration_pb2.Duration(seconds=GPU_TIMEOUT_SECONDS),
        scaling=run_v2.RevisionScaling(
            min_instance_count=GPU_MIN_INSTANCES,
            max_instance_count=GPU_MAX_INSTANCES,
        ),
        annotations={
            "run.googleapis.com/cpu-throttling": "false",
            "run.googleapis.com/gpu-type": GPU_TYPE,
        },
    )

    service = run_v2.Service(name=service_path, template=template)

    try:
        operation = client.create_service(parent=parent, service_id=service_name, service=service)
        action = "create"
    except Exception as create_error:
        try:
            operation = client.update_service(service=service)
            action = "update"
        except Exception as update_error:
            raise HTTPException(500, {
                "stage": "cloud_run_gpu_deploy",
                "create_error": str(create_error),
                "update_error": str(update_error),
            })

    rec.update({
        "deploy_status": "submitted",
        "deploy_action": action,
        "deploy_operation": operation.operation.name,
        "service_name": service_name,
        "service_path": service_path,
        "endpoint_url": f"https://{service_name}-{GCP_PROJECT_ID}.a.run.app",
        "deployed_at": now_iso(),
        "gpu_type": GPU_TYPE,
        "gpu_count": GPU_COUNT,
        "cpu": GPU_CPU,
        "memory": GPU_MEMORY,
    })
    s["models"][model["model_id"]] = rec
    save_state(s)
    return rec


def input_schema_for(model: Dict[str, Any]) -> Dict[str, Any]:
    latest = model.get("latest_version") or {}
    openapi_schema = latest.get("openapi_schema") or {}
    components = openapi_schema.get("components") or {}
    schemas = components.get("schemas") or {}
    return schemas.get("Input") or {}


def default_input_for(model: Dict[str, Any]) -> Dict[str, Any]:
    default_example = model.get("default_example") or {}
    return default_example.get("input") or {}


@app.get("/health")
def health():
    return {"ok": True, "time": now_iso()}


@app.get("/manifest")
def manifest():
    return load_manifest()


@app.post("/admin/refresh")
def refresh_manifest(limit: int = Query(default=MODEL_LIMIT, ge=1, le=250)):
    return aggregate_target_manifest(limit)


@app.get("/models")
def models():
    manifest = load_manifest()
    s = state()
    rows = []
    for m in manifest.get("models", []):
        rec = s["models"].get(m["model_id"], {})
        target = m.get("build_target") or {}
        rows.append({
            "model_id": m["model_id"],
            "task_bucket": m.get("task_bucket"),
            "model_family": m.get("model_family"),
            "description": m.get("description"),
            "github_repo": m.get("github_repo"),
            "run_count": m.get("run_count"),
            "cog_path": m.get("repo", {}).get("cog_paths", [{}])[0].get("cog_file_path"),
            "image_uri": rec.get("image_uri") or target.get("image_uri"),
            "build_status": rec.get("build_status"),
            "build_id": rec.get("build_id"),
            "build_logs_url": rec.get("build_logs_url"),
            "deploy_status": rec.get("deploy_status"),
            "service_name": rec.get("service_name") or target.get("cloud_run_service"),
            "endpoint_url": rec.get("endpoint_url"),
        })
    return {"models": rows, "manifest_generated_at": manifest.get("generated_at"), "models_count": len(rows)}


@app.get("/model")
def model_detail(model_id: str):
    model = get_model(model_id)
    rec = state()["models"].get(model_id, {})
    return {
        "model": model,
        "state": rec,
        "input_schema": input_schema_for(model),
        "default_input": default_input_for(model),
    }


@app.post("/build")
def build_model(model_id: str, req: BuildRequest = BuildRequest()):
    try:
        model = get_model(model_id)
        rec = state()["models"].get(model_id, {})
        if rec.get("image_uri") and rec.get("build_status") in {"QUEUED", "WORKING"} and not req.force:
            return rec
        return submit_cloud_build(model)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, {
            "stage": "build_model",
            "model_id": model_id,
            "error_type": type(e).__name__,
            "error": str(e),
        })


@app.get("/build/status")
def build_status(model_id: str):
    return refresh_build_status(model_id)


@app.post("/deploy")
def deploy_model(model_id: str, req: DeployRequest = DeployRequest()):
    try:
        model = get_model(model_id)
        return deploy_cloud_run_gpu(model)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, {
            "stage": "deploy_model",
            "model_id": model_id,
            "error_type": type(e).__name__,
            "error": str(e),
        })


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
<head>
  <title>Runner Control</title>
  <style>
    body { font-family: system-ui, sans-serif; background:#0b0f17; color:#f2f2f2; margin:24px; }
    .grid { display:grid; grid-template-columns: 420px 1fr; gap:20px; }
    .card { border:1px solid #333; border-radius:14px; padding:14px; margin:10px 0; background:#151922; }
    .pill { display:inline-block; border:1px solid #555; border-radius:999px; padding:2px 8px; margin:2px; font-size:12px; }
    button { padding:10px 14px; border-radius:10px; border:0; cursor:pointer; margin:4px; }
    pre, textarea { background:#101521; color:#dfe7ff; border:1px solid #333; border-radius:10px; padding:12px; width:100%; box-sizing:border-box; }
    textarea { min-height:180px; }
    a { color:#9cc4ff; }
  </style>
</head>
<body>
  <h1>Runner Control</h1>
  <button onclick="refreshManifest()">Aggregate SDXL / small LLM candidates</button>
  <button onclick="loadModels()">Reload manifest</button>
  <div id="summary"></div>
  <div class="grid">
    <div>
      <h2>Models</h2>
      <div id="models"></div>
    </div>
    <div>
      <h2 id="title">Select a model</h2>
      <div id="actions"></div>
      <h3>Default Input JSON</h3>
      <textarea id="inputJson">{}</textarea>
      <h3>Schema / Status</h3>
      <pre id="status">{}</pre>
    </div>
  </div>

<script>
let selected = null;

async function api(path, opts={}) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let payload;
  try { payload = JSON.parse(text); } catch(e) { payload = {raw:text}; }
  if (!res.ok) return {http_status: res.status, error: payload};
  return payload;
}

async function loadModels() {
  const data = await api('/models');
  document.getElementById('summary').innerHTML = `<p>Manifest: ${data.models_count || 0} models. Generated: ${data.manifest_generated_at || 'not yet'}</p>`;
  const el = document.getElementById('models');
  el.innerHTML = '';
  (data.models || []).forEach(m => {
    const div = document.createElement('div');
    div.className = 'card';
    div.innerHTML = `
      <h3>${m.model_id}</h3>
      <span class="pill">${m.task_bucket || ''}</span><span class="pill">${m.model_family || ''}</span>
      <div>${m.github_repo || ''}</div>
      <div>Build: ${m.build_status || 'not built'}</div>
      <div>Deploy: ${m.deploy_status || 'not deployed'}</div>
      ${m.build_logs_url ? `<div><a href="${m.build_logs_url}" target="_blank">Build logs</a></div>` : ''}
      ${m.endpoint_url ? `<div><a href="${m.endpoint_url}" target="_blank">Endpoint</a></div>` : ''}
      <p>${(m.description || '').slice(0,220)}</p>
      <button>Select</button>
    `;
    div.querySelector('button').onclick = () => selectModel(m.model_id);
    el.appendChild(div);
  });
}

async function selectModel(id) {
  selected = id;
  const data = await api('/model?model_id=' + encodeURIComponent(id));
  document.getElementById('title').textContent = id;
  document.getElementById('inputJson').value = JSON.stringify(data.default_input || {}, null, 2);
  document.getElementById('status').textContent = JSON.stringify(data, null, 2);
  document.getElementById('actions').innerHTML = `
    <button onclick="buildSelected()">Build image</button>
    <button onclick="statusSelected()">Check build status</button>
    <button onclick="deploySelected()">Deploy L4 endpoint</button>
  `;
}

async function buildSelected() {
  if (!selected) return;
  const data = await api('/build?model_id=' + encodeURIComponent(selected), {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
  document.getElementById('status').textContent = JSON.stringify(data, null, 2);
  loadModels();
}

async function statusSelected() {
  if (!selected) return;
  const data = await api('/build/status?model_id=' + encodeURIComponent(selected));
  document.getElementById('status').textContent = JSON.stringify(data, null, 2);
  loadModels();
}

async function deploySelected() {
  if (!selected) return;
  const data = await api('/deploy?model_id=' + encodeURIComponent(selected), {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
  document.getElementById('status').textContent = JSON.stringify(data, null, 2);
  loadModels();
}

async function refreshManifest() {
  const data = await api('/admin/refresh', {method:'POST'});
  document.getElementById('status').textContent = JSON.stringify(data, null, 2);
  loadModels();
}

loadModels();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
