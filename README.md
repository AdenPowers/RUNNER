# Runner Control

Cloud Run control-plane app for aggregating targeted Replicate Cog candidates, building model images with Cloud Build, pushing them to Artifact Registry, and deploying Cloud Run GPU endpoints.

## Required Cloud Run env vars

Plain variables:

- `GCP_PROJECT_ID=just-looking-472401`
- `GCP_REGION=us-central1`
- `ARTIFACT_REPOSITORY=model-images`
- `MODEL_LIMIT=25`
- `TARGET_MODEL_FAMILIES=sdxl,small-llm`

Secret references:

- `REPLICATE_API_TOKEN`
- `GITHUB_TOKEN`

## Flow

1. `/admin/refresh` aggregates SDXL candidates from Replicate and GitHub.
2. `/models` reads the saved local manifest.
3. `/build?model_id=...` submits an independent Cloud Build job.
4. Cloud Build installs Cog, clones the model repo, runs `cog push`, and pushes the image to Artifact Registry.
5. `/build/status?model_id=...` reads Cloud Build status.
6. `/deploy?model_id=...` deploys the built image to a Cloud Run L4 GPU service.


This build is narrowed to SDXL image-model self-deployment only. Replicate data is used as metadata/sample input; endpoints are intended to be self-hosted on Cloud Run GPU after image build.
