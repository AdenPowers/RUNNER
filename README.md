# Runner Control

Clean Cloud Run control-plane app for testing Replicate Cog models on GCP.

This repository is intentionally flat:

```text
Dockerfile
cloudbuild.yaml
requirements.txt
app/
  __init__.py
  main.py
```

No buildpacks. No nested source directory. No generated inline trigger YAML. Cloud Build builds this Dockerfile, pushes the image to Artifact Registry, then deploys the image to Cloud Run.

## Cloud Run service

Default service name in `cloudbuild.yaml`:

```text
runner-control
```

Default project and region:

```text
just-looking-472401
us-central1
```

The deploy step sets these plain environment variables:

```text
GCP_PROJECT_ID=just-looking-472401
GCP_REGION=us-central1
ARTIFACT_REPOSITORY=model-images
MODEL_LIMIT=10
```

Add these as Secret Manager-backed environment variables on the Cloud Run service after the first successful deploy:

```text
REPLICATE_API_TOKEN
GITHUB_TOKEN
```

## Required Artifact Registry repositories

For the control app image, this build uses the existing Cloud Run source repository:

```text
cloud-run-source-deploy
```

For model images built by the app, create this Docker repository once if it does not already exist:

```bash
gcloud artifacts repositories create model-images \
  --project=just-looking-472401 \
  --repository-format=docker \
  --location=us-central1 \
  --description="Cog model images"
```

## First test

After deployment, open:

```text
/health
```

Expected response:

```json
{"ok":true}
```
