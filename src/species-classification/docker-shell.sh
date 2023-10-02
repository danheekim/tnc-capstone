#!/bin/bash

set -e

export GCP_PROJECT="tnc-cameratraps"
export GCP_ZONE="us-central1-a"
export GOOGLE_APPLICATION_CREDENTIALS="../secrets/capstone_service_account.json"
export SECRETS_DIR=$(pwd)/../secrets/

# Build the image based on the Dockerfile
docker build -t species-classification -f Dockerfile .

# Run Container
docker run --rm -ti \
  -e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
  -e GCP_PROJECT=$GCP_PROJECT \
  -e GCP_ZONE=$GCP_ZONE \
  --mount type=bind,source="$(pwd)",target=/app \
  --mount type=bind,source="$SECRETS_DIR",target=/secrets \
  species-classification