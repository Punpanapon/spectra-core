#!/usr/bin/env bash
set -euo pipefail

# Load SERVICE_ARN from .env if it exists
if [ -f .env ]; then
    source .env
fi

if [ -z "${SERVICE_ARN:-}" ]; then
    echo "Error: SERVICE_ARN not set. Export it or add to .env file."
    echo "Example: export SERVICE_ARN=arn:aws:apprunner:region:account:service/service-name/service-id"
    exit 1
fi

echo "Pausing App Runner service: $SERVICE_ARN"
aws apprunner pause-service --service-arn "$SERVICE_ARN"
echo "Service pause initiated. Check AWS console for status."