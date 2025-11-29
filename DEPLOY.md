# SPECTRA Deployment Guide

## Pre-flight Checklist

### 1. Local Container Test
```bash
# Build image
docker build -t spectra-core:local .

# Run locally
docker run --rm -p 8501:8501 spectra-core:local

# Test at http://localhost:8501
```

### 2. AWS Configuration
```bash
# Configure AWS CLI
aws configure
# Enter: Access Key ID, Secret Access Key, Default region (ap-southeast-1), Output format (json)

# Optional: Set environment variables
export AWS_REGION=ap-southeast-1
export ECR_REPO=spectra-core
export IMAGE_TAG=latest
```

### 3. Push to ECR
```bash
# Push image to ECR
bash scripts/ecr_push.sh

# Copy the printed IMAGE_URI for App Runner
```

## App Runner Setup

### 1. Create Service
1. Go to AWS App Runner Console
2. Create service → Container image → Amazon ECR
3. Select the pushed image (use IMAGE_URI from push script)
4. Configure service:
   - **Port**: 8501
   - **CPU**: 1 vCPU
   - **Memory**: 2 GB
   - **Environment variables**: None required

### 2. Monitor Deployment
- Check service logs for startup issues
- Wait for service to reach "Running" status
- Test the public URL

## Cost Control

### Estimate Costs
```bash
# Light usage (2h active, 22h paused daily)
python scripts/cost_estimator.py --vcpu 1 --memgb 2 --active-h-per-day 2 --paused-h-per-day 22 --credit 100

# Heavy usage (8h active, 16h paused daily)  
python scripts/cost_estimator.py --vcpu 1 --memgb 2 --active-h-per-day 8 --paused-h-per-day 16 --credit 100
```

### Pause/Resume Service
```bash
# Set service ARN (copy from App Runner console)
export SERVICE_ARN=arn:aws:apprunner:region:account:service/service-name/service-id

# Pause when not in use
bash scripts/apprunner_pause.sh

# Resume when needed
bash scripts/apprunner_resume.sh
```

## Troubleshooting

### Common Issues
- **Build fails**: Check Docker Desktop is running with WSL integration
- **Push fails**: Verify AWS credentials and region
- **App won't start**: Check service logs in App Runner console
- **Upload issues**: Verify 2GB limits in Streamlit config

### Logs
- App Runner service logs available in AWS console
- Local container logs: `docker logs <container_id>`