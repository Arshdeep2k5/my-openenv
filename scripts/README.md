# Validation Scripts

Pre-submission validation scripts for OpenEnv environments.

## Quick Start

### Linux / macOS (Bash)

```bash
# Make executable
chmod +x scripts/validate-submission.sh

# Run with your HF Space URL
./scripts/validate-submission.sh https://your-space.hf.space

# Or specify a repo directory
./scripts/validate-submission.sh https://your-space.hf.space ./path/to/repo
```

### Windows (PowerShell)

```powershell
# Allow script execution (run once)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run with your HF Space URL
.\scripts\validate-submission.ps1 -PingUrl https://your-space.hf.space

# Or specify a repo directory
.\scripts\validate-submission.ps1 -PingUrl https://your-space.hf.space -RepoDir ./path/to/repo
```

## What Gets Validated

The validation scripts check three critical aspects:

1. **HF Space Connectivity** — Verifies your Space is live and responds to `/reset` endpoint
2. **Docker Build** — Ensures your Dockerfile builds successfully (600s timeout)
3. **openenv validate** — Confirms environment structure is correct for deployment

## Requirements

- **Docker**: [Get Docker](https://docs.docker.com/get-docker/)
- **openenv-core**: `pip install openenv-core`
- **curl**: Usually pre-installed on modern systems

## Usage Examples

### Validate current directory (recommended)
```bash
./scripts/validate-submission.sh https://my-team.hf.space
```

### Validate from anywhere, specify repo path
```bash
./scripts/validate-submission.sh https://my-team.hf.space ./my-env-repo
```

### Run from remote (Linux/macOS only)
```bash
curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh | bash -s -- https://my-team.hf.space
```

## Exit Codes

- **0**: All checks passed, submission ready
- **1**: One or more checks failed (see output for details)

## Troubleshooting

### "HF Space not reachable"
- Verify your Space URL is correct
- Check that your Space is running (not paused)
- Verify network connectivity

### "Docker build failed"
- Check Docker is installed and running
- Review Dockerfile for syntax errors
- Ensure all dependencies are pinned with exact versions

### "openenv validate failed"
- Verify `openenv.yaml` exists in repo root
- Check `server/app.py` has proper `main()` function and `if __name__ == '__main__'` guard
- Ensure all required files are present

## Notes

- Docker builds have a 600s timeout to prevent hanging
- HF Space connectivity timeout is 30s
- Consider running this script before final submission to catch issues early
