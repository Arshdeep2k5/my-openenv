#Requires -Version 5.1

<#
.SYNOPSIS
    OpenEnv Submission Validator
    
.DESCRIPTION
    Checks that your HF Space is live, Docker image builds, and openenv validate passes.
    
.PARAMETER PingUrl
    Your HuggingFace Space URL (e.g. https://your-space.hf.space)
    
.PARAMETER RepoDir
    Path to your repo (default: current directory)
    
.EXAMPLE
    .\validate-submission.ps1 -PingUrl https://my-team.hf.space
    
.EXAMPLE
    .\validate-submission.ps1 -PingUrl https://my-team.hf.space -RepoDir ./my-repo

.NOTES
    Prerequisites:
    - Docker:       https://docs.docker.com/get-docker/
    - openenv-core: pip install openenv-core
    - curl (usually pre-installed on Windows 10+)
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$PingUrl,

    [Parameter(Mandatory = $false)]
    [string]$RepoDir = "."
)

$ErrorActionPreference = "Stop"

# Configuration
$DOCKER_BUILD_TIMEOUT = 600
$TOTAL_CHECKS = 3
$PASSED_CHECKS = 0

# Colors (simplified for PowerShell)
function Write-Header {
    param([string]$Text)
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan -NoNewline
    Write-Host ""
    Write-Host "========================================`n" -ForegroundColor Cyan
}

function Write-Pass {
    param([string]$Text)
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] " -NoNewline
    Write-Host "PASSED" -ForegroundColor Green -NoNewline
    Write-Host " -- $Text"
    $script:PASSED_CHECKS++
}

function Write-Fail {
    param([string]$Text)
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] " -NoNewline
    Write-Host "FAILED" -ForegroundColor Red -NoNewline
    Write-Host " -- $Text"
}

function Write-Hint {
    param([string]$Text)
    Write-Host "  Hint: " -ForegroundColor Yellow -NoNewline
    Write-Host "$Text"
}

function Stop-At {
    param([string]$Step)
    Write-Host "`n" -NoNewline
    Write-Host "Validation stopped at $Step. " -ForegroundColor Red -BackgroundColor Black -NoNewline
    Write-Host "Fix the above before continuing.`n" -ForegroundColor Red -BackgroundColor Black
    exit 1
}

# Resolve repo directory
try {
    $RepoPath = (Resolve-Path $RepoDir -ErrorAction Stop).Path
}
catch {
    Write-Fail "Directory '$RepoDir' not found"
    exit 1
}

$PingUrl = $PingUrl.TrimEnd('/')

# Header
Write-Header "OpenEnv Submission Validator"
Write-Host "Repo:     $RepoPath"
Write-Host "Ping URL: $PingUrl"

# Step 1: Ping HF Space
Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] " -NoNewline
Write-Host "Step 1/3: Pinging HF Space" -ForegroundColor Cyan -NoNewline
Write-Host " ($PingUrl/reset) ..."

try {
    $body = @{} | ConvertTo-Json
    $response = Invoke-WebRequest `
        -Uri "$PingUrl/reset" `
        -Method Post `
        -ContentType "application/json" `
        -Body $body `
        -TimeoutSec 30 `
        -ErrorAction Stop

    if ($response.StatusCode -eq 200) {
        Write-Pass "HF Space is live and responds to /reset"
    }
    else {
        Write-Fail "HF Space /reset returned HTTP $($response.StatusCode) (expected 200)"
        Write-Hint "Make sure your Space is running and the URL is correct."
        Write-Hint "Try opening $PingUrl in your browser first."
        Stop-At "Step 1"
    }
}
catch {
    Write-Fail "HF Space not reachable (connection failed or timed out)"
    Write-Hint "Check your network connection and that the Space is running."
    Write-Hint "Try: Invoke-WebRequest -Uri '$PingUrl/reset' -Method Post"
    Stop-At "Step 1"
}

# Step 2: Docker build
Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] " -NoNewline
Write-Host "Step 2/3: Running docker build" -ForegroundColor Cyan -NoNewline
Write-Host " ..."

# Check if docker is available
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Fail "docker command not found"
    Write-Hint "Install Docker: https://docs.docker.com/get-docker/"
    Stop-At "Step 2"
}

# Find Dockerfile
$DockerContext = $null
if (Test-Path "$RepoPath/Dockerfile") {
    $DockerContext = $RepoPath
}
elseif (Test-Path "$RepoPath/server/Dockerfile") {
    $DockerContext = "$RepoPath/server"
}
else {
    Write-Fail "No Dockerfile found in repo root or server/ directory"
    Stop-At "Step 2"
}

Write-Host "  Found Dockerfile in $DockerContext"

# Run docker build with timeout
try {
    $buildJob = Start-Job -ScriptBlock {
        param($Context)
        & docker build $Context 2>&1
    } -ArgumentList $DockerContext

    $finished = $buildJob | Wait-Job -Timeout $DOCKER_BUILD_TIMEOUT
    
    if ($null -eq $finished) {
        # Timeout occurred
        Stop-Job -Job $buildJob -Force | Out-Null
        Write-Fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
        Stop-At "Step 2"
    }

    $buildOutput = Receive-Job -Job $buildJob
    $buildExitCode = $buildJob.State
    Remove-Job -Job $buildJob -Force

    if ($buildJob.State -eq "Completed") {
        Write-Pass "Docker build succeeded"
    }
    else {
        Write-Fail "Docker build failed"
        if ($buildOutput) {
            $buildOutput[-20..-1] | ForEach-Object { Write-Host $_ }
        }
        Stop-At "Step 2"
    }
}
catch {
    Write-Fail "Docker build error: $_"
    Stop-At "Step 2"
}

# Step 3: openenv validate
Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] " -NoNewline
Write-Host "Step 3/3: Running openenv validate" -ForegroundColor Cyan -NoNewline
Write-Host " ..."

if (-not (Get-Command openenv -ErrorAction SilentlyContinue)) {
    Write-Fail "openenv command not found"
    Write-Hint "Install it: pip install openenv-core"
    Stop-At "Step 3"
}

try {
    Push-Location $RepoPath
    $validateOutput = & openenv validate 2>&1
    $validateExitCode = $LASTEXITCODE
    Pop-Location

    if ($validateExitCode -eq 0) {
        Write-Pass "openenv validate passed"
        if ($validateOutput) {
            Write-Host "  $validateOutput"
        }
    }
    else {
        Write-Fail "openenv validate failed"
        if ($validateOutput) {
            $validateOutput | ForEach-Object { Write-Host $_ }
        }
        Stop-At "Step 3"
    }
}
catch {
    Write-Fail "openenv validate error: $_"
    Stop-At "Step 3"
}

# Summary
Write-Host "`n"
Write-Header "All 3/3 checks passed! Your submission is ready to submit."
exit 0
