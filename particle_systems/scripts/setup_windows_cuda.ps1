$ErrorActionPreference = "Stop"

$RepositoryRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepositoryRoot

uv sync --project particle_systems --no-group cpu --group cuda
uv run --project particle_systems --no-sync python -m particle_systems.verify_runtime --device cuda
