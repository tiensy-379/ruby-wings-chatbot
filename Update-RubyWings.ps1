# Update-RubyWings.ps1
# Rebuild index from knowledge.json, then git add/commit/push

# ensure working folder = script folder
Set-Location $PSScriptRoot

Write-Host "RubyWings updater - start"

# check OPENAI_API_KEY in current session
if (-not $env:OPENAI_API_KEY) {
    Write-Host "ERROR: OPENAI_API_KEY is not set in this session."
    Write-Host "Set it with: $env:OPENAI_API_KEY='sk-...'"
    exit 1
}
Write-Host "OPENAI_API_KEY found. Proceeding..."

# run build script
Write-Host "Running: python build_index.py"
python build_index.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: build_index.py failed (exit code $LASTEXITCODE)."
    exit $LASTEXITCODE
}
Write-Host "Build finished."

# check git status
$porcelain = git status --porcelain
if (-not $porcelain) {
    Write-Host "No changes to commit. Exiting."
    exit 0
}

# prepare commit message
param(
    [string]$Message = "Update knowledge + rebuild index"
)
# if user passed args, use them
if ($args.Length -gt 0) {
    $Message = $args -join " "
}

Write-Host "Staging files..."
git add .

Write-Host "Committing: $Message"
git commit -m $Message
if ($LASTEXITCODE -ne 0) {
    Write-Host "git commit returned non-zero ($LASTEXITCODE)."
    exit $LASTEXITCODE
}

Write-Host "Pushing to origin..."
git push
if ($LASTEXITCODE -ne 0) {
    Write-Host "git push failed (exit code $LASTEXITCODE)."
    exit $LASTEXITCODE
}

Write-Host "DONE: updated and pushed. Render will redeploy if auto-deploy is enabled."
