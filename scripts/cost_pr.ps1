# For every model in runs/ (except single_batch and ResNet-50):
#   1) benchmark  -> outputs/benchmark.csv   (params, GMACs, CPU ms; one row per config and L)
#   2) pr_curve   -> outputs/pr_curve.csv    (precision-recall curve; one block per evaluated tag)
# Steps already in their CSV are skipped, so the script can be re-run after an interruption.
#   .\scripts\cost_pr.ps1           run what is left
#   .\scripts\cost_pr.ps1 -List     only show what is left
param([switch]$List)

Set-Location (Split-Path $PSScriptRoot -Parent)     # run from the project root

$loops   = @(1, 2, 4, 8, 16)
$results = 'results.csv'
$bmCsv   = 'outputs/benchmark.csv'
$prCsv   = 'outputs/pr_curve.csv'
$txtDir  = 'widerface_evaluate/widerface_txt'
$env:PYTHONUNBUFFERED = '1'
New-Item -ItemType Directory -Force 'outputs' | Out-Null

# ---------- configs in runs/ ----------
$runs = Get-ChildItem runs -Directory
$runs = $runs | Where-Object { $_.Name -notlike 'single_batch*' -and $_.Name -notlike 'cfg_re50*' }
$configs = @()
foreach ($r in $runs) {
    $j = Join-Path $r.FullName 'run.json'
    if (-not (Test-Path $j)) { continue }
    $cfg = (Get-Content $j -Raw | ConvertFrom-Json).args.config
    if ($configs -notcontains $cfg) { $configs += $cfg }
}

# ---------- what is already done ----------
$bmDone = @{}
if (Test-Path $bmCsv) {
    foreach ($row in (Import-Csv $bmCsv)) {
        if ($row.device -eq 'cpu') { $bmDone["$($row.config)|$($row.L)"] = $true }
    }
}
$prDone = @{}
if (Test-Path $prCsv) {
    foreach ($row in (Import-Csv $prCsv)) { $prDone[$row.tag] = $true }
}

# ---------- plan: benchmark (per config) ----------
$bmTodo = @()
foreach ($cfg in $configs) {
    $py = "from data import config as c; k = getattr(c, '$cfg'); print(int(bool(k.get('loop') or k.get('pre_loop'))))"
    $hasLoop = python -c $py
    $Ls = @(0)
    if ($hasLoop -eq '1') { $Ls = $loops }
    $missing = $false
    foreach ($L in $Ls) {
        if (-not $bmDone.ContainsKey("$cfg|$L")) { $missing = $true }
    }
    if ($missing) { $bmTodo += $cfg }
}

# ---------- plan: pr_curve (per evaluated tag) ----------
$prTodo = @()
if (Test-Path $results) {
    $tags = @((Import-Csv $results).tag | Sort-Object -Unique)
    foreach ($t in $tags) {
        if ($prDone.ContainsKey($t)) { continue }
        if (Test-Path (Join-Path $txtDir $t)) { $prTodo += $t }
        else { Write-Host "skip ${t}: no prediction folder" }
    }
}

Write-Host ""
Write-Host "benchmark: $($bmTodo.Count) configs left"
foreach ($c in $bmTodo) { Write-Host "  $c" }
Write-Host "pr_curve: $($prTodo.Count) tags left"
foreach ($t in $prTodo) { Write-Host "  $t" }
if ($List) { return }

# ---------- run ----------
$i = 0
foreach ($c in $bmTodo) {
    $i++
    Write-Host ""
    Write-Host "===== benchmark [$i/$($bmTodo.Count)] $c ====="
    python -m analysis.custom.benchmark --configs $c --loops @loops --device cpu --threads 1
    if ($LASTEXITCODE -ne 0) { Write-Host "benchmark FAILED for $c" }
}
$i = 0
foreach ($t in $prTodo) {
    $i++
    Write-Host ""
    Write-Host "===== pr_curve [$i/$($prTodo.Count)] $t ====="
    python -m analysis.custom.pr_curve --tags $t
    if ($LASTEXITCODE -ne 0) { Write-Host "pr_curve FAILED for $t" }
}
Write-Host ""
Write-Host "done. CSVs: $bmCsv, $prCsv"