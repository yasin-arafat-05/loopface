# For every evaluated tag (in results.csv and with a prediction folder):
#   1) false_alarm_types  -> outputs/false_alarm_types.csv   (tp / dup / loc / bg at score 0.3 and 0.5)
#   2) score_separation   -> outputs/score_separation.csv + outputs/score_hist.csv   (AUROC of hit vs false scores)
# Each step is skipped when its result is already in its CSV, so the script can be re-run after an interruption.
#   .\scripts\false_alarm_score.ps1                       all evaluated tags
#   .\scripts\false_alarm_score.ps1 -List                 only show what is left
#   .\scripts\false_alarm_score.ps1 -Tags a_seed0,b_seed0 only these tags
param([switch]$List, [string[]]$Tags)

Set-Location (Split-Path $PSScriptRoot -Parent)     # run from the project root, wherever it is called from

$faThrs  = @('0.3', '0.5')
$results = 'results.csv'
$faCsv   = 'outputs/false_alarm_types.csv'
$ssCsv   = 'outputs/score_separation.csv'
$txtDir  = 'widerface_evaluate/widerface_txt'
$env:PYTHONUNBUFFERED = '1'
New-Item -ItemType Directory -Force 'outputs' | Out-Null

# ---------- which tags ----------
if (-not $Tags) {
    $Tags = @((Import-Csv $results).tag | Sort-Object -Unique)
}
$ready = @()
foreach ($t in $Tags) {
    if (Test-Path (Join-Path $txtDir $t)) { $ready += $t }
    else { Write-Host "skip ${t}: no prediction folder" }
}

# ---------- what is already done ----------
$faDone = @{}
if (Test-Path $faCsv) {
    foreach ($row in (Import-Csv $faCsv)) { $faDone["$($row.tag)|$($row.score_thr)"] = $true }
}
$ssDone = @{}
if (Test-Path $ssCsv) {
    foreach ($row in (Import-Csv $ssCsv)) { $ssDone[$row.tag] = $true }
}

# ---------- plan ----------
$plan = @()
foreach ($t in $ready) {
    $needFa = $false
    foreach ($thr in $faThrs) {
        if (-not $faDone.ContainsKey("$t|$thr")) { $needFa = $true }
    }
    $needSs = -not $ssDone.ContainsKey($t)
    if ($needFa -or $needSs) {
        $plan += [pscustomobject]@{ Tag = $t; Fa = $needFa; Ss = $needSs }
    }
}

Write-Host ""
Write-Host "$($plan.Count) tags with work left:"
foreach ($p in $plan) {
    $steps = @()
    if ($p.Fa) { $steps += 'false_alarm_types' }
    if ($p.Ss) { $steps += 'score_separation' }
    Write-Host "  $($p.Tag)   [$($steps -join ', ')]"
}
if ($List) { return }

# ---------- run ----------
$i = 0
foreach ($p in $plan) {
    $i++
    Write-Host ""
    Write-Host "===== [$i/$($plan.Count)] $($p.Tag) ====="

    if ($p.Fa) {
        python -m analysis.custom.false_alarm_types --tags $p.Tag --score_thrs @faThrs
        if ($LASTEXITCODE -ne 0) { Write-Host "false_alarm_types FAILED for $($p.Tag)" }
    }
    if ($p.Ss) {
        python -m analysis.custom.score_separation --tags $p.Tag
        if ($LASTEXITCODE -ne 0) { Write-Host "score_separation FAILED for $($p.Tag)" }
    }
}
Write-Host ""
Write-Host "done. CSVs: $faCsv, $ssCsv, outputs/score_hist.csv"