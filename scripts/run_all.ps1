# Evaluate every trained model in runs/ (except single_batch and ResNet-50) on WIDER FACE val.
# Loop models (loop or pre_loop in the config) are run for every L in $loops; others once (tag L0).
# Each run appends one row to results.csv; tags already there are skipped,
# so the script can be re-run after an interruption.
#   .\run_all.ps1           run what is left
#   .\run_all.ps1 -List     only show what is left
param([switch]$List)

$loops   = @(1, 2, 4, 6, 8, 12, 16)
$results = 'results.csv'

# ---------- tags already in results.csv ----------
$done = @{}
if (Test-Path $results) {
    foreach ($line in (Get-Content $results)) {
        foreach ($f in $line.Split(',')) { $done[$f.Trim('"', ' ')] = $true }
    }
}

# ---------- every (model, L) that is not evaluated yet ----------
$runs = Get-ChildItem runs -Directory
$runs = $runs | Where-Object { $_.Name -notlike 'single_batch*' -and $_.Name -notlike 'cfg_re50*' }

$todo = @()
foreach ($r in $runs) {
    $ckpt = Join-Path $r.FullName 'final.pth'
    if (-not (Test-Path $ckpt)) { Write-Host "skip $($r.Name): no final.pth"; continue }
    $cfg  = (Get-Content (Join-Path $r.FullName 'run.json') -Raw | ConvertFrom-Json).args.config
    $seed = $r.Name -replace '.*_s(\d+)$', '$1'
    $py   = "from data import config as c; k = getattr(c, '$cfg'); print(int(bool(k.get('loop') or k.get('pre_loop'))))"
    $hasLoop = python -c $py
    $Ls = @(0)
    if ($hasLoop -eq '1') { $Ls = $loops }
    foreach ($L in $Ls) {
        $tag = "${cfg}_L${L}_seed${seed}"
        if ($done.ContainsKey($tag)) { continue }
        $evalArgs = @('-m', $ckpt, '--config', $cfg)
        if ($L -gt 0) { $evalArgs += @('--loop', "$L") }
        $todo += [pscustomobject]@{ Tag = $tag; EvalArgs = $evalArgs }
    }
}
# released RetinaFace weights, for reference
if (-not $done.ContainsKey('released_mnet_L0')) {
    $todo += [pscustomobject]@{
        Tag      = 'released_mnet_L0'
        EvalArgs = @('-m', 'weights/mobilenet0.25_Final.pth', '--network', 'mobile0.25')
    }
}

Write-Host ""
Write-Host "$($todo.Count) evaluations left:"
foreach ($t in $todo) { Write-Host "  $($t.Tag)" }
if ($List) { return }

# ---------- run ----------
$i = 0
foreach ($t in $todo) {
    $i++
    Write-Host ""
    Write-Host "===== [$i/$($todo.Count)] $($t.Tag) ====="
    python test_widerface.py @($t.EvalArgs) --tag $t.Tag --results $results
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $($t.Tag)" }
}
Write-Host ""
Write-Host "done. results in $results"