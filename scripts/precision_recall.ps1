# For every trained model (except single_batch and ResNet-50), and every loop count L:
#   1) test_widerface.py    -> results.csv                      (AP)
#   2) precision_by_score   -> outputs/precision_by_score.csv   (false alarms)
#   3) recall_by_size       -> outputs/recall_by_size.csv       (recall per face size, 3 score thresholds)
# Each step is skipped when its result is already in its CSV, so the script can be re-run after an interruption.
#   .\eval_all.ps1          run what is left
#   .\eval_all.ps1 -List    only show what is left
param([switch]$List)

$loops     = @(1, 2, 4, 6, 8, 12, 16)
$scoreThrs = @('0.02', '0.3', '0.5')
$results   = 'results.csv'
$pbsCsv    = 'outputs/precision_by_score.csv'
$rbsCsv    = 'outputs/recall_by_size.csv'
$txtDir    = 'widerface_evaluate/widerface_txt'
$env:PYTHONUNBUFFERED = '1'
New-Item -ItemType Directory -Force 'outputs' | Out-Null

# ---------- what is already done ----------
# results.csv: every field of every row counts as a finished tag
$evalDone = @{}
if (Test-Path $results) {
    foreach ($line in (Get-Content $results)) {
        foreach ($f in $line.Split(',')) { $evalDone[$f.Trim('"', ' ')] = $true }
    }
}
# precision_by_score.csv: finished when the tag has any row
$pbsDone = @{}
if (Test-Path $pbsCsv) {
    foreach ($row in (Import-Csv $pbsCsv)) { $pbsDone[$row.tag] = $true }
}
# recall_by_size.csv: finished per (tag, score_thr)
$rbsDone = @{}
if (Test-Path $rbsCsv) {
    foreach ($row in (Import-Csv $rbsCsv)) { $rbsDone["$($row.tag)|$($row.score_thr)"] = $true }
}

# ---------- every (model, L) to evaluate ----------
$runs = Get-ChildItem runs -Directory
$runs = $runs | Where-Object { $_.Name -notlike 'single_batch*' -and $_.Name -notlike 'cfg_re50*' }

$items = @()
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
        $evalArgs = @('-m', $ckpt, '--config', $cfg)
        if ($L -gt 0) { $evalArgs += @('--loop', "$L") }
        $items += [pscustomobject]@{ Tag = "${cfg}_L${L}_seed${seed}"; EvalArgs = $evalArgs }
    }
}
# released RetinaFace weights, for reference
$items += [pscustomobject]@{
    Tag      = 'released_mnet_L0'
    EvalArgs = @('-m', 'weights/mobilenet0.25_Final.pth', '--network', 'mobile0.25')
}

# ---------- plan: which steps are left for each tag ----------
$plan = @()
foreach ($it in $items) {
    $needEval = -not $evalDone.ContainsKey($it.Tag)
    $needPbs  = -not $pbsDone.ContainsKey($it.Tag)
    $needRbs  = @()
    foreach ($thr in $scoreThrs) {
        if (-not $rbsDone.ContainsKey("$($it.Tag)|$thr")) { $needRbs += $thr }
    }
    if ($needEval -or $needPbs -or $needRbs.Count -gt 0) {
        $plan += [pscustomobject]@{
            Tag = $it.Tag; EvalArgs = $it.EvalArgs
            Eval = $needEval; Pbs = $needPbs; Rbs = $needRbs
        }
    }
}

Write-Host ""
Write-Host "$($plan.Count) tags with work left:"
foreach ($p in $plan) {
    $steps = @()
    if ($p.Eval) { $steps += 'eval' }
    if ($p.Pbs)  { $steps += 'precision' }
    if ($p.Rbs.Count -gt 0) { $steps += "recall@$($p.Rbs -join '/')" }
    Write-Host "  $($p.Tag)   [$($steps -join ', ')]"
}
if ($List) { return }

# ---------- run ----------
$i = 0
foreach ($p in $plan) {
    $i++
    Write-Host ""
    Write-Host "===== [$i/$($plan.Count)] $($p.Tag) ====="

    if ($p.Eval) {
        python test_widerface.py @($p.EvalArgs) --tag $p.Tag --results $results
        if ($LASTEXITCODE -ne 0) { Write-Host "eval FAILED, skipping the analyses of $($p.Tag)"; continue }
    }

    if (-not (Test-Path (Join-Path $txtDir $p.Tag))) {
        Write-Host "no prediction folder for $($p.Tag), skipping the analyses"
        continue
    }

    if ($p.Pbs) {
        python -m analysis.custom.precision_by_score --tags $p.Tag
        if ($LASTEXITCODE -ne 0) { Write-Host "precision_by_score FAILED for $($p.Tag)" }
    }

    foreach ($thr in $p.Rbs) {
        python -m analysis.custom.recall_by_size --tags $p.Tag --score_thr $thr
        if ($LASTEXITCODE -ne 0) { Write-Host "recall_by_size FAILED for $($p.Tag) at $thr" }
    }
}
Write-Host ""
Write-Host "done. CSVs: $results, $pbsCsv, $rbsCsv"