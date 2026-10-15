$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$jobs = @(
  @{ cfg = Join-Path $root 'remote_config.json';  id = 'f93cbf46fc'; file = 'j12_cq_aligned_nois_s0.jsonl'; local = Join-Path $root 'results\j12_cq_aligned_nois_s0.jsonl' },
  @{ cfg = Join-Path $root 'remote_config2.json'; id = '5579534be2'; file = 'j12_cq_aligned_nois_s1.jsonl'; local = Join-Path $root 'results\j12_cq_aligned_nois_s1.jsonl' }
)

function Get-Status($j) {
  $c = Get-Content -Raw $j.cfg | ConvertFrom-Json
  $h = @{ 'X-Token' = $c.token; 'ngrok-skip-browser-warning' = '1' }
  Invoke-RestMethod -Uri ($c.url + '/job/' + $j.id + '?since=0') -Headers $h -Method Get
}

function Download-Result($j) {
  $c = Get-Content -Raw $j.cfg | ConvertFrom-Json
  $h = @{ 'X-Token' = $c.token; 'ngrok-skip-browser-warning' = '1' }
  for ($try=0; $try -lt 20; $try++) {
    try {
      $b = Invoke-WebRequest -Uri ($c.url + '/file?path=' + $j.file) -Headers $h -Method Get
      [IO.File]::WriteAllBytes($j.local, $b.Content)
      return
    } catch { Start-Sleep -Seconds 30 }
  }
  throw "Could not download $($j.file)"
}

while ($true) {
  $states = @($jobs | ForEach-Object { try { Get-Status $_ } catch { [pscustomobject]@{ status = 'unknown' } } })
  if ($states.Count -eq 2 -and ($states | Where-Object { $_ -and $_.status -in @('done','error') }).Count -eq 2) { break }
  Start-Sleep -Seconds 30
}

foreach ($j in $jobs) { Download-Result $j }
$all = @()
foreach ($j in $jobs) {
  if (Test-Path $j.local) {
    $all += @(Get-Content -LiteralPath $j.local | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
  }
}
$valid = @($all | Where-Object { -not $_.error })
$conds = @('base','cq','cq_boost_p','cq_boost_pr')
$lines = [Collections.Generic.List[string]]::new()
$lines.Add('# CQ timestamp-prompt experiment')
$lines.Add('')
$lines.Add("Total records: $($all.Count); valid: $($valid.Count); errors: $($all.Count - $valid.Count)")
$lines.Add('')
$lines.Add('| Task | n | Base | CQ | CQ + prompt boost | CQ + prompt/raw boost |')
$lines.Add('|---|---:|---:|---:|---:|---:|')
foreach ($t in @('TA','TF','CMTB')) {
  $r = @($valid | Where-Object task -eq $t)
  $acc = foreach ($c in $conds) {
    if ($r.Count -eq 0) { 'n/a'; continue }
    $ok = @($r | Where-Object { $arr = $_.$c; $m = [double]::NegativeInfinity; $k = -1; for($i=0; $i -lt $arr.Count; $i++){ if([double]$arr[$i] -gt $m){$m=$arr[$i];$k=$i} }; $k -eq [int]$_.gold })
    [Math]::Round(100 * $ok.Count / $r.Count, 1)
  }
  $lines.Add("| $t | $($r.Count) | $($acc[0]) | $($acc[1]) | $($acc[2]) | $($acc[3]) |")
}
$lines.Add('')
$lines.Add('| Task | Gold | n | Base | CQ | Prompt boost | Prompt/raw boost |')
$lines.Add('|---|---|---:|---:|---:|---:|---:|')
foreach ($t in @('TA','TF','CMTB')) {
  $r0 = @($valid | Where-Object task -eq $t)
  $kmax = if ($t -eq 'TA') { 2 } else { 3 }
  for ($g=0; $g -lt $kmax; $g++) {
    $r = @($r0 | Where-Object { [int]$_.gold -eq $g })
    $acc = foreach ($c in $conds) {
      if ($r.Count -eq 0) { 'n/a'; continue }
      $ok = @($r | Where-Object { $arr = $_.$c; $m=[double]::NegativeInfinity; $k=-1; for($i=0; $i -lt $arr.Count; $i++){if([double]$arr[$i] -gt $m){$m=$arr[$i];$k=$i}}; $k -eq $g })
      [Math]::Round(100 * $ok.Count / $r.Count, 1)
    }
    $label = [char](65 + $g)
    $lines.Add("| $t | $label | $($r.Count) | $($acc[0]) | $($acc[1]) | $($acc[2]) | $($acc[3]) |")
  }
}
$out = Join-Path $root 'results\j12_cq_aligned_nois_final_report.md'
$lines -join "`r`n" | Set-Content -LiteralPath $out -Encoding UTF8
Set-Content -LiteralPath (Join-Path $root 'results\j12_cq_aligned_nois_monitor_done.txt') -Value (Get-Date -Format o) -Encoding ASCII
