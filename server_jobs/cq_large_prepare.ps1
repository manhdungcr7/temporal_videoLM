param([ValidateSet('Select','Split')][string]$Action='Select')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$out=Join-Path $root 'results/cq_large_20261014'
New-Item -ItemType Directory -Path $out -Force|Out-Null
if($Action -eq 'Select'){
  $old=@()
  foreach($kind in 'diagnostic','control'){
    foreach($s in 1,2){$old+=@((Get-Content -Raw (Join-Path $root "results/cq_${kind}_manifest_s${s}.json")|ConvertFrom-Json).selected)}
  }
  $exclusions=@{idx=@($old|ForEach-Object{$_.idx});video_paths=@($old|ForEach-Object{$_.video_path.Replace('\','/')}|Sort-Object -Unique)}
  $json=$exclusions|ConvertTo-Json -Depth 6
  [IO.File]::WriteAllText((Join-Path $out 'exclusions.json'),$json,[Text.UTF8Encoding]::new($false))
  $jobs=foreach($s in 1,2){
    & (Join-Path $PSScriptRoot 'cq_lab.ps1') -Action Upload -Server $s -File cq_large_exclusions.json -Dest (Join-Path $out 'exclusions.json')|Out-Host
    $job=(& (Join-Path $PSScriptRoot 'cq_lab.ps1') -Action Submit -Server $s -File (Join-Path $PSScriptRoot 'cq_large_select.py') -Name "cq_large_select_s${s}"|Out-String|ConvertFrom-Json)
    @{server=$s;job_id=$job.job_id}
  }
  [IO.File]::WriteAllText((Join-Path $out 'selection_jobs.json'),(ConvertTo-Json -InputObject @($jobs) -Depth 6),[Text.UTF8Encoding]::new($false))
  $jobs|ConvertTo-Json
}else{
  $all=@()
  foreach($s in 1,2){
    $file="cq_large_20261014_selection_s${s}.json"
    & (Join-Path $PSScriptRoot 'cq_lab.ps1') -Action Download -Server $s -File $file -Dest (Join-Path $out "selection_s${s}.json")|Out-Host
    $selection=Get-Content -Raw (Join-Path $out "selection_s${s}.json")|ConvertFrom-Json
    if($selection.status -ne 'done'){throw 'Selection is not complete'}
    $all+=@($selection.selected)
  }
  if($all.Count -ne 400 -or @($all|Group-Object idx|Where-Object Count -ne 1).Count){throw 'Expected 400 distinct sample IDs'}
  $excluded=(Get-Content -Raw (Join-Path $out 'exclusions.json')|ConvertFrom-Json).video_paths
  foreach($x in $all){if($x.duration -gt 50 -or $excluded -contains $x.video_path){throw 'Duration/exclusion check failed'}}
  $groups=@($all|Group-Object task,gold|Sort-Object Name)
  if($groups.Count -ne 8 -or @($groups|Where-Object Count -ne 50).Count){throw 'Expected eight cells of fifty'}
  $bins=@{1=[Collections.Generic.List[object]]::new();2=[Collections.Generic.List[object]]::new()}
  # Twenty-five from every cell go to each server; interleave all cells.
  for($k=0;$k -lt 25;$k++){
    foreach($g in $groups){$bins[1].Add($g.Group[2*$k]);$bins[2].Add($g.Group[2*$k+1])}
  }
  foreach($s in 1,2){
    $manifest=@{run_id='cq_large_20261014';server=$s;seed=2026101401;max_duration=50;per_cell=25;
      method='episode_constraints';source_sha256='0426cbccdbe41862408464e8214bbd6be4a855561b35004f75ed22ee6dec18e5';
      samples_sha256=$selection.samples_sha256;selected=@($bins[$s]);excluded_video_paths=$excluded;
      selection='50 per cell, duration-filtered, prior 16 development videos excluded; 25 per cell per server'}
    $dest=Join-Path $out "manifest_s${s}.json"
    [IO.File]::WriteAllText($dest,($manifest|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
    & (Join-Path $PSScriptRoot 'cq_lab.ps1') -Action Upload -Server $s -File "cq_large_manifest_s${s}.json" -Dest $dest|Out-Host
    Write-Output "Server $s : $($bins[$s].Count) samples, 25 per cell"
  }
}
