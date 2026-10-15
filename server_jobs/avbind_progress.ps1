param([switch]$Download,[string]$Registry='results/avbind_active_jobs.json',[string]$NameLike='*')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$registryData=Get-Content -Raw (Join-Path $root $Registry)|ConvertFrom-Json
$all=@()
foreach($job in $registryData.jobs){
  if($job.name -notlike $NameLike){continue}
  $cfgName=if($job.server -eq 1){'remote_config.json'}else{'remote_config2.json'}
  $cfg=Get-Content -Raw (Join-Path $root $cfgName)|ConvertFrom-Json
  $headers=@{'X-Token'=$cfg.token;'ngrok-skip-browser-warning'='1'}
  try {
    $state=Invoke-RestMethod -Uri ($cfg.url+'/job/'+$job.id) -Headers $headers -TimeoutSec 15
    [pscustomobject]@{Server=$job.server;Experiment=$job.name;Status=$state.status;Seconds=$state.seconds}|Format-Table -AutoSize
    if($state.error){Write-Output $state.error}
    $state.out -split '\n'|Where-Object{$_ -match 'AVBIND_PILOT|AVBIND_DAILY|AVBIND_CMTB|REVERSE_QUERY|BIND_TRAIN|BIND_STRUCTURAL|BIND_REAL_PILOT|ADDRESS_CACHE|ADDRESS_PRETRAIN'}|Select-Object -Last 2|Write-Output
  } catch {Write-Output "Server $($job.server) unavailable for $($job.name)"}
  if($Download -and $job.file){
    try {
      $response=Invoke-WebRequest -UseBasicParsing -Uri ($cfg.url+'/file?path='+[uri]::EscapeDataString($job.file)) -Headers $headers -TimeoutSec 30
      $target=Join-Path $root ('results/'+$job.name+'_s'+$job.server+'.jsonl')
      [IO.File]::WriteAllBytes($target,$response.RawContentStream.ToArray())
    } catch {Write-Output "Not yet available: $($job.file)"}
  }
  $local=Join-Path $root ('results/'+$job.name+'_s'+$job.server+'.jsonl')
  if($job.kind -in @('training','address_training','feature_extraction','alignment_training','loader')){continue}
  if($job.file -and (Test-Path -LiteralPath $local)){
    $records=@(Get-Content $local|Where-Object{$_.Trim()}|ForEach-Object{try{$_|ConvertFrom-Json}catch{}})
    $byIndex=@{};$withoutIndex=@()
    foreach($record in $records){
      if($null -eq $record.idx){$withoutIndex+=$record;continue}
      $key=[string]$record.idx
      if(-not $byIndex.ContainsKey($key) -or $record.binding -is [array] -or $byIndex[$key].binding -isnot [array]){$byIndex[$key]=$record}
    }
    $records=@($withoutIndex)+@($byIndex.Values)
    $groups=@($records|Group-Object task,gold,type)
    if($job.expectedGroups){
      $groups=@()
      foreach($property in $job.expectedGroups.PSObject.Properties){
        $groupKey=$property.Name
        $members=@($records|Where-Object{($_.task+'/'+$_.gold) -eq $groupKey})
        $groups+=[pscustomobject]@{Name=$groupKey;Group=$members;Expected=[int]$property.Value}
      }
    }
    foreach($group in $groups){
      $attempted=@($group.Group).Count
      $expected=if($null -ne $group.Expected){$group.Expected}else{$attempted}
      foreach($condition in 'base','ledger','linked','memory','binding'){
        $valid=@($group.Group|Where-Object{$_.$condition -is [array]})
        $correct=0
        $expectedConditions=if($job.conditions){$job.conditions}else{@('base','binding')}
        if(-not $valid.Count -and (-not $job.expectedGroups -or $condition -notin $expectedConditions)){continue}
        foreach($r in $valid){$scores=$r.$condition;$best=0;for($i=1;$i -lt $scores.Count;$i++){if($scores[$i] -gt $scores[$best]){$best=$i}};if(($r.gold -is [string] -and [string][char](65+$best) -eq $r.gold) -or ($r.gold -isnot [string] -and $best -eq $r.gold)){$correct++}}
        $errors=@($group.Group|Where-Object{($_.$condition -isnot [array]) -and ($_.error -or $_.errors_by_condition.$condition)}).Count
        $all+=[pscustomobject]@{Experiment=$job.name;Group=$group.Name;Method=$condition;Correct=$correct;Wrong=$valid.Count-$correct;Errors=$errors;Remaining=$expected-$valid.Count;Target=$expected}
      }
    }
  }
}
$all|Format-Table -AutoSize
