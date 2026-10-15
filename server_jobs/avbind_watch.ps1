param([int]$IntervalSeconds=30,[int]$Iterations=80,[string]$NameLike='*')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$lastCheckpoint=@{}
$finished=@{}
for($round=0;$round -lt $Iterations;$round++){
 $entries=(Get-Content -Raw (Join-Path $root 'results/avbind_active_jobs.json')|ConvertFrom-Json).jobs
 $unfinished=0
 foreach($entry in $entries){
  if($entry.name -notlike $NameLike){continue}
  if($finished.ContainsKey($entry.id)){continue}
  $configName=if($entry.server -eq 1){'remote_config.json'}else{'remote_config2.json'}
  $connection=Get-Content -Raw (Join-Path $root $configName)|ConvertFrom-Json
  $headers=@{'X-Token'=$connection.token;'ngrok-skip-browser-warning'='1'}
  try{
   $state=Invoke-RestMethod -Uri ($connection.url+'/job/'+$entry.id) -Headers $headers -TimeoutSec 15
   if($state.status -in @('queued','running')){$unfinished++}
   $tail=($state.out -split '\n'|Where-Object{$_ -match 'BIND_TRAIN |BIND_REAL_PILOT|BIND_STRUCTURAL_RESULT|ADDRESS_CACHE|ADDRESS_PRETRAIN'}|Select-Object -Last 1)
   Write-Output "$([DateTime]::Now.ToString('HH:mm:ss')) S$($entry.server) $($entry.name) $($state.status) $tail"
   if($state.error){Write-Output $state.error.Substring([Math]::Max(0,$state.error.Length-600))}
   if($entry.file){
    try{
     $response=Invoke-WebRequest -UseBasicParsing -Uri ($connection.url+'/file?path='+[uri]::EscapeDataString($entry.file)) -Headers $headers -TimeoutSec 30
     [IO.File]::WriteAllBytes((Join-Path $root ('results/'+$entry.name+'_s'+$entry.server+'.jsonl')),$response.RawContentStream.ToArray())
    }catch{}
   }
   if($state.status -in @('done','error')){$finished[$entry.id]=$true}
   if($entry.kind -eq 'training'){
    $steps=[regex]::Matches($state.out,'"step":\s*(\d+)')
    $step=if($steps.Count){[int]$steps[$steps.Count-1].Groups[1].Value}else{0}
    $checkpointStep=[int][Math]::Floor($step/64)*64
    if($checkpointStep -gt 0 -and $lastCheckpoint[$entry.server] -ne $checkpointStep){
     try{
      $target=Join-Path $root ('results/avbind_training_latest_s'+$entry.server+'.pt')
      Invoke-WebRequest -UseBasicParsing -Uri ($connection.url+'/file?path=avbind_training_latest.pt') -Headers $headers -OutFile $target -TimeoutSec 60
      $lastCheckpoint[$entry.server]=$checkpointStep
      Write-Output "Saved recoverable S$($entry.server) training checkpoint (~step $checkpointStep)"
     }catch{}
    }
    if($state.status -eq 'done'){
     foreach($name in @('avbind_reader_checkpoint_r3.pt','avbind_reader_metadata_r3.json','avbind_structural_holdout_r3.json')){
      $extension=[IO.Path]::GetExtension($name);$stem=[IO.Path]::GetFileNameWithoutExtension($name)
      try{Invoke-WebRequest -UseBasicParsing -Uri ($connection.url+'/file?path='+$name) -Headers $headers -OutFile (Join-Path $root ('results/'+$stem+'_s'+$entry.server+$extension)) -TimeoutSec 60}catch{}
     }
    }
   }
  }catch{Write-Output "S$($entry.server) temporarily unreachable for $($entry.name)";$unfinished++}
 }
 if($unfinished -eq 0){break}
 Start-Sleep -Seconds $IntervalSeconds
}
