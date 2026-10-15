param([string]$ExtractS1='2bf75cbdb9',[string]$ExtractS2='a131ed2981')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
function Read-Job([int]$Server,[string]$Id){
 $cfgName=if($Server -eq 1){'remote_config.json'}else{'remote_config2.json'}
 $cfg=Get-Content -Raw $cfgName|ConvertFrom-Json
 Invoke-RestMethod -Uri ($cfg.url+'/job/'+$Id) -Headers @{'X-Token'=$cfg.token;'ngrok-skip-browser-warning'='1'} -TimeoutSec 20
}
function Submit-Stage([int]$Server,[string]$File,[string]$Name){
 $existing=(Get-Content -Raw results/avbind_active_jobs.json|ConvertFrom-Json).jobs|Where-Object{$_.server -eq $Server -and $_.name -eq $Name}|Select-Object -Last 1
 if($existing){
  $current=Read-Job $Server $existing.id
  if($current.status -in @('queued','running','done')){return $existing.id}
  throw "Stage $Name has a failed job $($existing.id); fix its recorded error before submitting a missing-stage retry."
 }
 $reply=& server_jobs/cq_lab.ps1 -Action Submit -Server $Server -File $File -Name $Name
 ($reply -join "`n"|ConvertFrom-Json).job_id
}
function Save-Entry([int]$Server,[string]$Id,[string]$Name,[string]$File,[string]$Kind='evaluation'){
 $registryData=Get-Content -Raw results/avbind_active_jobs.json|ConvertFrom-Json
 if(@($registryData.jobs|Where-Object{$_.id -eq $Id}).Count){return}
 $entry=@{server=$Server;id=$Id;name=$Name;file=$File;kind=$Kind}
 if($Kind -eq 'evaluation'){
  $entry.expectedGroups=if($Server -eq 1){@{'TA/B'=5;'TA/A'=2}}else{@{'TF/C'=5;'TF/A'=2;'TF/B'=2}}
 }
 $registryData.jobs=@($registryData.jobs)+@([pscustomobject]$entry)
 [IO.File]::WriteAllText((Join-Path $root 'results/avbind_active_jobs.json'),($registryData|ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
}
function Wait-Stage([int]$Server,[string]$Id){
 while($true){
  $state=Read-Job $Server $Id
  $tail=$state.out -split '\n'|Where-Object{$_ -match 'NATIVE_POOL_EXTRACT|NATIVE_POOL_ALIGN_TRAIN|BIND_REAL_PILOT'}|Select-Object -Last 1
  Write-Host ("$([DateTime]::Now.ToString('HH:mm:ss')) S$Server $Id $($state.status) $tail")
  if($state.status -eq 'error'){throw "S$Server job $Id failed: $($state.error)"}
  if($state.status -eq 'done'){return $state}
  Start-Sleep -Seconds 30
 }
}
Save-Entry 1 $ExtractS1 'r18-media-pool-pairs' 'avbind_pooled_pairs_s1.jsonl' 'feature_extraction'
Save-Entry 2 $ExtractS2 'r18-media-pool-pairs' 'avbind_pooled_pairs_s2.jsonl' 'feature_extraction'
foreach($s in 1,2){
 $id=if($s -eq 1){$ExtractS1}else{$ExtractS2}
 $state=@(Wait-Stage $s $id)[-1]
 if($state.result.completed -ne $state.result.target){throw "S$s pooled extraction incomplete; resume missing pairs only"}
 foreach($ext in 'pt','jsonl'){
  $file="avbind_pooled_pairs_s$s.$ext"
  & server_jobs/cq_lab.ps1 -Action Download -Server $s -File $file -Dest ("results/"+$file)
 }
 & server_jobs/cq_lab.ps1 -Action Upload -Server 1 -File "avbind_pooled_pairs_s$s.pt" -Dest "results/avbind_pooled_pairs_s$s.pt"
}
$training=Submit-Stage 1 'server_jobs/avbind_pooled_alignment_train.py' 'r18-pooled-alignment-train'
Save-Entry 1 $training 'r18-pooled-alignment-train' '' 'alignment_training'
$trained=@(Wait-Stage 1 $training)[-1]
Write-Output ('R18_TRAINING_RESULT '+($trained.result.metadata.pseudo_description_match_correct|ConvertTo-Json -Compress))
foreach($ext in 'pt','json'){
 & server_jobs/cq_lab.ps1 -Action Download -Server 1 -File "avbind_native_alignment_r18.$ext" -Dest "results/avbind_native_alignment_r18.$ext"
}
foreach($s in 1,2){
 $existingPilot=(Get-Content -Raw results/avbind_active_jobs.json|ConvertFrom-Json).jobs|Where-Object{$_.server -eq $s -and $_.name -eq 'r18-native-pooled-route'}|Select-Object -Last 1
 if($existingPilot){Write-Host "S$s R18 pilot already submitted ($($existingPilot.id)); skip completed upload/load/submit stages.";continue}
 & server_jobs/cq_lab.ps1 -Action Upload -Server $s -File 'avbind_native_alignment_r18.pt' -Dest results/avbind_native_alignment_r18.pt
 $load=Submit-Stage $s 'server_jobs/avbind_pooled_guidance_load.py' 'r18-pooled-guidance-load'
 Save-Entry $s $load 'r18-pooled-guidance-load' '' 'loader'
 $loaded=@(Wait-Stage $s $load)[-1]
 $pilot=Submit-Stage $s 'server_jobs/avbind_pooled_guidance_pilot.py' 'r18-native-pooled-route'
 Save-Entry $s $pilot 'r18-native-pooled-route' "avbind_pooled_guidance_r18_s$s.jsonl"
}
while($true){
 & server_jobs/avbind_progress.ps1 -Download -NameLike 'r18-native-pooled-route'
 $entries=(Get-Content -Raw results/avbind_active_jobs.json|ConvertFrom-Json).jobs|Where-Object{$_.name -eq 'r18-native-pooled-route'}
 $running=$false
 foreach($entry in $entries){$s=Read-Job $entry.server $entry.id;if($s.status -eq 'error'){throw $s.error};if($s.status -ne 'done'){$running=$true}}
 if(-not $running){break}
 Start-Sleep -Seconds 30
}
Write-Output 'R18_PIPELINE_COMPLETE: check per-group metrics; completion is not a claim of method success.'
