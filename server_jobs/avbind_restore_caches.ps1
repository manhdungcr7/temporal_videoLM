param([int]$Server=1)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$cfgName=if($Server -eq 1){'remote_config.json'}else{'remote_config2.json'}
$cfg=Get-Content -Raw (Join-Path $root $cfgName)|ConvertFrom-Json
$headers=@{'X-Token'=$cfg.token;'ngrok-skip-browser-warning'='1'}
$items=@{}
foreach($file in (Get-ChildItem (Join-Path $root 'results') -Filter 'avbind*.jsonl')){
  foreach($line in (Get-Content -LiteralPath $file.FullName)){
    if(-not $line.Trim()){continue}
    $record=$line|ConvertFrom-Json
    if($null -eq $record.idx -or -not $record.task){continue}
    foreach($property in $record.PSObject.Properties){
      if($property.Name -notlike '*_trace' -or -not $property.Value.nodes){continue}
      $observations=@{}
      $speech=@()
      foreach($node in $property.Value.nodes){
        if($node.id -match '^([VA])(\d+)$'){
          $key=[int]$Matches[2]
          if(-not $observations.ContainsKey($key)){$observations[$key]=@{start=$node.interval[0];end=$node.interval[1];video='';audio='';rms=0}}
          $field=if($Matches[1] -eq 'V'){'video'}else{'audio'}
          $observations[$key][$field]=$node.content
        }elseif($node.id -match '^S\d+$'){
          $speech+=@{start=$node.interval[0];end=$node.interval[1];words=$node.content}
        }
      }
      if($observations.Count){
        $items['cq_physical_memory_'+$record.idx+'.json']=@($observations.Keys|Sort-Object|ForEach-Object{$observations[$_]})
        $items['avbind_utterances_'+$record.idx+'.json']=$speech
        break
      }
    }
  }
}
# Preserve the original independently extracted observations when available.
foreach($file in (Get-ChildItem (Join-Path $root 'results') -Filter 'cq_physical_memory_*.json')){
  # Array subexpressions strip ConvertFrom-Json's pipeline wrapper properties,
  # which Windows PowerShell otherwise serializes as {value:[...],Count:N}.
  $items[$file.Name]=@(Get-Content -Raw -LiteralPath $file.FullName|ConvertFrom-Json|ForEach-Object{$_})
}
$bundle=ConvertTo-Json -InputObject $items -Depth 16 -Compress
$body=@{path='avbind_restore_bundle.json';b64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($bundle))}|ConvertTo-Json -Compress
Invoke-RestMethod -Method Post -Uri ($cfg.url+'/upload') -Headers $headers -ContentType 'application/json' -Body $body -TimeoutSec 60|ConvertTo-Json
& (Join-Path $PSScriptRoot 'cq_lab.ps1') -Action Submit -Server $Server -File (Join-Path $PSScriptRoot 'avbind_restore_caches.py') -Name restore_independent_caches
