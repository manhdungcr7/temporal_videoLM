param([switch]$LocalOnly,[ValidateSet('all','verified_graph_v2','proposition_critic','stca','reader4_plain','reader4_reason','atomic_provenance','atomic_provenance_v2','pts_clock')][string]$Phase='all')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$manifest=Get-Content -Raw -LiteralPath (Join-Path $root 'results/research40_manifest.json')|ConvertFrom-Json
if(-not $LocalOnly){
 foreach($s in @(1,2)){
  $cf=if($s -eq 1){'remote_config.json'}else{'remote_config2.json'}
  $cfg=Get-Content -Raw -LiteralPath (Join-Path $root $cf)|ConvertFrom-Json
  $headers=@{'X-Token'=$cfg.token;'ngrok-skip-browser-warning'='1'}
  $files=if($Phase -eq 'all'){@("research40_progress_s$s.json","research40_baseline_s$s.jsonl","research40_attention_v2_s$s.jsonl","research40_reasoning_s$s.jsonl","research40_raw_kv_s$s.jsonl","research40_faithful_memory_s$s.jsonl","research40_interval_graph_s$s.jsonl","research40_source_contrast_s$s.jsonl","research40_verified_graph_s$s.jsonl","research40_verified_graph_v2_s$s.jsonl","research40_proposition_critic_s$s.jsonl","research40_stca_s$s.jsonl","research40_reader4_plain_s$s.jsonl","research40_reader4_reason_s$s.jsonl","research40_atomic_provenance_s$s.jsonl","research40_atomic_provenance_v2_s$s.jsonl","research40_pts_clock_s$s.jsonl")}else{@("research40_progress_s$s.json","research40_${Phase}_s$s.jsonl")}
  foreach($file in $files){
   $target=Join-Path $root ('results/'+$file)
   $pending=$target+'.download'
   try{
    Invoke-WebRequest -UseBasicParsing -Uri ($cfg.url+'/file?path='+$file) -Headers $headers -OutFile $pending -TimeoutSec 20
    if($file.EndsWith('.jsonl')){
     $incoming=@{}
     foreach($line in [IO.File]::ReadAllLines($pending)){
      $r=$line|ConvertFrom-Json
      if($r.status -ne 'ok' -or $r.manifest_sha256 -ne $manifest.manifest_sha256){throw 'Invalid result manifest/status'}
      $incoming[[string]$r.idx]=$true
     }
     if(Test-Path -LiteralPath $target){
      foreach($line in [IO.File]::ReadAllLines($target)){
       $r=$line|ConvertFrom-Json
       if(-not $incoming.ContainsKey([string]$r.idx)){throw 'Remote is missing completed local IDs; retaining local results'}
      }
     }
    }else{
     $r=[IO.File]::ReadAllText($pending)|ConvertFrom-Json
     if($r.manifest_sha256 -ne $manifest.manifest_sha256){throw 'Invalid progress manifest'}
    }
    Move-Item -LiteralPath $pending -Destination $target -Force
   }catch{if($file -like '*baseline*' -or $file -like '*progress*'){Write-Output "Server ${s}: $($_.Exception.Message)"}}
   finally{if(Test-Path -LiteralPath $pending){Remove-Item -LiteralPath $pending}}
  }
 }
}
$lookup=@{}
foreach($r in $manifest.selected){$lookup[[string]$r.idx]=$r}
$records=@{}
Get-ChildItem -LiteralPath (Join-Path $root 'results') -Filter 'research40_*_s*.jsonl'|Where-Object{$_.Name -notlike 'research40_attention_s*' -and $_.Name -notlike 'research40_memory_s*'}|ForEach-Object{
 foreach($line in [IO.File]::ReadAllLines($_.FullName)){
  try{$rec=$line|ConvertFrom-Json}catch{continue}
  if($rec.status -ne 'ok' -or $rec.manifest_sha256 -ne $manifest.manifest_sha256){continue}
  $id=[string]$rec.idx
  if(-not $records.ContainsKey($id)){$records[$id]=@{}}
  foreach($prop in $rec.scores.PSObject.Properties){$records[$id][$prop.Name]=@($prop.Value)}
 }
}
$methods=@($records.Values|ForEach-Object{$_.Keys}|Sort-Object -Unique)
if($Phase -ne 'all'){
 $pattern=switch($Phase){
  'verified_graph_v2' {'^verified_v2_'}
  'proposition_critic' {'^proposition_'}
  'stca' {'^(stca_|frozen_text_verifier$)'}
  'reader4_plain' {'^reader4_(plain|question_only)$'}
  'reader4_reason' {'^reader4_reason$'}
  'atomic_provenance' {'^atomic_(raw|reader)'}
  'atomic_provenance_v2' {'^atomic_v2_'}
  'pts_clock' {'^pts_'}
 }
 $methods=@($methods|Where-Object{$_ -eq 'base' -or $_ -match $pattern})
}
$summary=@()
foreach($method in $methods){
 $total=0;$correct=0;$cells=@{}
 foreach($id in $records.Keys){
  if(-not $records[$id].ContainsKey($method)){continue}
  $r=$lookup[$id];$scores=$records[$id][$method]
  $best=0;for($i=1;$i -lt $scores.Count;$i++){if($scores[$i] -gt $scores[$best]){$best=$i}}
  $ok=($best -eq ([int][char]$r.gold-65))
  $total++;if($ok){$correct++}
  $cell=$r.task+'/'+$r.gold
  if(-not $cells.ContainsKey($cell)){$cells[$cell]=@{right=0;wrong=0}}
  if($ok){$cells[$cell].right++}else{$cells[$cell].wrong++}
 }
 $row=[ordered]@{method=$method;total="$correct/$total"}
 foreach($cell in @('TA/A','TA/B','TF/A','TF/B','TF/C','CMTB/A','CMTB/B','CMTB/C')){
  $c=$cells[$cell];$row[$cell]=if($c){"$($c.right)D $($c.wrong)S"}else{'0D 0S'}
 }
 $summary+=[pscustomobject]$row
}
Write-Output "Fixed40 $($manifest.manifest_sha256.Substring(0,12)); D=correct, S=wrong; processed denominators shown."
$summary|Format-Table -AutoSize
