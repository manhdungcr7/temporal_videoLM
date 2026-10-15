param([string]$Round='1')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$records=@()
foreach($s in 1,2){
  $path=Join-Path $root "results/cq_diagnostic_round${Round}_s${s}.jsonl"
  if(Test-Path -LiteralPath $path){
    $records+=@(Get-Content -LiteralPath $path|Where-Object{$_.Trim()}|ForEach-Object{$_|ConvertFrom-Json})
  }
}
$valid=@($records|Where-Object{-not $_.error})
Write-Output "Round $Round : records=$($records.Count) valid=$($valid.Count) errors=$($records.Count-$valid.Count)"
if(-not $valid.Count){exit}
$names=@($valid[0].PSObject.Properties.Name|Where-Object{$_ -eq 'base' -or ($_ -like 'cq_*' -and $_ -notlike '*null*')})
$table=foreach($name in $names){
  $entry=[ordered]@{Method=$name}
  foreach($task in 'TA','TF'){
    $subset=@($valid|Where-Object{$_.task -eq $task -and $_.$name -is [array]})
    $good=0;$hist=@{}
    foreach($r in $subset){
      $arr=$r.$name;$best=0
      for($i=1;$i -lt $arr.Count;$i++){if([double]$arr[$i] -gt [double]$arr[$best]){$best=$i}}
      if($best -eq [int]$r.gold){$good++}
      $label=[string][char](65+$best)
      if(-not $hist.ContainsKey($label)){$hist[$label]=0}
      $hist[$label]++
    }
    $entry[$task]="$good/$($subset.Count)"
    $entry[$task+'_predictions']=($hist.Keys|Sort-Object|ForEach-Object{"${_}:$($hist[$_])"})-join ' '
  }
  [pscustomobject]$entry
}
$table|Format-Table -AutoSize

# Keep minority-class failures visible instead of pooling all TA/TF labels.
$groupTable=foreach($name in $names){
  $eligible=@($valid|Where-Object{$_.$name -is [array]})
  foreach($group in ($eligible|Group-Object task,gold|Sort-Object Name)){
    $good=0;$hist=@{}
    foreach($r in $group.Group){
      $scores=$r.$name;$best=0
      for($i=1;$i -lt $scores.Count;$i++){if([double]$scores[$i] -gt [double]$scores[$best]){$best=$i}}
      if($best -eq [int]$r.gold){$good++}
      $label=[string][char](65+$best)
      if(-not $hist.ContainsKey($label)){$hist[$label]=0}
      $hist[$label]++
    }
    [pscustomobject]@{
      Method=$name;Task=$group.Group[0].task
      Gold=[string][char](65+[int]$group.Group[0].gold)
      Correct=$good;Wrong=$group.Count-$good;N=$group.Count
      Predictions=($hist.Keys|Sort-Object|ForEach-Object{"${_}:$($hist[$_])"})-join ' '
    }
  }
}
$groupTable|Format-Table -AutoSize
