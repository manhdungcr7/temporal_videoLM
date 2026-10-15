param([int]$Server=1,[ValidateSet('r3','r4')][string]$Variant='r3')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$held=(Get-Content -Raw (Join-Path $root "results/avbind_structural_holdout_r3_s$Server.json")|ConvertFrom-Json)
$name=if($Variant -eq 'r4'){'r4-address-structure'}elseif($Server -eq 1){'r3-supervised-structure'}else{'r3-answer-only-structure'}
$records=@(Get-Content (Join-Path $root "results/${name}_s$Server.jsonl")|ForEach-Object{$_|ConvertFrom-Json})
$output=foreach($hop in 0,1){
 $tp=0;$fp=0;$fn=0;$nullCorrect=0;$topCorrect=0;$positiveRows=0
 foreach($record in $records){
  $row=$held[$record.idx];$expected=@($row.evidence.('hop'+($hop+1)))
  if($hop -eq 0 -and $expected.Count){
   $refs=@($row.nodes|Where-Object{$expected -contains $_.id})
   $expected=@($row.nodes|Where-Object{$node=$_; @($refs|Where-Object{$_.source -eq $node.source -and $_.content -eq $node.content}).Count}|ForEach-Object{$_.id})
  }
  $audit=$record.audits[0][$hop];$support=$audit.support[0][0]
  $nullValue=[double]$audit.null[0][0]
  if(($nullValue -gt .5) -eq (-not $expected.Count)){$nullCorrect++}
  $best=0
  for($j=0;$j -lt $row.nodes.Count;$j++){
   if($support[$j] -gt $support[$best]){$best=$j}
   $gold=$expected -contains $row.nodes[$j].id;$selected=$support[$j] -gt .5
   if($gold -and $selected){$tp++}elseif($gold){$fn++}elseif($selected){$fp++}
  }
  if($expected.Count){$positiveRows++;if($expected -contains $row.nodes[$best].id){$topCorrect++}}
 }
 [pscustomobject]@{Server=$Server;Hop=$hop+1;N=$records.Count;TP=$tp;FP=$fp;FN=$fn;
  Precision=[Math]::Round($tp/[Math]::Max(1,$tp+$fp),3);Recall=[Math]::Round($tp/[Math]::Max(1,$tp+$fn),3);
  NullCorrect=$nullCorrect;TopCorrect=$topCorrect;PositiveRows=$positiveRows}
}
$output|Format-Table -AutoSize
