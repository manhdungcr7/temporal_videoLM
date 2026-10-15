param(
  [ValidateSet('Submit','Status','Download','Upload','Ping','Restart')][string]$Action='Status',
  [int]$Server=1,
  [string]$File,
  [string]$Job,
  [string]$Name='cq_lab',
  [string]$Dest,
  [int]$Tail=0,
  [string[]]$Prelude,
  [string]$Bootstrap
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$cfgName=if($Server -eq 1){'remote_config.json'}else{'remote_config2.json'}
$cfg=Get-Content -Raw -LiteralPath (Join-Path $root $cfgName)|ConvertFrom-Json
$headers=@{'X-Token'=$cfg.token;'ngrok-skip-browser-warning'='1'}
if($Action -eq 'Submit'){
  $utils=Get-Content -Raw -LiteralPath (Join-Path $root 'repo/kaggle/jobs/job_utils.py')
  $j8=Get-Content -Raw -LiteralPath (Join-Path $root 'repo/kaggle/jobs/j8_acar.py')
  $j8=($j8 -split '\r?\nitems = sample_items',2)[0]
  $loc=Get-Content -Raw -LiteralPath (Join-Path $root 'remote_tests/debug_parallel.py')
  $loc=$loc.Substring($loc.IndexOf('def locate'))
  $loc=($loc -split '\r?\ndef parallel_verify',2)[0]
  $j12=Get-Content -Raw -LiteralPath (Join-Path $root 'repo/kaggle/jobs/j12_chunk_prompts.py')
  $j12=($j12 -split '\r?\nTASKS = ',2)[0]
  $taskCode=Get-Content -Raw -LiteralPath $File
  if($Prelude){
    $helperCode=''
    foreach($part in $Prelude){
      $helpers=Get-Content -Raw -LiteralPath $part
      $helpers=($helpers -split '\r?\nCONDITIONS\s*=',2)[0]
      $helperCode+=$helpers+"`n"
    }
    $taskCode=$helperCode+$taskCode
  }
  $prefix="time_left = lambda: 10 ** 9`nJ8_MAX_DUR = 50.0`nPROGRESS_GROUPS = True`nLAB_SERVER = $Server`n"
  if($Bootstrap){$prefix+=(Get-Content -Raw -LiteralPath $Bootstrap)+"`n"}
  $code=$prefix+$utils+"`n"+$j8+"`nTOK = PROC.tokenizer`n"+$loc+"`n"+$j12+"`n"+$taskCode
  $body=@{code=$code;name=$Name}|ConvertTo-Json -Compress
  Invoke-RestMethod -Method Post -Uri ($cfg.url+'/submit') -Headers $headers -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 60 | ConvertTo-Json -Depth 12
} elseif($Action -eq 'Status'){
  $state=Invoke-RestMethod -Uri ($cfg.url+'/job/'+$Job+'?since=0') -Headers $headers -TimeoutSec 40
  if($Tail -gt 0){$state.out=($state.out -split '\n'|Select-Object -Last $Tail)-join "`n"}
  $state | ConvertTo-Json -Depth 16
} elseif($Action -eq 'Download'){
  Invoke-WebRequest -UseBasicParsing -Uri ($cfg.url+'/file?path='+[uri]::EscapeDataString($File)) -Headers $headers -OutFile $Dest -TimeoutSec 60
  Write-Output "Downloaded $File"
} elseif($Action -eq 'Upload'){
  $body=@{path=$File;b64=[Convert]::ToBase64String([IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $Dest)))}|ConvertTo-Json -Compress
  Invoke-RestMethod -Method Post -Uri ($cfg.url+'/upload') -Headers $headers -ContentType 'application/json' -Body $body -TimeoutSec 60|ConvertTo-Json
} elseif($Action -eq 'Restart'){
  Invoke-RestMethod -Method Post -Uri ($cfg.url+'/restart') -Headers $headers -ContentType 'application/json' -Body '{}' -TimeoutSec 20 | ConvertTo-Json
} else {
  Invoke-RestMethod -Uri ($cfg.url+'/info') -Headers $headers -TimeoutSec 20 | ConvertTo-Json -Depth 5
}
