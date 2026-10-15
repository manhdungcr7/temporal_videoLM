param(
 [ValidateSet('Info','Submit','Status','Upload','UploadRuntime','Download')][string]$Action='Info',
 [ValidateSet(1,2)][int]$Server=1,
 [string]$File,[string]$Remote,[string]$Job,[string]$Name='research40',
 [int]$Tail=30,[switch]$IncludeResult
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$configFile=if($Server -eq 1){'remote_config.json'}else{'remote_config2.json'}
$connection=Get-Content -Raw -LiteralPath (Join-Path $root $configFile)|ConvertFrom-Json
$headers=@{'X-Token'=$connection.token;'ngrok-skip-browser-warning'='1'}
switch($Action){
 'Info' {Invoke-RestMethod -Uri ($connection.url+'/info') -Headers $headers -TimeoutSec 20|ConvertTo-Json -Depth 10}
 'Submit' {
  [string]$code=[IO.File]::ReadAllText((Resolve-Path -LiteralPath $File),[Text.Encoding]::UTF8)
  $code="R40_SERVER = $Server`n"+$code
  $body=@{code=$code;name=$Name}|ConvertTo-Json -Compress
  Invoke-RestMethod -Method Post -Uri ($connection.url+'/submit') -Headers $headers -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 40|ConvertTo-Json
 }
 'Status' {
  $state=Invoke-RestMethod -Uri ($connection.url+'/job/'+$Job) -Headers $headers -TimeoutSec 30
  if($Tail -gt 0){$state.out=($state.out -split '\n'|Select-Object -Last $Tail)-join "`n"}
  if(-not $IncludeResult){$state.result=$null}
  $state|ConvertTo-Json -Depth 20
 }
 'Upload' {
  $body=@{path=$Remote;b64=[Convert]::ToBase64String([IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $File)))}|ConvertTo-Json -Compress
  Invoke-RestMethod -Method Post -Uri ($connection.url+'/upload') -Headers $headers -ContentType 'application/json' -Body $body -TimeoutSec 40|ConvertTo-Json
 }
 'UploadRuntime' {
  foreach($part in @('research40_baseline.py','research40_attention.py','research40_reasoning.py','research40_memory.py')){
   $source=Join-Path $PSScriptRoot $part
   $body=@{path=$part;b64=[Convert]::ToBase64String([IO.File]::ReadAllBytes($source))}|ConvertTo-Json -Compress
   $response=Invoke-RestMethod -Method Post -Uri ($connection.url+'/upload') -Headers $headers -ContentType 'application/json' -Body $body -TimeoutSec 40
   Write-Output "Runtime $part : $($response.bytes) bytes"
  }
 }
 'Download' {
  Invoke-WebRequest -UseBasicParsing -Uri ($connection.url+'/file?path='+[uri]::EscapeDataString($Remote)) -Headers $headers -OutFile $File -TimeoutSec 45
  Write-Output "Saved $File"
 }
}
