param(
 [ValidateSet('Submit','Status','Upload','Get')][string]$Action,
 [ValidateSet(1,2)][int]$Server=1,
 [string]$File,[string]$Remote,[string]$Job,[string]$Name='pcebr_v8',
 [string]$Save
)
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$root=Split-Path -Parent $PSScriptRoot
$configFile=if($Server -eq 1){'remote_config.json'}else{'remote_config2.json'}
$connection=Get-Content -Raw -LiteralPath (Join-Path $root $configFile)|ConvertFrom-Json
$headers=@{'X-Token'=$connection.token;'ngrok-skip-browser-warning'='1'}
$enc=[Text.UTF8Encoding]::new($false)
switch($Action){
 'Submit' {
  $code="PC_SERVER = $Server`nLAB_SERVER = $Server`n"+[IO.File]::ReadAllText((Resolve-Path -LiteralPath $File),[Text.Encoding]::UTF8)
  $body=@{code=$code;name=$Name}|ConvertTo-Json -Compress
  $result=Invoke-RestMethod -Method Post -Uri ($connection.url+'/submit') -Headers $headers -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 50
 }
 'Status' {
  $result=Invoke-RestMethod -Uri ($connection.url+'/job/'+$Job) -Headers $headers -TimeoutSec 40
 }
 'Upload' {
  $body=@{path=$Remote;b64=[Convert]::ToBase64String([IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $File)))}|ConvertTo-Json -Compress
  $result=Invoke-RestMethod -Method Post -Uri ($connection.url+'/upload') -Headers $headers -ContentType 'application/json' -Body $body -TimeoutSec 90
 }
 'Get' {
  Invoke-WebRequest -UseBasicParsing -Uri ($connection.url+'/file?path='+[uri]::EscapeDataString($Remote)) -Headers $headers -OutFile $File -TimeoutSec 180
  Write-Output "saved $File"
  exit
 }
}
$json=$result|ConvertTo-Json -Depth 70
if($Save){[IO.File]::WriteAllText((Join-Path $root $Save),$json,$enc)}
if($Action -eq 'Status'){
 [pscustomobject]@{server=$Server;status=$result.status;seconds=$result.seconds;error=$result.error;tail=($result.out -split "`n"|Select-Object -Last 5)-join "`n"}|ConvertTo-Json -Depth 5
}else{$json}
