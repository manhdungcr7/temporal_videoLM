param([int]$Server=1)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$paths=@(
  'server_jobs/cq_diagnostic_round3.py','server_jobs/cq_diagnostic_round6.py',
  'server_jobs/cq_diagnostic_round7.py','server_jobs/cq_diagnostic_round8.py',
  'server_jobs/cq_diagnostic_round9.py','server_jobs/cq_diagnostic_round10.py',
  'server_jobs/cq_diagnostic_round11.py','server_jobs/cq_diagnostic_round12.py',
  'server_jobs/cq_diagnostic_round13.py','server_jobs/cq_diagnostic_round14.py',
  'server_jobs/cq_diagnostic_round15.py','repo/kaggle/jobs/cq_provenance_answer.py'
)
$sources=@($paths|ForEach-Object{@{path=$_;source=[IO.File]::ReadAllText((Join-Path $root $_))}})
$bundle=@{sources=$sources}|ConvertTo-Json -Depth 5 -Compress
$cfgName=if($Server -eq 1){'remote_config.json'}else{'remote_config2.json'}
$cfg=Get-Content -Raw -LiteralPath (Join-Path $root $cfgName)|ConvertFrom-Json
$headers=@{'X-Token'=$cfg.token;'ngrok-skip-browser-warning'='1'}
$body=@{path='cq_inference_sources.json';b64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($bundle))}|ConvertTo-Json -Compress
Invoke-RestMethod -Method Post -Uri ($cfg.url+'/upload') -Headers $headers -ContentType 'application/json' -Body $body -TimeoutSec 60|ConvertTo-Json
