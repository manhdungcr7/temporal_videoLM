param([int]$Server=1)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$files=@('repo/kaggle/jobs/avbind_reader.py','repo/kaggle/jobs/avbind_neural.py',
 'repo/kaggle/jobs/avbind_native.py','repo/kaggle/jobs/avbind_synthetic.py',
 'repo/kaggle/jobs/avbind_binding_cases.py','server_jobs/avbind_train_structural.py',
 'server_jobs/avbind_train_structural_r2.py','server_jobs/avbind_eval_structural.py',
 'server_jobs/avbind_neural_pilot.py')
foreach($file in $files){
 & (Join-Path $PSScriptRoot 'cq_lab.ps1') -Action Upload -Server $Server -File ([IO.Path]::GetFileName($file)) -Dest (Join-Path $root $file)
}
& (Join-Path $PSScriptRoot 'cq_lab.ps1') -Action Submit -Server $Server -File (Join-Path $PSScriptRoot 'avbind_neural_load.py') -Name r3_reader_load -Bootstrap (Join-Path $PSScriptRoot 'avbind_neural_source_r3.py')
& (Join-Path $PSScriptRoot 'cq_lab.ps1') -Action Submit -Server $Server -File (Join-Path $PSScriptRoot 'avbind_native_load.py') -Name r3_native_load
& (Join-Path $PSScriptRoot 'cq_lab.ps1') -Action Submit -Server $Server -File (Join-Path $PSScriptRoot 'avbind_train_structural_r3.py') -Name r3_train
& (Join-Path $PSScriptRoot 'cq_lab.ps1') -Action Submit -Server $Server -File (Join-Path $PSScriptRoot 'avbind_eval_structural_r3.py') -Name r3_structural_eval
& (Join-Path $PSScriptRoot 'cq_lab.ps1') -Action Submit -Server $Server -File (Join-Path $PSScriptRoot 'avbind_native_pilot_r3.py') -Name r3_native_pilot
