path = '/kaggle/working/svhalluc_prompt_lab_runner/env/lib/python3.13/site-packages/qwen_omni_utils/v2_5/vision_process.py'
lines = open(path).readlines()
for i in range(330, 380):
    if i < len(lines):
        print(f"{i+1}: {lines[i]}", end='')
RESULT = "DONE"

