import os,sys
from huggingface_hub import snapshot_download
p=snapshot_download('Qwen/Qwen2.5-Omni-3B',revision='f75b40e3da2003cdd6e1829b1f420ca70797c34e',
    cache_dir=sys.argv[1],allow_patterns=['*.json','*.safetensors','*.txt','*.model','merges.txt','vocab*'])
print('DONE',p)
