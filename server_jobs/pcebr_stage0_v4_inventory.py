"""Read-only inventory; returns experiment filenames, sizes and runtime health."""
import json
import torch

RESULT = {'out': str(OUT), 'server': globals().get('ARGON_SERVER'), 'files': [], 'cuda_ok': False}
for pattern in ('uploads/*pcebr*', 'uploads/avbind*', 'uploads/*manifest*',
                'uploads/chronodesmear*', 'argon_latent40/decoder_cache/*rot0.pt'):
    for path in OUT.glob(pattern):
        if path.is_file():
            RESULT['files'].append({'name': str(path.relative_to(OUT)), 'bytes': path.stat().st_size})
RESULT['cuda_ok'] = bool(torch.isfinite(torch.ones(1, device='cuda:0') + torch.ones(1, device='cuda:0')).all())
print(json.dumps(RESULT), flush=True)
