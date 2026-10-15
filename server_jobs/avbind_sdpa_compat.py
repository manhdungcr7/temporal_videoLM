"""T4 memory fix: explicit KV repeats instead of unsupported fused GQA.

No media reduction, mask/position edits, or attention-score modification.
Validate the algebra on the GPU before enabling the compatibility path.
"""
import transformers.integrations.sdpa_attention as sdpa_integration
if not globals().get('AVB_SDPA_COMPAT'):
    if not hasattr(sdpa_integration,'use_gqa_in_sdpa'):
        raise RuntimeError('Unrecognized Transformers SDPA implementation')
    from torch.nn.attention import sdpa_kernel,SDPBackend
    q=torch.randn(1,8,128,64,device='cuda:0',dtype=torch.float16)
    k=torch.randn(1,2,128,64,device=q.device,dtype=q.dtype);v=torch.randn_like(k)
    with sdpa_kernel(SDPBackend.MATH):
        reference=torch.nn.functional.scaled_dot_product_attention(q,k,v,enable_gqa=True,is_causal=True)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        candidate=torch.nn.functional.scaled_dot_product_attention(q,k.repeat_interleave(4,dim=1),v.repeat_interleave(4,dim=1),is_causal=True)
    error=float((reference-candidate).abs().max())
    if error>.02:raise RuntimeError(f'Unexpected kernel equivalence error: {error}')
    sdpa_integration.use_gqa_in_sdpa=lambda attention_mask,key:False
    AVB_SDPA_COMPAT=dict(explicit_kv_repeat=True,toy_max_abs_error=error,
                        media_reduced=False,attention_masks_modified=False)
    (OUT/'avbind_sdpa_compat.json').write_text(json.dumps(AVB_SDPA_COMPAT),encoding='utf-8')
    print('T4_SDPA_COMPAT',json.dumps(AVB_SDPA_COMPAT),flush=True)
    del q,k,v,reference,candidate;torch.cuda.empty_cache()
