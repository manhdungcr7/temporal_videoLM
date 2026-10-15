import inspect
from transformers.integrations.sdpa_attention import sdpa_attention_forward
RESULT=dict(sdpa_source=inspect.getsource(sdpa_attention_forward),
            prepare_source=inspect.getsource(prepare_video),
            config_implementation=MODEL_.config._attn_implementation,
            enabled=dict(flash=torch.backends.cuda.flash_sdp_enabled(),efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),math=torch.backends.cuda.math_sdp_enabled()))
print('ATTENTION_RUNTIME_AUDIT',json.dumps(RESULT),flush=True)
