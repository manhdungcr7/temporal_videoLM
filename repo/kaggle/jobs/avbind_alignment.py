"""Native source-to-text semantic calibration, not a standalone novelty claim.

Clock metadata is never an input to this projection. Source-conditioned small
residual maps align isolated native features to frozen observation-text features.
It is trained on question-blind physical AV/text pairs, not benchmark QA labels.
Repeated identical pseudo descriptions are multiple positives, not false
negative distinct events; occurrence intervals remain separate downstream.
"""
import torch
from torch import nn
import torch.nn.functional as F


class NativeSemanticAlignment(nn.Module):
    def __init__(self,dim=2048,rank=64,num_sources=2):
        super().__init__()
        self.dim=dim;self.rank=rank;self.num_sources=num_sources
        self.down=nn.Linear(dim,rank,bias=False)
        self.source=nn.Embedding(num_sources,rank)
        self.up=nn.Parameter(torch.zeros(num_sources,rank,dim))
        self.bias=nn.Parameter(torch.zeros(num_sources,dim))

    def forward(self,content,sources):
        if content.shape[:-1]!=sources.shape or content.shape[-1]!=self.dim:
            raise ValueError('Source/content shape mismatch')
        if sources.dtype!=torch.long or ((sources<0)|(sources>=self.num_sources)).any():
            raise ValueError('Invalid physical source identity')
        latent=F.gelu(self.down(content)+self.source(sources))
        delta=torch.einsum('...r,...rd->...d',latent,self.up[sources])
        return F.layer_norm(content+delta+self.bias[sources],(self.dim,))


def semantic_pair_loss(projected,text,positive_mask,temperature=.07):
    if projected.shape!=text.shape or positive_mask.shape!=(len(text),len(text)):
        raise ValueError('Pair matrix shape mismatch')
    if not positive_mask.any(-1).all() or not positive_mask.any(0).all():
        raise ValueError('Every row/column requires an observed positive')
    logits=F.normalize(projected,dim=-1)@F.normalize(text,dim=-1).T/temperature
    negative=float('-inf')
    forward=torch.logsumexp(logits,-1)-torch.logsumexp(logits.masked_fill(~positive_mask,negative),-1)
    reverse=torch.logsumexp(logits,0)-torch.logsumexp(logits.masked_fill(~positive_mask,negative),0)
    reconstruction=(1-F.cosine_similarity(projected,text,dim=-1)).mean()
    return .5*(forward.mean()+reverse.mean())+reconstruction


def occurrence_token_bias(query_features,projected_records,record_spans,total_tokens,
                          temperature=.07,gain=2):
    """Each arbitrary query token distributes evidence to all occurrence records.

    Entropy gates undiscriminating token affinities; no stopword list, question
    classifier or enforced source budget. Record token counts calibrate density,
    so a visual patch count does not alone outweigh an audio observation.
    """
    if len(projected_records)!=len(record_spans) or len(projected_records)<2:
        raise ValueError('At least two aligned source records are required')
    logits=F.normalize(query_features.float(),dim=-1)@F.normalize(projected_records.float(),dim=-1).T/temperature
    weights=logits.softmax(-1)
    entropy=-(weights*weights.clamp_min(1e-12).log()).sum(-1)
    gate=(1-entropy/torch.log(torch.tensor(float(len(record_spans)),device=weights.device))).clamp(0,1)
    bias=torch.zeros(len(query_features),total_tokens,device=weights.device)
    covered=sum(end-start for start,end in record_spans)
    for index,(start,end) in enumerate(record_spans):
        if not 0<=start<end<=total_tokens:raise ValueError('Invalid occurrence extent')
        extra=gain*gate*weights[:,index]*covered/(end-start)
        bias[:,start:end]=extra.log1p()[:,None]
    return bias,dict(gate=gate,record_weights=weights)
