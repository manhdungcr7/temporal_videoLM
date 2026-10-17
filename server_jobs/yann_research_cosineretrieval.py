"""R7: bounded cosine retrieval; matched media-conditioned vs question-only semantic query."""
source=(OUT/'uploads/yann_research_distillretrieval.py').read_text()
source=source.replace("'distillretrieval", "'cosineretrieval").replace('YR_RETRIEVAL','YR_COSINE')
source=source.replace('if YR_SERVER==1:', 'if True:')
source=source.replace("config.update(name='reader_distilled_retrieval' if YR_SERVER==2 else 'grounding_only_retrieval'", "config.update(name='question_only_cosine_retrieval' if YR_SERVER==2 else 'media_conditioned_cosine_retrieval'")
needle='history=[];best=math.inf;beststep=None'
replacement='''
question_states=torch.load(YR_DIR/'textquery/question_states.pt',map_location='cpu',weights_only=True)
if YR_SERVER==2:
    for gs in (train_groups,val_groups,diag_groups):
        for g in gs:
            for c in g.values():
                c['original_media_conditioned_q']=c['q'];c['q']=question_states[c['question']]['mean']
def attention(c,context):
    b=F.layer_norm(c['bank'].to(dev).float(),(2048,));q=F.layer_norm(context['q'].to(dev).float(),(2048,))
    with torch.autocast('cuda',dtype=torch.float16):
        queries=probe.query(q).reshape(4,64);keys=probe.key(b)+probe.mod_key(c['mods'].to(dev))
        scores=8.*(F.normalize(queries.float(),dim=-1)@F.normalize(keys.float(),dim=-1).T)
        return F.softmax(scores,dim=-1)
config.update(query='question-only text mean at block18' if YR_SERVER==2 else 'media-conditioned final-prompt state at block18',
    objective='endpoint CE only in both conditions',scorer='cosine *8, fixed scale',
    selection='minimum recording-averaged validation endpoint CE; includes step0')
history=[];best=math.inf;beststep=None
'''
assert needle in source;source=source.replace(needle,replacement)
exec(compile(source,'r7_matched_cosine_query','exec'),globals())
