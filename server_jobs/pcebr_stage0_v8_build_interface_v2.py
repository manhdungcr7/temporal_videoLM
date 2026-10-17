"""Finish general lexical/clause interface corrections; retain all CLRV constants."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
p=R/'server_jobs/pcebr_stage0_v8_token_interface.py'
s=p.read_text(encoding='utf-8')
old='    return source\n'
extra='''    # General modality adjectives describe the channel, not an event entity.
    marker = 'GENERIC_EVASION_PATTERNS = ('
    assert source.count(marker) == 1
    source=source.replace(marker, "DEICTIC_META_WORDS.update({'visual','auditory','acoustic','audiovisual'})\\nVISUAL_MODALITY_CUES.update({'visual'})\\nAUDIO_MODALITY_CUES.update({'auditory','acoustic'})\\n\\n"+marker)
    # A leading temporal/relative subordinate clause owns its internal 'when'.
    # Prefer its comma/semicolon boundary over an inner connector.
    marker="    connector_found = split_pos is not None\\n    for idx_r, (r, t_str, words) in ([] if split_pos is not None else enumerate(row_words)):"
    replacement="""    leading_word = next((ws[0] for _,_,ws in row_words if ws), '')
    if leading_word in ('relative','when','while','before','after','during'):
        boundary = next((i+1 for i,(_,text_piece,_) in enumerate(row_words[:-2]) if i>=2 and (',' in text_piece or ';' in text_piece)), None)
        if boundary is not None:
            split_pos = boundary
    connector_found = split_pos is not None
    for idx_r, (r, t_str, words) in ([] if split_pos is not None else enumerate(row_words)):"""
    assert source.count(marker)==1
    source=source.replace(marker,replacement)
    return source
'''
assert s.count(old)==1
(R/'server_jobs/pcebr_stage0_v8_token_interface_v2.py').write_text(s.replace(old,extra),encoding='utf-8',newline='\n')
print('created parser v2; constants and weights unchanged')
