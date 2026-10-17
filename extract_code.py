import hashlib

with open('temp_step769.txt', 'r', encoding='utf-8') as f:
    text = f.read()

m = text.find('Augmented-Dustbin Prescribed-Marginal')
doc_start = text.rfind('"""', 0, m)
print('doc_start:', doc_start)
code = text[doc_start:]

# Remove any trailing markdown closing tags if present
if code.endswith('</USER_REQUEST>'):
    code = code[:-len('</USER_REQUEST>')].rstrip()

print('Length of extracted code:', len(code))
print('First 200 chars:\n', code[:200])
print('Last 300 chars:\n', code[-300:])

# Check SHA256
sha_full = hashlib.sha256(code.encode('utf-8')).hexdigest()
sha_no_lf = hashlib.sha256(code.rstrip('\r\n').encode('utf-8')).hexdigest()
print('SHA256 full:', sha_full)
print('SHA256 no LF:', sha_no_lf)

with open('stage0_pcebr_v7_joint_sinkhorn_clc.py', 'w', encoding='utf-8') as out:
    out.write(code)

print('Wrote to stage0_pcebr_v7_joint_sinkhorn_clc.py successfully!')

