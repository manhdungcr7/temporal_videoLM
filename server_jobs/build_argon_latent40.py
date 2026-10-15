"""Extract Argon's exact supplied Python as a generated, archived artifact."""
import argparse, ast, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('attachment',type=Path)
    args=parser.parse_args()
    text=args.attachment.read_text(encoding='utf-8')
    start=text.index('SHARD = 0\n')
    end=text.index('RESULT = run_shard()',start)+len('RESULT = run_shard()')
    code=text[start:end]+'\n'
    ast.parse(code)
    folder=ROOT/'results/argon_latent40';folder.mkdir(exist_ok=True)
    (folder/'proposal_original.txt').write_text(text,encoding='utf-8')
    (folder/'proposal_original.py').write_text(code,encoding='utf-8')
    print(json.dumps({'lines':len(code.splitlines()),'sha256':hashlib.sha256(code.encode()).hexdigest()}))
