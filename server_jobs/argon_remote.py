"""Authenticated GPU transport; credentials stay in ignored config files."""
import argparse
import base64
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def call(server, path, body=None, raw=False, timeout=60):
    file = ROOT / ('remote_config.json' if server == 1 else 'remote_config2.json')
    config = json.loads(file.read_text(encoding='utf-8'))
    request = urllib.request.Request(config['url'].rstrip('/') + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'X-Token': config['token'], 'ngrok-skip-browser-warning': '1',
                 'Content-Type': 'application/json', 'User-Agent': 'argon-latent40/1'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    return data if raw else json.loads(data)

def upload(server, path, name=None):
    path = Path(path)
    return call(server, '/upload', {'path': name or path.name,
        'b64': base64.b64encode(path.read_bytes()).decode()})

def submit(server, code, name):
    return call(server, '/submit', {'code': code, 'name': name})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['info', 'submit', 'status', 'upload', 'get'])
    parser.add_argument('server', type=int, choices=[1, 2])
    parser.add_argument('target', nargs='?')
    parser.add_argument('--name', default='argon_latent40')
    parser.add_argument('--local')
    args = parser.parse_args()
    if args.action == 'info':
        result = call(args.server, '/info')
    elif args.action == 'upload':
        result = upload(args.server, args.target, args.local)
    elif args.action == 'submit':
        code = 'ARGON_SERVER = %d\n' % args.server + Path(args.target).read_text(encoding='utf-8')
        result = submit(args.server, code, args.name)
    elif args.action == 'status':
        result = call(args.server, '/job/' + args.target)
        result['out'] = result.get('out', '')[-10000:]
    else:
        from urllib.parse import quote
        data = call(args.server, '/file?path=' + quote(args.target), raw=True, timeout=180)
        Path(args.local).write_bytes(data)
        result = {'saved': args.local, 'bytes': len(data)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
