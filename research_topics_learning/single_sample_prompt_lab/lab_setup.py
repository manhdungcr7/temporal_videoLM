"""Create/repair a venv without ensurepip; use the host pip to manage it."""
import importlib.metadata
import json
import subprocess
import sys
import venv
from pathlib import Path


def prepare_environment(directory):
    directory=Path(directory).resolve()
    version=importlib.metadata.version('pip')
    parts=version.split('.')
    if tuple(int(part) for part in parts[:2]) < (22,3):
        raise RuntimeError('Host pip >=22.3 is required for --python; found '+version)
    # Repeat even when the interpreter exists: an earlier ensurepip failure leaves
    # an incomplete venv. No deletion, pip bootstrap or system package changes.
    venv.EnvBuilder(system_site_packages=True,with_pip=False).create(directory)
    python=directory/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    completed=subprocess.run([str(python),'-c',
        'import json,sys; print(json.dumps({"prefix":sys.prefix,"base":sys.base_prefix}))'],
        check=True,capture_output=True,text=True,encoding='utf-8')
    info=json.loads(completed.stdout)
    if Path(info['prefix']).resolve()!=directory or info['prefix']==info['base']:
        raise RuntimeError('The target interpreter is not the requested virtual environment')
    # Official pip --python also supports targets without their own pip module.
    return python,[sys.executable,'-m','pip','--python',str(python)]
