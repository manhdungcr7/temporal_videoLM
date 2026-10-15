"""Compatibility repair only: do not touch inference prompts or weights."""
import subprocess
def ffmpeg(*args):
    result=subprocess.run([FF,'-hide_banner','-y',*args],capture_output=True)
    if result.returncode and b'fps_mode' in result.stderr:
        args=['-vsync' if arg=='-fps_mode' else arg for arg in args]
        result=subprocess.run([FF,'-hide_banner','-y',*args],capture_output=True)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[-600:])
