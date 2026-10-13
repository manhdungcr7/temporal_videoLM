"""Standard-library client; the notebook kernel never imports/reloads the model."""
import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from uuid import uuid4

PREFIX='__SVHLAB_JSON__'


class PromptLab:
    def __init__(self, python, root, output, mock=False):
        self.root=Path(root);self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.log_path=self.output/'worker.log'
        self.log=self.log_path.open('a',encoding='utf-8')
        self.responses=queue.Queue()
        command=[str(python),'-u',str(self.root/'lab_worker.py'),'--output',str(self.output)]
        if mock:command.append('--mock')
        self.process=subprocess.Popen(command,cwd=str(self.root),stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.log,text=True,encoding='utf-8',errors='replace',bufsize=1)
        self.reader=threading.Thread(target=self._read,daemon=True);self.reader.start()
        self.ready=self._wait(None,3600)
        if self.ready.get('event')!='ready':raise RuntimeError(self.ready)

    def _read(self):
        for line in self.process.stdout:
            if line.startswith(PREFIX):
                try:self.responses.put(json.loads(line[len(PREFIX):]))
                except Exception:self.responses.put({'event':'fatal','error':'Malformed worker response'})

    def _wait(self, request_id, timeout):
        started=time.monotonic();last_print=started
        while time.monotonic()-started<timeout:
            try:response=self.responses.get(timeout=1)
            except queue.Empty:
                if self.process.poll() is not None:raise RuntimeError('Worker stopped; see '+str(self.log_path))
                if time.monotonic()-last_print>=30:
                    print('Đang tải/chạy model; log:',self.log_path,flush=True);last_print=time.monotonic()
                continue
            if response.get('event')=='fatal':raise RuntimeError(response.get('error'))
            if request_id is None or response.get('request_id')==request_id:return response
            # Late reply after a notebook interrupt is not reused for a newer prompt.
        raise TimeoutError('Worker timed out; see '+str(self.log_path))

    def is_alive(self):return self.process.poll() is None

    def request(self, payload, timeout=1800):
        if not self.is_alive():raise RuntimeError('Run the model-loading cell again')
        request_id=uuid4().hex
        self.process.stdin.write(json.dumps({**payload,'request_id':request_id},ensure_ascii=False)+'\n')
        self.process.stdin.flush()
        response=self._wait(request_id,timeout)
        if not response.get('ok'):raise RuntimeError(response.get('error'))
        return response['result']

    def close(self):
        if self.is_alive():self.process.terminate();self.process.wait(timeout=30)
        self.log.close()
