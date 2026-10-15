"""Low-activity monitor: wait45s between snapshots, exit on completion/error."""
import concurrent.futures,json,time
from argon_manage import STATE,checkpoint

jobs=json.loads(STATE.read_text())
while True:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        status=list(pool.map(checkpoint,jobs))
    print(json.dumps([dict(server=s['server'],status=s['status'],completed=s['completed'],
        seconds=s['seconds'],error=s['error']) for s in status]),flush=True)
    if any(s['status']=='error' for s in status):raise SystemExit(2)
    if all(s['status']=='done' for s in status):break
    time.sleep(45)
