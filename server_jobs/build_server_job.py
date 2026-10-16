"""Ghép một job trong repo/kaggle/jobs thành MỘT file Python để chạy trên server GPU từ xa (remote_client.py run ...).

Server chạy mã trong namespace của kaggle_rootcause (đã có MODEL_, PROC, Lay-không-có, cut, fetch, ...). Vì vậy file ghép phải có,
theo đúng thứ tự:
    1. cấu hình tường minh (time_left, PER_LABEL, SEED, J12_*, J13_* ...)      <- BẮT BUỘC, xem README mục 9
    2. job_utils.py                                                              (sample_items, score, run_loop, ...)
    3. phần helper của j8_acar.py (Lay, forward, profiles, ...), bỏ phần chạy     (cần cho j12 / j13 / p*)
    4. locate() lấy từ remote_tests/debug_parallel.py                            (cần cho j12 / j13)
    5. file job

Cách dùng (chạy từ thư mục gốc của repo):
    python server_jobs/build_server_job.py j12  OUT.py [--shard K/N] [--prompts oneword|is] [--per-label 4] [--seed 2468]
    python server_jobs/build_server_job.py j13  OUT.py [--measure is|full] [--decode]
    python server_jobs/build_server_job.py j14  OUT.py --per-label 8 --tasks TA,TF           # cổng Hop 2 (khớp câu dò tiếng -> câu dò hình)
    python server_jobs/build_server_job.py j15  OUT.py --per-label 4 --seed 777 [--shard K/N] # nguồn gốc hiện tượng dồn đáp án
    python server_jobs/build_server_job.py plain  OUT.py --job-file repo/kaggle/jobs/t2_order_blindness.py   # job không cần helper j8
Rồi:  REMOTE_CFG=remote_config5.json python remote_client.py run OUT.py --name ten --nowait
"""
import argparse, ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
J = ROOT / 'repo' / 'kaggle' / 'jobs'
rd = lambda p: Path(p).read_text(encoding='utf-8')

PROMPTS = {
    'is': ("\\nThe essential visual content of chunk {t} in one short phrase is",
           "\\nThe essential audio content of chunk {t} in one short phrase is", "\\n"),
    'oneword': (" The essential visual content of chunk {t} in one word:",
                " The essential audio content of chunk {t} in one word is:", " "),
    'timestamp': (" Visual information from {t0}s to {t1}s, summarized in one short phrase: ",
                  " Audio information from {t0}s to {t1}s, summarized in one short phrase: ", " "),
}


def j8_helpers():
    src = rd(J / 'j8_acar.py').split('\nitems = sample_items')[0]            # chỉ giữ phần định nghĩa
    return '\n'.join(l for l in src.split('\n') if not l.startswith('exec((ROOT'))


def locate_fn():
    src = rd(ROOT / 'remote_tests' / 'debug_parallel.py')
    i = src.index('def locate')
    return src[i:src.index('\ndef parallel_verify')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('job'); ap.add_argument('out')
    ap.add_argument('--job-file'); ap.add_argument('--per-label', type=int, default=4); ap.add_argument('--seed', type=int, default=2468)
    ap.add_argument('--max-dur', type=float, default=42.0)
    ap.add_argument('--shard', help='K/N: server này lấy item K, K+N, ... (chia việc cho nhiều server)')
    ap.add_argument('--prompts', choices=list(PROMPTS), default='timestamp')
    ap.add_argument('--tasks', default='TA,TF,CMTB'); ap.add_argument('--out-name', default=None)
    ap.add_argument('--measure', default='is'); ap.add_argument('--decode', action='store_true')
    a = ap.parse_args()

    cfg = f"time_left = lambda: 10 ** 9\nPER_LABEL = {a.per_label}; SEED = {a.seed}; J8_MAX_DUR = {a.max_dur}\n"
    if a.job == 'j13' and a.prompts == 'timestamp':          # j13 formats its prompts with {t}; the time-range prompts need j12
        a.prompts = 'oneword'
    if a.job in ('j12', 'j13'):
        pv, pa, sep = PROMPTS[a.prompts]
        cfg += f"J12_PV = '{pv}'\nJ12_PA = '{pa}'\nJ12_SEP = '{sep}'\nJ12_BOOST_LAYERS = {tuple(range(13, 28))!r}\nJ12_TASKS = {tuple(a.tasks.split(','))!r}\n"
        cfg += f"J12_OUT = '{a.out_name or 'j12_out.jsonl'}'\n"
        cfg += "J12_CONDS = (('none', 'cq', 'shared'), ('boost_p', 'cq_boost_p', 'shared'), ('boost_pr', 'cq_boost_pr', 'shared'))\n"
    body = cfg + rd(J / 'job_utils.py') + '\n'
    if a.job == 'plain':
        body += rd(a.job_file)
    else:
        body += j8_helpers() + f"\nPER_LABEL = {a.per_label}; SEED = {a.seed}\nTOK = PROC.tokenizer\n" + locate_fn() + '\n'
        j12 = rd(J / 'j12_chunk_prompts.py')
        if a.job == 'j12':
            if a.shard: k, n = a.shard.split('/'); body += f"J12_SHARD = ({k}, {n})\n"
            body += j12
        elif a.job in ('j14', 'j15', 'j16'):                                            # probe jobs that reuse j12's definitions
            if a.shard: k, n = a.shard.split('/'); body += f"J12_SHARD = ({k}, {n})\n"
            name = {'j14': 'j14_hop2_gate.py', 'j15': 'j15_bias_source.py', 'j16': 'j16_token_relevance.py'}[a.job]
            body += j12.split('\nTASKS = ')[0] + '\n'
            if a.out_name: body += f"{a.job.upper()}_OUT = '{a.out_name}'\n"
            body += rd(J / name)
        elif a.job in ('j17', 'j18', 'j19', 'j25'):                                            # jobs that need only the J8 helpers
            if a.shard: k, n = a.shard.split('/'); body += f"J12_SHARD = ({k}, {n})\n"
            name = {'j17': 'j17_question_first.py', 'j18': 'j18_tmrope_time.py', 'j19': 'j19_layer_lens.py', 'j25': 'j25_pretrain_checks.py'}[a.job]
            if a.out_name: body += f"{a.job.upper()}_OUT = '{a.out_name}'\n"
            body += rd(J / name)
        elif a.job == 'j13':
            body += j12.split('\nTASKS = ')[0]                                   # chỉ định nghĩa (cq_attention, cq_forward, ...), không chạy
            body += (f"\nPER_LABEL = 8; J13_OUT = '{a.out_name or 'j13_out.jsonl'}'; J13_MEASURE = '{a.measure}'; J13_DECODE = {a.decode}\n"
                     + rd(J / 'j13_relevance_heads.py'))
    ast.parse(body)
    Path(a.out).write_text(body, encoding='utf-8')
    print('wrote', a.out, len(body), 'bytes')


main()
