"""draft_job, job_templates and check_job(script_text): drafted scripts follow the rules, parse, and run end to end in
a dry run with stub srun/scontrol/apptainer and a fake vLLM server (tests/draftbin). Nothing here talks to the cluster."""
import os
import random
import shutil
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path

from amadeus import drafts, templates

HERE = Path(__file__).resolve().parent
BIN = HERE / 'draftbin'

SHAPES = {
    'cpu': dict(kind='cpu', name='t-cpu', cores=8, time='5'),
    'cpu-array': dict(kind='cpu-node', name='t-cpun', time='5', array='0-3%2'),
    'cpu-nodes': dict(kind='cpu-node', name='t-cpun2', nodes=2, time='5', container='$DATA/containers/x.sif'),
    'gpu': dict(kind='gpu', name='t-gpu', gpus=2, time='5', python_env='$DATA/envs/train', modules=['CUDA/12.6.0']),
    'gpu-node': dict(kind='gpu-node', name='t-gn', time='5'),
    'gpu-nodes': dict(kind='gpu-node', name='t-gn2', nodes=2, time='5'),
    'vllm': dict(kind='gpu-node', name='t-vllm', time='10', vllm_model='$DATA/models/Qwen3.5-4B',
                 vllm_servers_per_node=4, vllm_args="--max-model-len 4096 --limit-mm-per-prompt '{\"image\": 0}'"),
    'production': dict(kind='gpu-node', name='t-prod', nodes=8, time='02:00:00', dev=False),
    'runs': dict(kind='gpu-node', name='t-runs', time='10', command='echo setup',
                 runs=['python3 train.py --seed 0', 'python3 train.py --seed 1\necho "it\'s done"']),
    'runs-8': dict(kind='gpu-node', name='t-runs8', time='10', python_env='$DATA/envs/train',
                   runs=[f'python eval.py --shard {i}' for i in range(8)], run_minutes=[5] * 8),
    'array-args': dict(kind='cpu', name='t-arr', cores=4, time='5',
                       array_args=['--data $DATA/a "x y"', "--data b --tag it's", '--x `c` \\d'],
                       command='python3 run.py $ARGS --out "$OUT"'),
    'array-args-nodes': dict(kind='gpu-node', name='t-arr2', nodes=2, time='60', dev=False,
                             array_args=['--lr 1e-4', '--lr 3e-4', '--lr 1e-3']),
    'preset-qwen': dict(kind='gpu-node', name='t-q38', time='10', vllm_preset='qwen38-fp8'),
    'preset-glm': dict(kind='gpu-node', name='t-glm', nodes=2, time='60', dev=False, vllm_preset='glm53-fp8',
                       vllm_args='--max-num-seqs 32'),
}


def free_port_base(n: int) -> int:
    for _ in range(50):
        base = random.randint(20000, 40000)
        try:
            socks = []
            for p in range(base, base + n):
                s = socket.socket()
                s.bind(('127.0.0.1', p))
                socks.append(s)
            for s in socks:
                s.close()
            return base
        except OSError:
            for s in socks:
                s.close()
    raise RuntimeError('no free ports')


class Shapes(unittest.TestCase):
    def test_every_shape_passes_the_rules_and_parses(self):
        for key, kw in SHAPES.items():
            with self.subTest(key):
                d = drafts.draft(**kw)
                self.assertEqual(d['check']['errors'], [], d['script'])
                self.assertEqual(d['check']['warnings'], [])
                r = subprocess.run(['bash', '-n'], input=d['script'], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn('#SBATCH --account=p123456', d['script'])
                self.assertEqual(d['script'].count('\n#ASC --vanilla\n'), 0 if kw['kind'] == 'gpu' else 1)
                self.assertTrue(d['script'].startswith('#!/bin/bash\n#ASC --vanilla\n') or kw['kind'] == 'gpu')

    def test_resources_in_the_tested_forms(self):
        s = drafts.draft(**SHAPES['cpu'])['script']
        for o in ('--nodes=1', '--ntasks=1', '--cpus-per-task=8', '--threads-per-core=1', '--mem=32G',
                  '--qos=dev_zen4_0768'):
            self.assertIn(f'#SBATCH {o}\n', s)
        s = drafts.draft(**SHAPES['gpu-nodes'])['script']
        for o in ('--nodes=2', '--ntasks-per-node=1', '--cpus-per-task=352', '--gres=gpu:4', '--exclusive', '--mem=0',
                  '--exclude=n3010-019,n3010-027,n3012-034'):
            self.assertIn(f'#SBATCH {o}\n', s)
        s = drafts.draft(**SHAPES['gpu'])['script']
        self.assertIn('#SBATCH --gres=gpu:2\n', s)
        lmod = s.index('init/lmod/sh')
        self.assertLess(lmod, s.index('set -euo pipefail'))      # Lmod before set -u (AGENTS.md §8)
        self.assertIn('module load CUDA/12.6.0', s)
        s = drafts.draft(**SHAPES['cpu-array'])['script']
        self.assertIn('#SBATCH --output=logs/%x-%A_%a.out\n', s)

    def test_production_needs_approval(self):
        d = drafts.draft(**SHAPES['production'])
        self.assertIn('#SBATCH --qos=zen4_0768_h100x4\n', d['script'])
        self.assertTrue(any('8 nodes' in a for a in d['check']['needs_approval']))
        self.assertEqual(d['check']['estimate']['max_tokens'], 64.0)

    def test_refusals(self):
        for kw, words in [(dict(kind='gpu', name='x', gpus=4), 'gpu-node'),
                          (dict(kind='cpu', name='x', time='20'), 'dev QoS'),
                          (dict(kind='gpu-node', name='x', nodes=3), 'at most 2 nodes'),
                          (dict(kind='cpu', name='x', nodes=2), 'cpu-node'),
                          (dict(kind='foo', name='x'), 'cpu-node'),
                          (dict(kind='cpu', name='bad name'), 'name'),
                          (dict(kind='cpu', name='x', time='80:00:00', dev=False), '72 h'),
                          (dict(kind='cpu', name='x', array='1-3; rm'), 'array'),
                          (dict(kind='gpu', name='x', vllm_model='$DATA/m'), 'gpu-node'),
                          (dict(kind='gpu-node', name='x', vllm_model='$DATA/m', vllm_servers_per_node=3), 'TP'),
                          (dict(kind='cpu', name='x', container='a b.sif'), 'plain path'),
                          (dict(kind='cpu', name='x', container='$DATA/a.sif', python_env='$DATA/env'), 'pick one'),
                          (dict(kind='cpu', name='x', modules=['CUDA; rm']), 'module')]:
            with self.subTest(kw):
                with self.assertRaises(drafts.DraftError) as cm:
                    drafts.draft(**kw)
                self.assertIn(words, str(cm.exception))

    def test_extras_refused(self):
        gn = dict(kind='gpu-node', name='x', time='10')
        vl = dict(gn, vllm_model='$DATA/m')
        for kw, words in [(dict(kind='cpu', name='x', runs=['a']), 'gpu-node'),
                          (dict(gn, nodes=2, runs=['a']), 'nodes=1'),
                          (dict(vl, runs=['a']), 'no vllm_model'),
                          (dict(gn, runs=['a'] * 9), 'at most 8'),
                          (dict(gn, runs=['a', ' ']), 'needs a command'),
                          (dict(gn, runs=['a', 'b'], run_minutes=[3]), 'one estimate per run'),
                          (dict(gn, runs=['a'], run_minutes=[30]), 'slowest run'),
                          (dict(gn, array='0-3', array_args=['a']), 'array_args sets --array'),
                          (dict(gn, array_throttle=2), 'goes with array_args'),
                          (dict(gn, array_args=['a\nb']), 'one line'),
                          (dict(vl, vllm_preset='llama'), 'qwen38-fp8'),
                          (dict(gn, vllm_preset='glm53-fp8', vllm_servers_per_node=2), 'one TP4 server'),
                          (dict(vl, load_cap_s=10), 'load_cap_s')]:
            with self.subTest(kw):
                with self.assertRaises(drafts.DraftError) as cm:
                    drafts.draft(**kw)
                self.assertIn(words, str(cm.exception))

    def test_runtime_defaults(self):
        for key, kw in SHAPES.items():
            s = drafts.draft(**kw)['script']
            with self.subTest(key):
                self.assertNotIn('$(nproc)', s)                         # nproc honours OMP_NUM_THREADS
                self.assertIn('${SLURM_CPUS_ON_NODE:-?} CPUs', s)
                self.assertEqual('export HF_HUB_OFFLINE=1 ' in s, kw['kind'].startswith('gpu'))
                multi_gpu = kw['kind'] == 'gpu-node' and kw.get('nodes', 1) > 1 and 'vllm_model' not in kw \
                    and 'vllm_preset' not in kw
                self.assertEqual("export NCCL_IB_HCA='^=mlx5_0'" in s, multi_gpu)
        s = drafts.draft(**SHAPES['vllm'])['script']
        self.assertIn('OMP_NUM_THREADS=1\n', s[s.index('serve_node()'):s.index('launch()')])  # each server
        self.assertIn('LOAD_CAP=${LOAD_CAP:-420}', s)
        self.assertIn('LOAD_CAP=${LOAD_CAP:-600}', drafts.draft(**SHAPES['vllm'], load_cap_s=600)['script'])
        self.assertIn('`free` shows the whole node', ' '.join(drafts.draft(**SHAPES['cpu'])['notes']))

    def test_preflight_lines(self):
        s = drafts.draft(**SHAPES['gpu'])['script']
        self.assertIn('need "$VIRTUAL_ENV/bin/python"\n', s)
        self.assertIn('need "$SIF"\n', drafts.draft(**SHAPES['cpu-nodes'])['script'])
        self.assertIn('need "$VLLM_SIF" "$MODEL"\n', drafts.draft(**SHAPES['vllm'])['script'])
        s = drafts.draft(kind='gpu-node', name='x', time='10', vllm_model='Qwen/Qwen3.5-4B')['script']
        self.assertIn('need "$VLLM_SIF"\n', s)                         # a repo id is not a path
        self.assertNotIn('need', drafts.draft(**SHAPES['cpu'])['script'])

    def test_array_args(self):
        d = drafts.draft(**SHAPES['array-args'])
        s = d['script']
        self.assertIn('#SBATCH --array=0-2%2\n', s)                     # dev QoS: 2 nodes at once
        self.assertIn('    "--data $DATA/a \\"x y\\""\n', s)
        self.assertIn('    "--x \\`c\\` \\\\d"\n', s)
        self.assertIn('OUT=results/t-arr-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-a$TASK', s)
        d = drafts.draft(**SHAPES['array-args-nodes'])
        self.assertIn('#SBATCH --array=0-2%1\n', d['script'])            # 2 nodes x 1 task < 4 nodes
        self.assertEqual(d['check']['needs_approval'], ['up to 24 tokens at the time limit (> 20)'])
        s = drafts.draft(kind='cpu-node', name='x', time='60', dev=False, array_args=['a'] * 5)['script']
        self.assertIn('#SBATCH --array=0-4%3\n', s)
        s = drafts.draft(kind='cpu-node', name='x', time='60', dev=False, array_args=['a'] * 5, array_throttle=5)['script']
        self.assertIn('#SBATCH --array=0-4\n', s)

    def test_presets(self):
        d = drafts.draft(**SHAPES['preset-qwen'])
        s = d['script']
        self.assertIn('MODEL=$DATA/models/Qwen3.8-Flash-Next-FP8', s)
        self.assertIn('VLLM_SIF=$DATA/containers/vllm-openai_nightly-20261008.sif', s)
        self.assertIn('--moe-backend triton', s)
        self.assertIn('xhigh', ' '.join(d['notes']))
        self.assertIn('Dev QoS: a vLLM start', ' '.join(d['notes']))
        d = drafts.draft(**SHAPES['preset-glm'])
        self.assertIn("--kv-cache-dtype bfloat16 --gpu-memory-utilization 0.95 --max-num-seqs 64 --max-model-len 65536",
                      d['script'])
        self.assertIn('--reasoning-parser glm47 --max-num-seqs 32)', d['script'])     # caller args last: they win
        self.assertIn('65,536', ' '.join(d['notes']))
        self.assertNotIn('Dev QoS', ' '.join(d['notes']))

    def test_run_minutes_spread_is_noted(self):
        notes = ' '.join(drafts.draft(kind='gpu-node', name='x', time='10', runs=['a', 'b'], run_minutes=[3, 8])['notes'])
        self.assertIn('idle', notes)
        notes = ' '.join(drafts.draft(kind='gpu-node', name='x', time='10', runs=['a', 'b'], run_minutes=[7, 8])['notes'])
        self.assertNotIn('idle', notes)

    def test_minutes_become_hh_mm(self):
        self.assertIn('#SBATCH --time=00:05:00\n', drafts.draft(kind='cpu', name='x', time='5')['script'])
        self.assertIn('#SBATCH --time=26:00:00\n',
                      drafts.draft(kind='cpu', name='x', time='1-02:00:00', dev=False)['script'])


class DryRun(unittest.TestCase):
    """The drafted scripts run end to end against stubs: srun runs the tasks locally, scontrol lists fake nodes."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / 'work'
        self.work.mkdir()
        (self.work / 'REVISION').write_text('abc1234\n')
        model = self.tmp / 'data' / 'models' / 'tiny'
        model.mkdir(parents=True)
        (model / 'config.json').write_text('{}')
        (self.tmp / 'data' / 'containers').mkdir()
        for sif in ('containers/vllm-openai_v0.30.0.sif', 'c.sif'):     # the images the drafts check for
            (self.tmp / 'data' / sif).write_text('sif')
        (self.tmp / 'data' / 'blobs').mkdir()
        (self.tmp / 'data' / 'blobs' / 'abc').write_text('weights')
        (model / 'model.safetensors').symlink_to(self.tmp / 'data' / 'blobs' / 'abc')   # as in a HF cache snapshot
        for sub in ('.cache/huggingface', 'tok'):                 # nested directories, copied in parallel
            (model / sub).mkdir(parents=True)
            for n in range(20):
                (model / sub / f'f{n}').write_text(f'{sub}{n}')
        (self.tmp / 'scratch').mkdir()
        self.port = None

    def tearDown(self):
        subprocess.run(['pkill', '-f', f'hang-marker-{self.tmp.name}'], capture_output=True)
        if self.port:
            subprocess.run(['pkill', '-f', f'fake_vllm.py {self.port}'], capture_output=True)
            for k in range(1, 4):
                subprocess.run(['pkill', '-f', f'fake_vllm.py {self.port + k}'], capture_output=True)
        shutil.rmtree(self.tmp)

    def run_job(self, script: str, nodes: int = 1, extra_env: dict | None = None, timeout: int = 60, args=()):
        path = self.tmp / 'job.slurm'
        path.write_text(script)
        env = dict(os.environ, PATH=f'{BIN}:{os.environ["PATH"]}', DRY=str(self.tmp), DATA=str(self.tmp / 'data'),
                   SCRATCH=str(self.tmp / 'scratch'), USER='tester', SLURM_JOB_ID='4242', SLURM_JOB_NAME='dry',
                   SLURM_JOB_NODELIST='fake[1-2]', SLURM_JOB_NUM_NODES=str(nodes), SLURM_CPUS_PER_TASK='352',
                   FAKE_NODES=' '.join(['localhost'] * nodes), STAGE_ROOT=str(self.tmp / 'local'))
        env.update(extra_env or {})
        return subprocess.run(['bash', str(path), *args], cwd=self.work, env=env, capture_output=True, text=True,
                              timeout=timeout)

    def test_single_node_job(self):
        s = drafts.draft(kind='cpu', name='dry', cores=4, time='5', command='echo "work $SLURM_JOB_ID" > out.txt')['script']
        r = self.run_job(s)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual((self.work / 'out.txt').read_text(), 'work 4242\n')
        self.assertIn('revision abc1234', r.stdout)
        self.assertIn(' done', r.stdout)

    def test_multi_node_job_retries_a_failed_launch(self):
        cmd = 'echo "$NODE_RANK of $NNODES, head $HEAD, left $(seconds_left)" > node$NODE_RANK.txt\ncat <<EOF > here$NODE_RANK.txt\nrank $NODE_RANK\nEOF'
        s = drafts.draft(kind='gpu-node', name='dry', nodes=2, time='5', command=cmd)['script']
        r = self.run_job(s, nodes=2, extra_env={'FAKE_SRUN_FAIL_FIRST': '1'})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('launch attempt 1; retrying', r.stdout)
        for i in (0, 1):
            text = (self.work / f'node{i}.txt').read_text()
            self.assertTrue(text.startswith(f'{i} of 2, head localhost, left '), text)
            self.assertGreater(int(text.split('left ')[1]), 500)        # the job's end from scontrol
            self.assertEqual((self.work / f'here{i}.txt').read_text(), f'rank {i}\n')
        self.assertEqual((self.tmp / 'srun.log').read_text().count('srun n=2'), 2)

    def test_multi_node_failure_is_the_job_exit_code(self):
        s = drafts.draft(kind='cpu-node', name='dry', nodes=2, time='5', command='exit 3')['script']
        r = self.run_job(s, nodes=2)
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn('launch attempt 2; retrying', r.stdout)      # it failed fast, so it was tried three times
        self.assertNotIn('launch attempt 3; retrying', r.stdout)   # no wait after the last attempt
        self.assertIn('done (exit 3)', r.stdout)

    def test_multi_node_failure_is_not_hidden_by_later_lines(self):
        s = drafts.draft(kind='gpu-node', name='dry', nodes=2, time='5',
                         command='python3 -c "import sys; sys.exit(2)"\necho "rank $NODE_RANK finished"')['script']
        r = self.run_job(s, nodes=2)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertNotIn('finished', r.stdout)

    def test_script_arguments_reach_every_node(self):
        s = drafts.draft(kind='cpu-node', name='dry', nodes=2, time='5',
                         command='echo "config=$1 n=$#" > a$NODE_RANK.txt')['script']
        r = self.run_job(s, nodes=2, args=['cfg yaml'])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for i in (0, 1):
            self.assertEqual((self.work / f'a{i}.txt').read_text(), 'config=cfg yaml n=1\n')

    def test_container_and_array_reach_every_node(self):
        s = drafts.draft(kind='cpu-node', name='dry', nodes=2, time='5', array='0-3', container='$DATA/c.sif',
                         command='run echo "task $TASK" > t$NODE_RANK.txt')['script']
        r = self.run_job(s, nodes=2, extra_env={'SLURM_ARRAY_TASK_ID': '3'})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for i in (0, 1):
            self.assertEqual((self.work / f't{i}.txt').read_text(), 'task 3\n')
        self.assertIn('c.sif echo task 3', (self.tmp / 'apptainer.log').read_text())

    def vllm_job(self, spn: int, extra_env: dict | None = None, command: str = ''):
        self.port = free_port_base(4)
        s = drafts.draft(kind='gpu-node', name='dry', time='10', vllm_model='$DATA/models/tiny', command=command,
                         vllm_servers_per_node=spn, vllm_args="--max-model-len 4096 --x '{\"a\": 1}'")['script']
        return self.run_job(s, extra_env={'SERVE_PORT': str(self.port), **(extra_env or {})})

    def test_vllm_servers_and_client(self):
        r = self.vllm_job(2)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('2 of 2 servers up', r.stdout)
        self.assertIn(f'hi from {self.port}', r.stdout)                  # the demo client reached both servers
        self.assertIn(f'hi from {self.port + 1}', r.stdout)
        logs = self.work / 'logs' / 'dry-4242'
        self.assertIn('CUDA_VISIBLE_DEVICES=0,1', (logs / 'server0-0.log').read_text())
        self.assertIn('CUDA_VISIBLE_DEVICES=2,3', (logs / 'server0-1.log').read_text())
        calls = (self.tmp / 'apptainer.log').read_text()
        self.assertIn('--tensor-parallel-size 2 --max-model-len 4096 --x {"a": 1}', calls)
        self.assertNotIn('--api-key ', r.stdout)                         # the key is never printed
        step = (logs / 'step0.log').read_text()
        self.assertIn('staged', step)                                    # weights copied to the stage root...
        self.assertIn(f'serve {self.tmp}/local/tiny ', calls)            # ...and served from there
        self.assertNotIn('failed', step)
        self.assertFalse((self.tmp / 'local' / 'tiny').exists())         # removed after the job
        self.assertTrue((self.tmp / 'scratch' / 'cache-dry' / 'vllm-n0-s1').is_dir())   # a cache per server
        ps = subprocess.run(['pgrep', '-f', f'fake_vllm.py {self.port}'], capture_output=True, text=True)
        self.assertEqual(ps.stdout.strip(), '', 'servers left running after the job')

    def test_vllm_step_that_exits_is_launched_again(self):
        r = self.vllm_job(1, {'FAKE_VLLM_FAIL_ONCE': '1'})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('launching it once more', r.stdout)
        self.assertIn('CUDA error (fake)', r.stdout)                     # why the first launch failed
        self.assertIn('1 of 1 servers up', r.stdout)

    def test_vllm_server_that_dies_beside_others_is_dropped_at_once(self):
        r = self.vllm_job(2, {'FAKE_VLLM_FAIL_ONCE': '1'})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('dropped: it exited', r.stdout)
        self.assertIn('1 of 2 servers up', r.stdout)

    def test_vllm_client_wait_and_failures(self):
        r = self.vllm_job(1, command='for s in 0 1; do ( sleep 1; echo "shard $s done" ) & done\nwait\nlog "clients finished"')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)          # `wait` does not wait for the servers
        self.assertIn('clients finished', r.stdout)
        r = self.vllm_job(1, command='python3 -c "import sys; sys.exit(5)"\necho "client finished"')
        self.assertEqual(r.returncode, 5, r.stdout + r.stderr)
        self.assertNotIn('client finished', r.stdout)
        self.assertIn('done (exit 5)', r.stdout)                         # servers stopped, cleanup ran


    def test_runs_on_one_node(self):
        runs = ['echo "run $RUN gpu $CUDA_VISIBLE_DEVICES omp $OMP_NUM_THREADS cache $XDG_CACHE_HOME" > r$RUN.txt',
                'echo "run $RUN" > r$RUN.txt\nfalse\necho "not reached"',
                'cat setup.txt > r$RUN.txt']
        s = drafts.draft(kind='gpu-node', name='dry', time='10', command='echo prepared > setup.txt', runs=runs)['script']
        r = self.run_job(s)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)         # run 1 failed under set -e
        self.assertIn('done: 1 of 3 runs failed', r.stdout)
        self.assertIn('run 1 finished (exit 1)', r.stdout)
        self.assertIn('run 2 finished (exit 0)', r.stdout)
        self.assertEqual((self.work / 'r0.txt').read_text(),
                         f'run 0 gpu 0 omp 58 cache {self.tmp}/scratch/cache-dry/r0\n')
        self.assertEqual((self.work / 'r2.txt').read_text(), 'prepared\n')
        log = (self.work / 'logs' / 'dry-4242-r1.log').read_text()
        self.assertNotIn('not reached', log)
        self.assertTrue((self.tmp / 'scratch' / 'cache-dry' / 'r2').is_dir())
        s = drafts.draft(kind='gpu-node', name='dry', time='10', runs=runs[::2])['script']
        r = self.run_job(s)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('run 1 on GPU 1', r.stdout)
        s = drafts.draft(kind='gpu-node', name='dry', time='10', command='exit 4', runs=runs[:1])['script']
        r = self.run_job(s)
        self.assertEqual(r.returncode, 4, r.stdout + r.stderr)
        self.assertIn('setup failed (exit 4)', r.stdout)
        self.assertNotIn('run 0 on GPU', r.stdout)

    def test_runs_see_the_setup_shell(self):                            # review: setup ran in a subshell
        (self.work / 'proj').mkdir()
        setup = 'cd proj\nexport MODE=offline\nhelper() { echo "helper $1"; }'
        s = drafts.draft(kind='gpu-node', name='dry', time='10', command=setup,
                         runs=['echo "$MODE $(basename "$PWD") $(helper $RUN)" > r$RUN.txt'])['script']
        r = self.run_job(s)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual((self.work / 'proj' / 'r0.txt').read_text(), 'offline proj helper 0\n')
        self.assertTrue((self.work / 'logs' / 'dry-4242-r0.log').exists())     # run logs stay in the job directory
        s = drafts.draft(kind='gpu-node', name='dry', time='10', command='false\necho no', runs=['echo r'])['script']
        r = self.run_job(s)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn('setup failed (exit 1)', r.stdout)
        self.assertNotIn('no\n', r.stdout)

    def test_staging_skips_the_owner_only_cache(self):                  # NOTES.md, Storage (owner-only HF .cache)
        self.assertEqual(drafts.STAGE.count('-path ./.cache -prune'), 2)
        if os.geteuid() == 0:
            self.skipTest('root reads everything')
        locked = self.tmp / 'data' / 'models' / 'tiny' / '.cache' / 'huggingface'
        locked.chmod(0)
        try:
            r = self.vllm_job(1)
        finally:
            locked.chmod(0o755)               # before tearDown removes the tree
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        step = (self.work / 'logs' / 'dry-4242' / 'step0.log').read_text()
        self.assertIn('staged', step)
        self.assertNotIn('failed', step)

    def test_array_args_reach_the_task(self):
        d = drafts.draft(**dict(SHAPES['array-args'], command='printf "%s|" $ARGS > a$TASK.txt; echo "$OUT" >> a$TASK.txt'))
        for task, want in [(0, f'--data|{self.tmp}/data/a|"x|y"|'), (1, "--data|b|--tag|it's|"), (2, '--x|`c`|\\d|')]:
            with self.subTest(task):
                r = self.run_job(d['script'], extra_env={'SLURM_ARRAY_TASK_ID': str(task), 'SLURM_ARRAY_JOB_ID': '77'})
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertEqual((self.work / f'a{task}.txt').read_text(), f'{want}results/t-arr-77-a{task}\n')
                self.assertTrue((self.work / f'results/t-arr-77-a{task}').is_dir())

    def test_missing_input_fails_fast(self):
        s = drafts.draft(kind='cpu', name='dry', time='5', python_env='$DATA/envs/nope', command='touch ran')['script']
        r = self.run_job(s)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn(f'missing {self.tmp}/data/envs/nope/bin/python', r.stdout)
        self.assertFalse((self.work / 'ran').exists())
        (self.tmp / 'data' / 'envs' / 'ok' / 'bin').mkdir(parents=True)
        (self.tmp / 'data' / 'envs' / 'ok' / 'bin' / 'python').write_text('')
        s = drafts.draft(kind='cpu', name='dry', time='5', python_env='$DATA/envs/ok', command='touch ran')['script']
        self.assertEqual(self.run_job(s).returncode, 0)
        s = drafts.draft(kind='gpu-node', name='dry', time='10', vllm_model='$DATA/models/typo')['script']
        r = self.run_job(s)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn('missing', r.stdout)
        self.assertFalse((self.tmp / 'srun.log').exists())               # no server step was launched

    def test_vllm_load_deadline(self):
        hang = self.tmp / 'hangbin'                                      # a vllm serve that never loads its weights
        hang.mkdir()
        (hang / 'apptainer').write_text('#!/bin/bash\necho "apptainer $*" >> "$DRY/apptainer.log"\n'
                                        f"exec python3 -c 'import time; time.sleep(60)' hang-marker-{self.tmp.name}\n")
        (hang / 'apptainer').chmod(0o755)
        self.port = free_port_base(1)
        s = drafts.draft(kind='gpu-node', name='dry', time='10', vllm_model='$DATA/models/tiny', load_cap_s=60)['script']
        r = self.run_job(s, timeout=40, extra_env={'SERVE_PORT': str(self.port), 'LOAD_CAP': '2',
                                                   'PATH': f'{hang}:{BIN}:{os.environ["PATH"]}'})
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn('server 0 dropped: weights not loaded 2s after its launch', r.stdout)
        self.assertIn('no server came up', r.stdout)


class Templates(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        jobs = self.tmp / 'other' / 'jobs'
        (jobs / 'lib').mkdir(parents=True)
        (jobs / 'bench.slurm').write_text('#!/bin/bash\n#ASC --vanilla\n# Benchmark on whole nodes.\n'
                                          '#SBATCH --partition=zen4_0768_h100x4\n#SBATCH --nodes=2\n'
                                          '# Usage: sbatch -N2 jobs/bench.slurm\nset -u\nsource jobs/lib/start.sh\n')
        (jobs / 'lib' / 'start.sh').write_text('start() { :; }\n')
        self.old = os.environ.get('AMADEUS_TEMPLATE_DIRS')
        os.environ['AMADEUS_TEMPLATE_DIRS'] = str(jobs)

    def tearDown(self):
        if self.old is None:
            os.environ.pop('AMADEUS_TEMPLATE_DIRS', None)
        else:
            os.environ['AMADEUS_TEMPLATE_DIRS'] = self.old
        shutil.rmtree(self.tmp)

    def test_index_and_get(self):
        idx = {e['name']: e for e in templates.index()}
        self.assertIn(f'{templates.ROOT.name}/mcp_smoke', idx)
        b = idx['other/bench']
        self.assertEqual(b['summary'], 'Benchmark on whole nodes. Usage: sbatch -N2 jobs/bench.slurm')
        self.assertEqual(b['resources'], {'partition': 'zen4_0768_h100x4', 'nodes': '2', 'vanilla': True})
        g = templates.get('bench')
        self.assertIn('source jobs/lib/start.sh', g['text'])
        self.assertEqual(g['helpers'], {'other/jobs/lib/start.sh': 'start() { :; }\n'})
        self.assertIn('error', templates.get('nope'))

    def test_listed_twice_still_found(self):
        jobs = self.tmp / 'other' / 'jobs'
        os.environ['AMADEUS_TEMPLATE_DIRS'] = os.pathsep.join([str(jobs), str(jobs / '..' / 'jobs')])
        self.assertEqual([e['name'] for e in templates.index()].count('other/bench'), 1)
        self.assertEqual(templates.get('other/bench')['name'], 'other/bench')

    def test_helper_paths_and_dirty_files(self):
        jobs = self.tmp / 'other' / 'jobs'
        (jobs / 'deep.slurm').write_text('#!/bin/bash\n# x\nsource $DATA/opt/tool/lib/start.sh\n'
                                         'bash "$ROOT/jobs/lib/start.sh"\n')
        self.assertEqual(list(templates.get('deep')['helpers']), ['other/jobs/lib/start.sh'])
        git = ['git', '-C', str(self.tmp / 'other'), '-c', 'user.name=t', '-c', 'user.email=t@t']
        subprocess.run(git + ['init', '-q'], check=True)
        subprocess.run(git + ['add', 'jobs/bench.slurm', 'jobs/lib'], check=True)
        subprocess.run(git + ['commit', '-qm', 'x'], check=True)
        src = {e['name']: e['source'] for e in templates.index()}
        self.assertRegex(src['other/bench'], r'^other@[0-9a-f]+$')
        self.assertTrue(src['other/deep'].endswith('-dirty'))           # not committed
        (jobs / 'bench.slurm').write_text((jobs / 'bench.slurm').read_text() + 'echo changed\n')
        self.assertTrue(templates.get('bench')['source'].endswith('-dirty'))


if __name__ == '__main__':
    unittest.main()
