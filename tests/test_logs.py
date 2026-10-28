import gzip
import os
import unittest

from amadeus import logs as L

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')

# snippets from real job logs (whole ones in tests/fixtures/, user and project IDs replaced)
TLS = """(Worker_TP0 pid=2059986) INFO 10-08 10:29:42 [fp8.py:359] Using TRITON Fp8 MoE backend
(Worker_TP0 pid=2059986) Exception in thread Thread-1 (_report_usage_worker):
(Worker_TP0 pid=2059986) Traceback (most recent call last):
(Worker_TP0 pid=2059986)   File "/usr/lib/python3.12/threading.py", line 1073, in _bootstrap_inner
(Worker_TP0 pid=2059986)     self.run()
(Worker_TP0 pid=2059986)   File "/usr/local/lib/python3.12/dist-packages/vllm/usage/usage_lib.py", line 268, in _send_to_server
(Worker_TP0 pid=2059986)     client.post(_USAGE_STATS_SERVER, json=data)
(Worker_TP0 pid=2059986)     raise OSError(
(Worker_TP0 pid=2059986) OSError: Could not find a suitable TLS CA certificate bundle, invalid path: /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
(Worker_TP0 pid=2059986) INFO 10-08 10:30:12 [gpu_model_runner.py:4210] Model loading took 35.2 GiB
"""
DP_FAIL = """(Worker_DP0_TP3_EP3 pid=2827467) ERROR 10-07 19:00:42 [multiproc_executor.py:944] WorkerProc failed to start.
(Worker_DP0_TP3_EP3 pid=2827467) ERROR 10-07 19:00:42 [multiproc_executor.py:944] Traceback (most recent call last):
(Worker_DP0_TP3_EP3 pid=2827467) ERROR 10-07 19:00:42 [multiproc_executor.py:944]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_DP0_TP2_EP2 pid=2827145) ERROR 10-07 19:00:42 [multiproc_executor.py:944] Traceback (most recent call last):
(Worker_DP0_TP3_EP3 pid=2827467) ERROR 10-07 19:00:42 [multiproc_executor.py:944]     return func(*args, **kwargs)
(Worker_DP0_TP3_EP3 pid=2827467) ERROR 10-07 19:00:42 [multiproc_executor.py:944]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_DP0_TP3_EP3 pid=2827467) ERROR 10-07 19:00:42 [multiproc_executor.py:944] NotImplementedError: Qwen4Exp HC does not support sequence-parallel MoE
(Worker_DP0_TP2_EP2 pid=2827145) ERROR 10-07 19:00:42 [multiproc_executor.py:944]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(Worker_DP0_TP2_EP2 pid=2827145) ERROR 10-07 19:00:42 [multiproc_executor.py:944] NotImplementedError: Qwen4Exp HC does not support sequence-parallel MoE
(EngineCore_DP0 pid=2824085) INFO 10-07 19:00:42 [multiproc_executor.py:472] [shutdown] Executor: waiting for worker exit count=4
[rank0]:[W1007 19:00:45.405276610 ProcessGroupNCCL.cpp:1624] Warning: WARNING: destroy_process_group() was not called before program exit
(EngineCore_DP0 pid=2824085) Traceback (most recent call last):
(EngineCore_DP0 pid=2824085)   File "/usr/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
(EngineCore_DP0 pid=2824085) Exception: WorkerProc initialization failed due to an exception in a background process. See stack trace for root cause.
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 10, in <module>
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
/usr/lib/python3.12/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked shared_memory objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
srun: error: n3011-008: task 0: Exited with exit code 1
"""
PROLOG = """[2026-10-08T11:34:19.014] error: Prolog hung on node n3012-022
[2026-10-08T11:34:19.014] error: Prolog hung on node n3010-022
srun: Step created for StepId=1001001.1
"""
STOP_SERVERS = """11:32:39 n3010-016 node=0 role=replica ip=10.0.0.10 cpus=352 gpus=0,1,2,3 tp=4 dp=2/1
srun: forcing job termination
srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
[2026-10-08T11:32:47.165] error: *** STEP 1001002.2 ON n3010-016 CANCELLED AT 2026-10-08T11:32:47 DUE to SIGNAL Killed ***
srun: error: n3010-016: task 0: Killed
"""
USERID = """11:32:39 n3010-019 node=1 role=replica ip=10.0.0.11 cpus=352 gpus=0,1,2,3 tp=4 dp=2/1
\x1b[31mFATAL:  \x1b[0m Couldn't determine user account information: user: unknown userid 100001
srun: error: n3010-019: task 0: Exited with exit code 255
"""
TQDM = ('Maximum request concurrency: 16\n\r  0%|          | 0/64 [00:00<?, ?it/s]'
        '\r  2%|▏         | 1/64 [00:12<12:40, 12.07s/it]'
        '\r  3%|▎         | 2/64 [00:12<05:20,  5.16s/it]'
        '\r100%|██████████| 64/64 [00:39<00:00,  1.64it/s]\n'
        '============ Serving Benchmark Result ============\n')
SHARDS = '\n'.join(f'Loading safetensors checkpoint shards: {p:3d}% Completed | {n}/131 [00:{n // 10:02d}<00:30,  3.56it/s]'
                   for n, p in ((0, 0), (2, 2), (65, 50), (131, 100))) + '\nINFO model loaded\n'
TRAIN_LOG = ('Loading weights: 100%|██████████| 723/723 [00:29<00:00, 24.6it/s]\n'
         'g0:  15%|█▍        | 184/1250 [09:09<53:00,  2.98s/it]\rg0:  15%|█▍        | 185/1250 '
         '[09:12<52:57,  2.98s/it, loss=0.412]\r')


def rows(state, exit_code='0:0', elapsed=600, **kw):
    return [{'State': state, 'ExitCode': exit_code, 'ElapsedRaw': str(elapsed), **kw}]


class Clean(unittest.TestCase):
    def test_ansi_and_junk(self):
        out = L.clean(USERID + 'bin\x00ary\x07 junk�\n')
        self.assertIn("FATAL:   Couldn't determine", out)
        self.assertNotIn('\x1b', out)
        self.assertIn('binary junk', out)

    def test_bytes_and_crlf(self):
        self.assertEqual(L.clean(b'INFO a \r\nINFO b\r\n'), 'INFO a\nINFO b\n')

    def test_carriage_returns_keep_last_state(self):
        out = L.clean(TQDM)
        self.assertIn('64/64 [00:39<00:00,  1.64it/s]', out)
        self.assertNotIn('1/64', out)
        self.assertEqual(out.splitlines()[0], 'Maximum request concurrency: 16')

    def test_progress_mode_keeps_updates(self):
        out = L.clean(TQDM, progress=True)
        self.assertIn('1/64 [00:12<12:40', out)
        self.assertEqual(sum('/64 [' in l for l in out.splitlines()), 4)

    def test_newline_bars_collapse(self):
        out = L.clean(SHARDS).splitlines()
        self.assertEqual(out[0], 'Loading safetensors checkpoint shards: 100% Completed | 131/131 [00:13<00:30,  3.56it/s]')
        self.assertEqual(out[1], 'INFO model loaded')

    def test_text_after_bar_survives(self):
        out = L.clean('\rFetching 144 files:   0%|          | 0/144 [00:00<?, ?it/s]Still waiting to acquire lock on '
                      '/data/x/.gitignore.lock (elapsed: 0.1 seconds)\nnext\n')
        self.assertIn('Still waiting to acquire lock', out)
        self.assertIn('next', out)


class Progress(unittest.TestCase):
    def test_tqdm_bar(self):
        (bar,) = L.progress(TQDM)
        self.assertEqual((bar['n'], bar['total'], bar['percent'], bar['rate'], bar['kind']), (64, 64, 100, '1.64it/s', 'work'))
        self.assertNotIn('label', bar)

    def test_vllm_shards_are_loading(self):
        (bar,) = L.progress(SHARDS)
        self.assertEqual(bar['label'], 'Loading safetensors checkpoint shards')
        self.assertEqual((bar['n'], bar['total'], bar['kind']), (131, 131, 'loading'))

    def test_loading_bar_is_not_work(self):
        # 2026-10-12: 'Loading weights ... 723/723' was read as 'datasets built'; the training bar is g0 185/1250
        load, train = L.progress(TRAIN_LOG)
        self.assertEqual((load['label'], load['kind']), ('Loading weights', 'loading'))
        self.assertEqual((train['label'], train['n'], train['total'], train['kind']), ('g0', 185, 1250, 'work'))
        self.assertEqual((train['eta'], train['rate'], train['postfix']), ('52:57', '2.98s/it', 'loss=0.412'))
        self.assertEqual(train['percent'], 15)

    def test_unknown_rate_and_eta_dropped(self):
        (bar,) = L.progress('Fetching 73 files:   0%|          | 0/73 [00:00<?, ?it/s]')
        self.assertNotIn('rate', bar)
        self.assertNotIn('eta', bar)
        self.assertEqual(bar['kind'], 'loading')

    def test_worker_prefix_labels_and_warmup(self):
        text = ('(Worker_TP0 pid=1) Capturing CUDA graphs (PIECEWISE):  50%|#####     | 25/51 [00:03<00:03, 8.1it/s]\n'
                '(Worker_TP1 pid=2) Capturing CUDA graphs (PIECEWISE):  40%|####      | 20/51 [00:03<00:04, 7.9it/s]\n')
        bars = L.progress(text)
        self.assertEqual([b['label'] for b in bars], ['(Worker_TP0) Capturing CUDA graphs (PIECEWISE)',
                                                      '(Worker_TP1) Capturing CUDA graphs (PIECEWISE)'])
        self.assertEqual(bars[0]['kind'], 'warmup')

    def test_percent_computed_without_bar_and_limit(self):
        text = ''.join(f'step{i}: {i * 10}/{i * 20} [00:01<00:01, 1.0it/s]\n' for i in range(1, 12))
        bars = L.progress(text, limit=3)
        self.assertEqual([b['label'] for b in bars], ['step9', 'step10', 'step11'])
        self.assertEqual(bars[-1]['percent'], 50.0)

    def test_hf_file_bars_are_loading(self):                          # review: hf download per-file bars
        (bar,) = L.progress('model.safetensors:  45%|####5     | 3.6G/8.0G [00:12<00:15, 290MB/s]')
        self.assertEqual(bar['kind'], 'loading')
        (bar,) = L.progress('step:  45%|####5     | 450/1000 [00:12<00:15, 3.1it/s]')
        self.assertEqual(bar['kind'], 'work')

    def test_byte_units(self):
        (bar,) = L.progress('model.safetensors:  53%|#####3    | 2.12G/4.00G [00:03<00:02, 702MB/s]')
        self.assertEqual((bar['n'], bar['total'], bar['rate']), ('2.12G', '4.00G', '702MB/s'))


class Benign(unittest.TestCase):
    def test_known_noise(self):
        for line in ['OSError: Could not find a suitable TLS CA certificate bundle, invalid path: /etc/pki/x.pem',
                     '/usr/lib/python3.12/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: '
                     'There appear to be 6 leaked shared_memory objects to clean up at shutdown',
                     'srun: forcing job termination',
                     '[2026-10-08T11:32:47.165] error: *** STEP 1001002.2 ON n3010-016 CANCELLED AT '
                     '2026-10-08T11:32:47 DUE to SIGNAL Killed ***',
                     'srun: error: n3010-016: task 0: Killed',
                     'DEBUG: Server startup function called',
                     '[notice] A new release of pip is available: 24.0 -> 26.2.1',
                     'W1012 22:59:01.123456 1234 torch/distributed/run.py:803] Setting OMP_NUM_THREADS',
                     '2026/10/07 17:45:01  warn xattr{/scratch/fs123456/x/rootfs} ignoring ENOTSUP on llistxattr',
                     '2026-10-08 12:31:38  [TileLang:tilelang.cache.kernel_cache:WARNING] (kernel_cache.py:357): '
                     'Found kernel in memory cache',
                     'n3011-008:2827467:2827467 [3] NCCL INFO ncclOsDlopen(libnccl-env.so) failed: No such file']:
            self.assertTrue(L.benign(line), line)

    def test_real_errors_are_not_benign(self):
        for line in ['srun: error: n3011-008: task 0: Exited with exit code 1',
                     '[2026-10-08T11:54:21.017] error: *** JOB 1001001 ON n3010-020 CANCELLED AT 2026-10-08T11:54:21 '
                     'DUE TO TIME LIMIT ***',
                     'Traceback (most recent call last):',
                     'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB',
                     '/var/spool/slurmd/job1001003/slurm_script: line 85: STATE: unbound variable']:
            self.assertFalse(L.benign(line), line)

    def test_filter_hides_harmless_traceback_keeps_real_one(self):
        text, hidden = L.filter_benign(TLS + DP_FAIL)
        self.assertNotIn('TLS CA', text)
        self.assertNotIn('usage_lib.py', text)
        self.assertNotIn('leaked shared_memory', text)
        self.assertIn('Model loading took 35.2 GiB', text)
        self.assertIn('NotImplementedError: Qwen4Exp HC', text)
        self.assertEqual(hidden[L.NOISE[0][1]], 8)


class Diagnose(unittest.TestCase):
    def names(self, d):
        return [f['name'] for f in d.get('findings', [])]

    def test_apptainer_unknown_userid_names_node(self):
        d = L.diagnose(USERID, rows('FAILED', '1:0', 120))
        self.assertEqual(d['category'], 'infrastructure')
        top = d['findings'][0]
        self.assertEqual((top['name'], top['nodes']), ('apptainer_unknown_userid', ['n3010-019']))
        self.assertIn('--exclude', d['summary'])
        self.assertIn('task_exit_255', self.names(d))

    def test_launch_failure_is_transient(self):
        log = PROLOG + ('srun: error: Task launch for StepId=1001004.0 failed on node n3012-022: Socket timed out on '
                        'send/recv operation\nsrun: error: n3012-022: task 3: Exited with exit code 140\n')
        d = L.diagnose(log, rows('FAILED', '140:0', 70))
        self.assertEqual(d['category'], 'infrastructure')
        self.assertEqual(self.names(d)[:2], ['socket_timeout', 'prolog_hung'])
        self.assertTrue(d['findings'][0]['transient'])
        self.assertEqual(d['findings'][1]['nodes'], ['n3012-022', 'n3010-022'])

    def test_prolog_hung_but_completed_is_benign(self):
        d = L.diagnose(PROLOG + 'done (exit 0)\n', rows('COMPLETED'))
        self.assertEqual(d['category'], 'benign')
        self.assertEqual(d['findings'][0]['category'], 'benign')
        self.assertNotIn('note', d)

    def test_node_fail_early(self):
        d = L.diagnose('', rows('NODE_FAIL', '0:0', 63, NodeList='n3011-[020-023]'))
        self.assertEqual(d['category'], 'infrastructure')
        f = d['findings'][0]
        self.assertEqual(f['name'], 'state_node_fail')
        self.assertTrue(f['transient'])
        self.assertIn('63 s', f['meaning'])
        self.assertIn('sinfo', f['action'])
        self.assertEqual(f['nodes'], ['n3011-[020-023]'])

    def test_timeout(self):
        log = '[2026-10-08T11:54:21.017] error: *** JOB 1001001 ON n3010-020 CANCELLED AT 2026-10-08T11:54:21 DUE TO TIME LIMIT ***\n'
        d = L.diagnose(log, rows('TIMEOUT', '0:0', 1200))
        self.assertEqual(d['category'], 'script')
        self.assertEqual(self.names(d), ['state_timeout', 'time_limit'])

    def test_unbound_variable(self):
        d = L.diagnose('/var/spool/slurmd/job1001003/slurm_script: line 85: STATE: unbound variable\n',
                       rows('FAILED', '1:0', 4))
        self.assertEqual((d['category'], self.names(d)[0]), ('script', 'unbound_variable'))
        self.assertIn('declare -A', d['findings'][0]['action'])

    def test_script_environment_signatures(self):
        cases = {"/tmp/x.c:5:10: fatal error: Python.h: No such file or directory": 'python_h_missing',
                 "/var/spool/slurmd/job1/slurm_script: line 9: module: command not found": 'module_not_found',
                 "cp: cannot make directory '/local/user/model/./.cache': File exists": 'cp_parents_race',
                 "sbatch: error: allocation failure: Requested node configuration is not available": 'node_config',
                 "user@n3010-020: Permission denied (publickey,gssapi-keyex,gssapi-with-mic).": 'ssh_node_refused',
                 "12:01:02 n3010-014 staging to /local/user/m failed: serving from /data/fs000000/x/m": 'staging_fallback',
                 "OSError: [Errno 28] No space left on device": 'disk_full',
                 "slurmstepd: error: Detected 1 oom_kill event in StepId=1.0. Some of the step tasks have been OOM Killed.": 'host_oom'}
        for line, name in cases.items():
            self.assertEqual(self.names(L.diagnose(line + '\n'))[0], name, line)

    def test_vllm_root_cause_is_first_exception(self):
        d = L.diagnose(DP_FAIL, rows('FAILED', '1:0', 240))
        self.assertEqual(d['category'], 'program')
        self.assertEqual(d['exceptions'][0], 'NotImplementedError: Qwen4Exp HC does not support sequence-parallel MoE')
        self.assertEqual(len(d['exceptions']), 3)
        self.assertIn('vllm_engine_init', self.names(d))
        self.assertIn('task_exit_1', self.names(d))
        self.assertIn('leaked shared_memory warning at shutdown', d['benign_hidden'])

    def test_tls_traceback_only_is_benign(self):
        d = L.diagnose(TLS + STOP_SERVERS, rows('COMPLETED'))
        self.assertEqual(d['category'], 'benign')
        self.assertNotIn('exceptions', d)
        self.assertNotIn('findings', d)
        self.assertEqual(d['benign_hidden'][L.NOISE[0][1]], 8)

    def test_engine_dead_after_shutdown_is_benign(self):
        dead = ('(APIServer pid=9) ERROR 10-08 12:36:01 [core_client.py:600] vllm.v1.engine.exceptions.EngineDeadError: '
                'EngineCore encountered an issue. See stack trace (above) for the root cause.\n')
        d = L.diagnose('(APIServer pid=9) INFO 10-08 12:36:00 [launcher.py:110] [shutdown] API server: shutdown '
                       'triggered\n' + dead, rows('COMPLETED'))
        self.assertEqual(d['category'], 'benign')
        self.assertEqual(self.names(d), ['vllm_engine_dead_shutdown'])
        d = L.diagnose(dead)
        self.assertEqual((d['category'], self.names(d)), ('program', ['vllm_engine_dead']))

    def test_shm_wait_harmless_during_compile_alarm_after_startup(self):
        wait = ('(EngineCore pid=5) WARNING 10-10 16:06:12 [shm_broadcast.py:466] No available shared memory broadcast '
                'block found in 60 seconds. This typically happens when some processes are hanging.\n')
        self.assertEqual(L.diagnose(wait)['category'], 'benign')
        d = L.diagnose('(APIServer pid=1) INFO 10-10 16:10:00 [launcher.py:80] Application startup complete.\n' + wait)
        self.assertEqual((d['category'], self.names(d)), ('program', ['shm_broadcast_wait_late']))

    def test_shm_broadcast_timeout_traceback(self):
        log = ('(EngineCore_DP0 pid=7) Traceback (most recent call last):\n'
               '(EngineCore_DP0 pid=7)   File "/usr/local/lib/python3.12/dist-packages/vllm/distributed/'
               'device_communicators/shm_broadcast.py", line 512, in acquire_read\n'
               '(EngineCore_DP0 pid=7) TimeoutError: RPC call to sample_tokens timed out.\nn3013-013 server 4 died\n')
        d = L.diagnose(log, rows('FAILED', '1:0'))
        self.assertEqual(self.names(d)[0], 'shm_broadcast_timeout')
        self.assertEqual(d['findings'][0]['nodes'], ['n3013-013'])

    def test_gpu_memory_signatures(self):
        free = ('ValueError: Free memory on device cuda:3 (91.1/93.09 GiB) on startup is less than desired GPU memory '
                'utilization (0.98, 91.23 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.')
        kv = ('ValueError: To serve at least one request with the models\'s max seq len (1048576), (28.5 GiB KV cache is '
              'needed, which is larger than the available KV cache memory (0.36 GiB).')
        oom = 'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 93.1 GiB'
        for line, name in ((free, 'vllm_free_memory'), (kv, 'vllm_kv_too_small'), (oom, 'cuda_oom')):
            d = L.diagnose(f'Traceback (most recent call last):\n  File "x.py", line 1, in <module>\n{line}\n')
            self.assertEqual((d['category'], self.names(d)), ('program', [name]), name)
            self.assertEqual(d['exceptions'], [line[:L.MAX_LINE]])

    def test_completed_but_errors_short_run(self):
        # a short run: COMPLETED after 4:35 of 2 h, every request refused
        log = 'Traceback (most recent call last):\n  File "client.py", line 9\nopenai.BadRequestError: Error code: 400\n'
        d = L.diagnose(log, rows('COMPLETED', '0:0', 275, TimelimitRaw='120'))
        self.assertEqual(d['category'], 'program')
        self.assertIn('COMPLETED, but the log shows errors', d['note'])
        self.assertIn('275 s of a 7200 s limit', d['note'])

    def test_one_killed_run_fails_the_job(self):
        # a packed node: FAILED 1:0 because run 2 was stopped (143) while three runs finished with 0
        d = L.diagnose('run 0 done (exit 0)\nrun 2 done (exit 143)\nsrun: error: n3010-014: task 2: Exited with exit '
                       'code 143\n', [{'state': 'FAILED', 'exit_code': '1:0', 'elapsed_s': 5000}])
        self.assertIn('task_exit_143', self.names(d))
        f = next(x for x in d['findings'] if x['name'] == 'task_exit_143')
        self.assertIn('SIGTERM', f['meaning'])
        self.assertEqual(d['category'], 'program')

    def test_signal_in_exit_code(self):
        d = L.diagnose('', rows('CANCELLED by 100001', '0:15'))
        self.assertIn('signal_15', self.names(d))
        self.assertIn('state_cancelled', self.names(d))

    def test_cancelled_is_benign(self):                                 # review: not 'unknown: signal 15'
        d = L.diagnose('', rows('CANCELLED', '0:15'))
        self.assertEqual(d['category'], 'benign')
        self.assertIn('cancelled', d['summary'])
        self.assertEqual(L.diagnose('', rows('FAILED', '0:9'))['category'], 'unknown')   # unexplained signal

    def test_prolog_hung_does_not_outrank_the_real_failure(self):        # review: dp-fail job logs
        log = PROLOG + DP_FAIL
        self.assertEqual(L.diagnose(log, rows('FAILED', '1:0', 240))['category'], 'program')
        d = L.diagnose(PROLOG + 'srun: error: n3012-022: task 0: Exited with exit code 1\n', rows('FAILED', '1:0', 70))
        self.assertEqual((d['category'], d['findings'][0]['name']), ('infrastructure', 'prolog_hung'))   # failed fast
        d = L.diagnose(PROLOG, rows('FAILED', '1:0', 900))
        self.assertNotEqual(d['category'], 'infrastructure')               # ran 15 min: the launch was fine

    def test_torchrun_rank_prefix(self):                                 # review: '[rank1]: Traceback'
        log = ('[rank1]: Traceback (most recent call last):\n[rank1]:   File "/x/train.py", line 10, in <module>\n'
               '[rank1]: ValueError: bad shape\n')
        d = L.diagnose(log, rows('FAILED', '1:0'))
        self.assertEqual(d['exceptions'], ['ValueError: bad shape'])
        self.assertIn('python_exception', self.names(d))

    def test_generic_script_hits_rank_after_specific_ones(self):        # review
        log = ('ls: cannot access /local/x: No such file or directory\nTraceback (most recent call last):\n'
               '  File "x.py", line 1\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB\n')
        self.assertEqual(L.diagnose(log, rows('FAILED', '1:0'))['category'], 'program')
        warn = ("WARNING 10-08 11:51:18 [cpu.py:608] Failed to import from vllm._C: ModuleNotFoundError(\"No module "
                "named 'vllm._C'\")\n")
        d = L.diagnose(warn, rows('COMPLETED'))
        self.assertEqual(d['category'], 'benign')
        self.assertNotIn('note', d)
        d = L.diagnose('WARNING: No module named foo, using the fallback\n', rows('FAILED', '1:0'))
        self.assertNotIn('no_python_module', self.names(d))
        d = L.diagnose("ModuleNotFoundError: No module named 'foo'\n", rows('FAILED', '1:0'))
        self.assertEqual(d['category'], 'script')

    def test_router_open_files(self):
        d = L.diagnose('2026-10-07 17:52:20 ERROR vllm_router_rs::routers::http::router: src/routers/http/router.rs:854: '
                       'Failed to send typed request worker_url=http://10.0.0.12:8000 route=/v1/chat/completions '
                       'error=error sending request for url\n' * 3)
        self.assertEqual((d['category'], d['findings'][0]['count']), ('script', 3))
        self.assertIn('ulimit -n 65536', d['summary'])

    def test_running_clean_and_failed_unknown(self):
        self.assertEqual(L.diagnose('step 10\n', rows('RUNNING', '0:0'))['category'], 'benign')
        d = L.diagnose('step 10\n', rows('FAILED', '3:0'))
        self.assertEqual(d['category'], 'unknown')
        self.assertIn('exit 3', d['findings'][0]['meaning'])

    def test_compact(self):
        d = L.diagnose(DP_FAIL * 50, rows('FAILED', '1:0'))
        self.assertLessEqual(len(d['findings']), L.MAX_FINDINGS)
        self.assertLessEqual(len(d['exceptions']), 5)
        self.assertTrue(all(len(f['line']) <= L.MAX_LINE for f in d['findings']))
        self.assertLess(len(repr(d)), 4000)


class RealLogs(unittest.TestCase):
    def read(self, rel):                         # 'results/<run>/<file>' or 'logs/<file>', stored gzipped
        name = rel.split('/', 1)[1].replace('/', '--') + '.gz'
        with gzip.open(os.path.join(FIXTURES, name), 'rb') as f:
            return f.read()

    def test_unknown_userid_fixture(self):
        d = L.diagnose(self.read('results/vllm-qwen38-16n-userid/replicas-server1.log'), rows('FAILED', '1:0', 120))
        self.assertEqual(d['findings'][0]['nodes'], ['n3010-019'])

    def test_healthy_16_node_server(self):
        # every one of these logs matched grep 'Traceback|Error' before (agent session report)
        d = L.diagnose(self.read('results/vllm-qwen38-16n-healthy/replicas-server0.log'))
        self.assertEqual(d['category'], 'benign')
        self.assertNotIn('exceptions', d)

    def test_dp_ep_failure(self):
        d = L.diagnose(self.read('results/vllm-qwen38-2n-fail/dp-server0.log'), rows('FAILED', '1:0'))
        self.assertTrue(d['exceptions'][0].startswith('NotImplementedError: Qwen4Exp HC'))

    def test_job_logs_are_a_program_failure(self):              # what job_details reads (StdOut+StdErr)
        text = self.read('logs/vllm-qwen38-2n-fail.out') + self.read('logs/vllm-qwen38-2n-fail.err')
        self.assertEqual(L.diagnose(text, rows('FAILED', '1:0', 240))['category'], 'program')

    def test_cancelled_by_the_user(self):
        text = self.read('logs/vllm-dsv41-b200-cancelled.out') + self.read('logs/vllm-dsv41-b200-cancelled.err')
        self.assertEqual(L.diagnose(text, rows('CANCELLED', '0:15', 2100))['category'], 'benign')

    def test_shutdown_engine_dead(self):
        d = L.diagnose(self.read('results/vllm-dsv41-b200-shutdown/tp4x2-server1.log'), rows('COMPLETED'))
        self.assertEqual(d['category'], 'benign')

    def test_bench_progress(self):
        (bar,) = L.progress(self.read('results/vllm-qwen38-1n-bench/bench-c16.log'))
        self.assertEqual((bar['n'], bar['total']), (64, 64))


class Milestones(unittest.TestCase):
    LOG = ('12:00:01 n3010-014 staged 173G in 21s\n'
           '12:04:12 server 0 (http://n3010-014:8000) ready after 251s\n'
           '12:04:13 server 1 (http://n3010-015:8000) ready after 252s\n'
           '12:04:13 2 of 2 servers up after 252s: http://n3010-014:8000/v1 http://n3010-015:8000/v1\n')

    def test_vllm_defaults(self):
        res = L.find_milestones(self.LOG, L.VLLM_MILESTONES)
        self.assertEqual([r['matched'] for r in res], [True, True, True, False])
        self.assertEqual(res[1]['count'], 2)
        self.assertEqual(res[1]['line_no'], 2)
        self.assertIn('ready after 251s', res[1]['line'])
        self.assertNotIn('line', res[3])

    def test_cleaned_and_invalid(self):
        res = L.find_milestones('\x1b[32mApplication startup complete.\x1b[0m\r\n', ['^Application startup complete', '('])
        self.assertTrue(res[0]['matched'])
        self.assertIn('invalid regex', res[1]['error'])

    def test_missing_note(self):
        self.assertIn('prolog', L.missing_note('RUNNING', 48))
        self.assertIn('prolog', L.missing_note('RUNNING', 900, 'Prolog'))
        self.assertEqual(L.missing_note('RUNNING', 900), 'not there')
        self.assertIn('pending', L.missing_note('PENDING'))


class Table(unittest.TestCase):
    def test_signature_rows(self):
        names = [s['name'] for s in L.SIGNATURES]
        self.assertEqual(len(names), len(set(names)))
        cats = {L.INFRA, L.SCRIPT, L.PROGRAM, L.BENIGN, L.UNKNOWN}
        for s in L.SIGNATURES:
            self.assertIn(s['category'], cats, s['name'])
            self.assertTrue(s['meaning'], s['name'])
            if s.get('scope') == 'exception':
                self.assertIn('exc', s)
            if s['category'] not in (L.BENIGN,):
                self.assertTrue(s['action'], s['name'])
        for code, (cat, meaning) in L.EXIT_CODES.items():
            self.assertIn(cat, cats)


if __name__ == '__main__':
    unittest.main()
