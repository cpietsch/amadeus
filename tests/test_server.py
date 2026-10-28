"""End to end through an in-process MCP client, against a fake cluster (tests/fake_ssh.sh + tests/fakebin)."""
import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent

GOOD = """#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=dev_zen4_0768_h100x4
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
echo hi
"""


# Stand-in for `musica_auth.sh --check` (never the real one): $FAKE_HOME/cert holds the minutes left (no file: no
# certificate), FAKE_CHECK_S makes it slow; every call is logged.
FAKE_CHECK = r'''#!/bin/bash
echo "$*" >> "$FAKE_HOME/auth-calls.log"
sleep "${FAKE_CHECK_S:-0}"
[ -f "$FAKE_HOME/cert" ] || { echo "no certificate"; exit 1; }
left=$(cat "$FAKE_HOME/cert")
echo "certificate valid until $(date -d "+$left min +30 sec" +%Y-%m-%dT%H:%M:%S) ($left min left)"
[ "$left" -ge 60 ]
'''


class FakeCluster(unittest.TestCase):
    def setUp(self):
        from amadeus import auth
        self.tmp = Path(tempfile.mkdtemp())
        self.home = self.tmp / 'home'
        (self.home / 'data' / 'proj').mkdir(parents=True)
        (self.home / 'scratch').mkdir()
        (self.home / 'data' / 'proj' / 'job.slurm').write_text(GOOD)
        check = self.tmp / 'musica_auth.sh'
        check.write_text(FAKE_CHECK)
        check.chmod(0o755)
        (self.home / 'cert').write_text('900')       # plenty left: no certificate fields unless a test changes it
        self.env = {'AMADEUS_SSH': f'bash {HERE / "fake_ssh.sh"}', 'FAKE_HOME': str(self.home),
                    'AMADEUS_LEDGER': str(self.tmp / 'jobs.jsonl'), 'AMADEUS_RETRY_DELAY': '0',
                    'AMADEUS_LOCAL_ROOT': str(self.tmp), 'AMADEUS_ISSUES': str(self.tmp / 'issues'),
                    'AMADEUS_AUTH_SCRIPT': str(check), 'FAKE_CHECK_S': '0'}
        self.old = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        auth.forget_certificate()
        from amadeus import server
        server._ALARMS_SENT.clear()               # job ids repeat across tests

    def tearDown(self):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp)

    def call(self, name, args=None):
        from mcp import Client
        from amadeus.server import mcp

        async def go():
            async with Client(mcp) as c:
                r = await c.call_tool(name, args or {})
                self.assertFalse(r.is_error, r.content)
                return r.structured_content if r.structured_content is not None else json.loads(r.content[0].text)
        return asyncio.run(go())

    def call_raw(self, name, args=None):
        from mcp import Client
        from amadeus.server import mcp

        async def go():
            async with Client(mcp) as c:
                return await c.call_tool(name, args or {})
        return asyncio.run(go())

    def slurm_log(self):
        p = self.home / 'slurm.log'
        return p.read_text() if p.exists() else ''

    def test_budget_per_user_with_uids_named(self):
        uid = os.getuid()
        name = subprocess.run(['id', '-un'], capture_output=True, text=True).stdout.strip()
        (self.home / 'sacct.txt').write_text(
            f'{uid}|zen4_0768_h100x4|3600|billing=1,gres/np_zen4_0768_h100x4=100\n'
            'someone|zen4_0768|7200|gres/np_zen4_0768=50\n'
            'someone|zen5_2304_b200x8|1800|gres/np_zen5_2304_b200x8=100\n')
        r = self.call('budget')
        self.assertEqual(r['tokens_by_user'], {name: 4.0, 'someone': 0.96})
        self.assertEqual(r['other_partitions_node_hours'], {'someone': 0.5})
        self.assertEqual(r['tokens_total'], 5.0)

    def test_budget_without_account_shows_own_jobs(self):
        old = os.environ.get('AMADEUS_CONFIG')
        os.environ['AMADEUS_CONFIG'] = str(self.home / 'missing.toml')
        try:
            (self.home / 'sacct.txt').write_text('someone|zen4_0768_h100x4|3600|gres/np_zen4_0768_h100x4=100\n')
            r = self.call('budget')
        finally:
            os.environ['AMADEUS_CONFIG'] = old
        self.assertEqual((r['tokens_total'], r['account']), (4.0, None))
        self.assertIn('no Slurm account configured', r['account_note'])

    def test_cluster_load_planned_nodes_are_not_idle(self):
        # Slurm's %F counts planned (held by backfill for jobs next in line), reserved and maintenance nodes as idle
        (self.home / 'sinfo.txt').write_text(
            'cpu|planned|8\ncpu|allocated|59\ncpu|mixed|2\ncpu|idle|1\ncpu|idle+planned|3\ncpu|idle-|1\n'
            'cpu|reserved|2\ncpu|idle+maint|1\ncpu|idle*|1\ncpu|idle~|4\ncpu|drained*|1\n'
            'gpu|allocated|10\ngpu|idle|2\n')
        r = self.call('cluster_load')
        cpu = r['nodes']['cpu']
        self.assertEqual({k: cpu[k] for k in ('allocated', 'idle', 'planned', 'other', 'total')},
                         {'allocated': 61, 'idle': 5, 'planned': 12, 'other': 5, 'total': 83})
        self.assertEqual(cpu['states']['planned'], 8)
        self.assertEqual({k: r['nodes']['gpu'][k] for k in ('allocated', 'idle', 'planned', 'other', 'total')},
                         {'allocated': 10, 'idle': 2, 'planned': 0, 'other': 0, 'total': 12})
        self.assertIn('planned_note', r)
        (self.home / 'sinfo.txt').write_text('gpu|allocated|10\ngpu|idle|2\n')
        self.assertNotIn('planned_note', self.call('cluster_load'))

    def test_node_state_classes(self):
        from amadeus.server import _node_class
        for state, cls in [('idle', 'idle'), ('idle~', 'idle'), ('idle#', 'idle'), ('idle%', 'idle'),
                           ('planned', 'planned'), ('PLANNED', 'planned'), ('idle+planned', 'planned'),
                           ('plnd', 'planned'), ('idle-', 'planned'), ('mixed+planned', 'allocated'),
                           ('mixed-', 'allocated'), ('allocated', 'allocated'), ('mix', 'allocated'),
                           ('completing', 'allocated'), ('allocated+drain', 'other'), ('draining', 'other'),
                           ('drained*', 'other'), ('down*', 'other'), ('reserved', 'other'), ('resv', 'other'),
                           ('idle+reserved', 'other'), ('maint', 'other'), ('idle+maint', 'other'),
                           ('idle$', 'other'), ('idle*', 'other'), ('idle@', 'other'), ('idle^', 'other'),
                           ('reboot_requested', 'other'), ('fail', 'other'), ('future', 'other'),
                           ('unknown', 'other'), ('', 'other')]:
            self.assertEqual(_node_class(state), cls, state)

    def test_tools_listed(self):
        from mcp import Client
        from amadeus.server import mcp

        async def go():
            async with Client(mcp) as c:
                tools = {t.name for t in (await c.list_tools()).tools}
                res = await c.read_resource('amadeus://runbook')
                return tools, res
        tools, res = asyncio.run(go())
        for t in ('update_job', 'report_issue', 'list_issues', 'access_status', 'start_login', 'run_command', 'list_jobs', 'job_details', 'job_log', 'wait_for_job',
                  'cluster_load', 'budget', 'check_job', 'submit_job', 'draft_job', 'job_templates', 'cancel_job', 'push_files', 'fetch_files',
                  'ledger_entries', 'node_exec'):
            self.assertIn(t, tools)
        self.assertIn('Operating MUSICA', res.contents[0].text)

    def test_run_command(self):
        r = self.call('run_command', {'command': 'echo $DATA; exit 3'})
        self.assertEqual(r['exit_code'], 3)
        self.assertTrue(r['stdout'].strip().endswith('/home/data'))
        self.assertIn('error', self.call('run_command', {'command': 'salloc -N1'}))
        self.assertIn('submit_job', self.call('run_command', {'command': 'cd x && sbatch -N 16 job.slurm'})['error'])
        self.assertIn('error', self.call('run_command', {'command': 'scontrol update JobId=1 TimeLimit=72:00:00'}))
        r = self.call('run_command', {'command': 'rm -rf $SCRATCH/old'})
        self.assertIn('approval', r['error'])
        r = self.call('run_command', {'command': 'rm -rf $SCRATCH/old', 'approval': 'Alex: delete old'})
        self.assertEqual(r['exit_code'], 0)
        self.assertEqual(self.slurm_log(), '')

    def test_run_command_sacctmgr_read_only(self):
        r = self.call('run_command', {'command': 'sacctmgr -nP show qos format=Name,MaxWall'})
        self.assertEqual(r['exit_code'], 0)
        self.assertIn('3-00:00:00', r['stdout'])
        for c in ('sacctmgr -i delete user x', 'sacctmgr modify user x set qos=long', 'sacctmgr',
                  "timeout 30 bash -c 'sacctmgr add qos x'"):
            self.assertIn('the human (account administration)', self.call('run_command', {'command': c})['error'], c)
        self.assertEqual(self.slurm_log(), 'sacctmgr -nP show qos format=Name,MaxWall\n')

    def test_check_script_text_without_pushing(self):
        tmpdir = self.tmp / 'login-tmp'
        tmpdir.mkdir()
        os.environ['TMPDIR'] = str(tmpdir)
        try:
            draft = self.call('draft_job', {'kind': 'gpu-node', 'name': 'drafted', 'time': '5', 'command': 'echo hi'})
            r = self.call('check_job', {'script_text': draft['script'], 'cwd': '$DATA/proj'})
        finally:
            os.environ.pop('TMPDIR')
        self.assertTrue(r['ready'], r)
        self.assertIn('Job 999', r['test_only']['output'])
        self.assertEqual(r['options']['job-name'], 'drafted')
        self.assertRegex(self.slurm_log(), r'sbatch --test-only \S*/amadeus-\w+\.slurm')
        self.assertEqual((self.home / 'test-only.slurm').read_text(), draft['script'])   # the text reached sbatch
        self.assertEqual(list(tmpdir.iterdir()), [])                         # the temporary script is removed
        self.assertIn('error', self.call('check_job', {'cwd': '$DATA/proj'}))
        self.assertIn('error', self.call('check_job', {'script': 'job.slurm', 'script_text': GOOD}))
        r = self.call('check_job', {'script_text': GOOD, 'cwd': '$DATA/newproj'})     # not pushed yet
        self.assertTrue(r['ready'], r)
        self.assertTrue(any('does not exist on the cluster yet' in w for w in r['warnings']))
        bad = self.call('check_job', {'script_text': GOOD.replace('#SBATCH --time=00:10:00\n', ''), 'cwd': '$DATA/proj'})
        self.assertFalse(bad['ready'])
        self.assertTrue(any('--time' in e for e in bad['errors']))

    def test_script_text_passed_as_script_path_is_refused_without_ssh(self):
        from amadeus import server
        long_line = '#!/bin/bash ' + 'x' * 200
        cases = [('check_job', {'script': GOOD, 'cwd': '$DATA/proj'}),
                 ('submit_job', {'script': GOOD, 'cwd': '$DATA/proj', 'purpose': 'smoke test'}),
                 ('submit_job', {'script': '#SBATCH --time=5', 'purpose': 'smoke test'}),
                 ('submit_job', {'script': long_line, 'purpose': 'smoke test'})]
        with mock.patch.object(server.remote, 'run', side_effect=AssertionError('SSH call')) as run:
            for name, args in cases:
                err = self.call(name, args)['error']
                self.assertIn('path of a job script on the cluster', err)
                self.assertIn('check_job(script_text=...)', err)
                self.assertIn('push_files', err)
                self.assertNotIn('echo hi', err)                         # the script is not echoed back
                self.assertLess(len(err), 400)
            run.assert_not_called()
        self.assertEqual(self.slurm_log(), '')
        self.assertTrue(self.call('check_job', {'script': 'job.slurm', 'cwd': '$DATA/proj'})['ready'])   # paths still work

    def test_draft_and_templates_tools(self):
        r = self.call('draft_job', {'kind': 'cpu', 'name': 'x', 'cores': 4, 'time': '5'})
        self.assertIn('#SBATCH --cpus-per-task=4', r['script'])
        self.assertEqual(r['check']['errors'], [])
        self.assertIn('error', self.call('draft_job', {'kind': 'gpu', 'name': 'x', 'gpus': 4}))
        idx = self.call('job_templates')
        from amadeus import templates
        self.assertIn(f'{templates.ROOT.name}/mcp_smoke', [t['name'] for t in idx['templates']])
        self.assertIn('#SBATCH --job-name=mcp-smoke', self.call('job_templates', {'name': 'mcp_smoke'})['text'])
        self.assertEqual(self.slurm_log(), '')                                # nothing went to the cluster

    def test_check_and_submit(self):
        r = self.call('check_job', {'script': 'job.slurm', 'cwd': '$DATA/proj'})
        self.assertTrue(r['ready'], r)
        self.assertIn('Job 999', r['test_only']['output'])
        r = self.call('submit_job', {'script': 'job.slurm', 'cwd': '$DATA/proj', 'purpose': 'smoke test'})
        self.assertTrue(r['submitted'], r)
        self.assertEqual(r['job_id'], '4242')
        log = self.slurm_log()
        self.assertIn('--account=p123456', log)
        self.assertIn('--exclude=n3010-019', log)
        self.assertIn('--hold', log)
        self.assertIn('scontrol release 4242', log)
        self.assertFalse(r['held'])
        self.assertTrue((self.home / 'data' / 'proj' / 'logs').is_dir())    # log directory created
        entries = [json.loads(l) for l in (self.tmp / 'jobs.jsonl').read_text().splitlines()]
        self.assertEqual((entries[-1]['job_id'], entries[-1]['purpose']), ('4242', 'smoke test'))

    def test_approval_required(self):
        args = {'script': 'job.slurm', 'cwd': '$DATA/proj', 'purpose': 'big',
                'sbatch_args': ['--qos=zen4_0768_h100x4', '-N', '4', '--time=00:30:00']}
        r = self.call('submit_job', args)
        self.assertFalse(r['submitted'])
        self.assertTrue(r['needs_approval'])
        self.assertNotIn('--parsable', self.slurm_log())
        (self.home / 'scontrol.txt').write_text('JobId=4242 JobName=smoke JobState=PENDING Partition=zen4_0768_h100x4 '
                                                'QOS=zen4_0768_h100x4 NumNodes=4 TimeLimit=00:30:00\n')
        r = self.call('submit_job', {**args, 'approval': 'Alex: "4 nodes for 30 min OK"'})
        self.assertTrue(r['submitted'], r)
        self.assertIn('scontrol release 4242', self.slurm_log())

    def test_slurm_view_overrules_the_script(self):
        # the script says 1 node, but Slurm reads 16 (e.g. an option spelling the checks did not catch): cancelled
        (self.home / 'scontrol.txt').write_text('JobId=4242 JobName=smoke JobState=PENDING Partition=zen4_0768_h100x4 '
                                                'QOS=zen4_0768_h100x4 NumNodes=16 TimeLimit=00:10:00\n')
        r = self.call('submit_job', {'script': 'job.slurm', 'cwd': '$DATA/proj', 'purpose': 'x',
                                     'sbatch_args': ['--qos=zen4_0768_h100x4']})
        self.assertFalse(r['submitted'], r)
        self.assertIn('scancel 4242', self.slurm_log())
        self.assertNotIn('release', self.slurm_log())

    def test_errors_refuse(self):
        r = self.call('submit_job', {'script': 'job.slurm', 'cwd': '$DATA/proj', 'purpose': 'x',
                                     'sbatch_args': ['--time=01:00:00']})
        self.assertFalse(r['submitted'])
        self.assertTrue(any('dev QoS' in e for e in r['errors']))

    def test_concurrent_nodes_count(self):
        (self.home / 'squeue.txt').write_text('1|a|RUNNING|1:00|1:00:00|2|zen4_0768_h100x4|q|n1|N/A|x\n'
                                              '2|b|PENDING|0:00|1:00:00|1|zen4_0768_h100x4|q|Priority|N/A|x\n')
        r = self.call('check_job', {'script': 'job.slurm', 'cwd': '$DATA/proj'})
        self.assertEqual(r['in_use']['nodes'], 3)
        self.assertTrue(any('at once' in a for a in r['needs_approval']))   # 3 in use + 1 = 4 nodes at once
        jobs = self.call('list_jobs')
        self.assertEqual(jobs['in_use']['nodes'], 3)
        self.assertGreater(jobs['in_use']['tokens'], 0)

    def test_fresh_job_that_already_ended_does_not_count(self):
        r = self.call('submit_job', {'script': 'job.slurm', 'cwd': '$DATA/proj', 'purpose': 'p'})
        self.assertTrue(r['submitted'], r)
        # squeue no longer shows it: still counted while sacct does not know it, or shows it running
        self.assertEqual(self.call('check_job', {'script': 'job.slurm', 'cwd': '$DATA/proj'})['in_use']['nodes'], 1)
        (self.home / 'sacct.txt').write_text('4242|RUNNING\n')
        self.assertEqual(self.call('check_job', {'script': 'job.slurm', 'cwd': '$DATA/proj'})['in_use']['nodes'], 1)
        (self.home / 'sacct.txt').write_text('4242|COMPLETED\n')            # a job that finished within seconds
        self.assertEqual(self.call('check_job', {'script': 'job.slurm', 'cwd': '$DATA/proj'})['in_use']['nodes'], 0)

    def test_cancel_needs_approval_for_foreign_jobs(self):
        r = self.call('cancel_job', {'job_id': '777', 'reason': 'stuck'})
        self.assertFalse(r['cancelled'])
        self.call('submit_job', {'script': 'job.slurm', 'cwd': '$DATA/proj', 'purpose': 'p'})
        r = self.call('cancel_job', {'job_id': '4242', 'reason': 'test'})
        self.assertTrue(r['cancelled'])
        self.assertIn('scancel 4242', self.slurm_log())
        r = self.call_raw('cancel_job', {'job_id': '1; rm -rf ~', 'reason': 'x'})
        self.assertTrue(r.is_error)
        (self.home / 'scancel.fail').write_text('')
        n = len((self.tmp / 'jobs.jsonl').read_text().splitlines())
        r = self.call('cancel_job', {'job_id': '4242', 'reason': 'again'})
        self.assertFalse(r['cancelled'])
        self.assertEqual(len((self.tmp / 'jobs.jsonl').read_text().splitlines()), n)

    def test_update_job(self):
        r = self.call('update_job', {'job_id': '777', 'time_limit': '02:00:00'})
        self.assertFalse(r['updated'])                     # not ours, no approval
        self.call('submit_job', {'script': 'job.slurm', 'cwd': '$DATA/proj', 'purpose': 'p'})
        (self.home / 'scontrol.txt').write_text('JobId=4242 JobName=smoke JobState=PENDING Partition=zen4_0768_h100x4 '
                                                'QOS=zen4_0768_h100x4 NumNodes=2 TimeLimit=01:00:00\n')
        r = self.call('update_job', {'job_id': '4242', 'time_limit': '04:00:00'})   # 2 nodes x 4 h = 32 tokens
        self.assertFalse(r['updated'])
        self.assertTrue(r['needs_approval'])
        r = self.call('update_job', {'job_id': '4242', 'time_limit': '02:00:00'})   # 16 tokens, pending: allowed
        self.assertTrue(r['updated'], r)
        self.assertIn('scontrol update JobId=4242 TimeLimit=02:00:00', self.slurm_log())

    def test_update_job_refuses_raising_a_running_jobs_limit(self):
        self.call('submit_job', {'script': 'job.slurm', 'cwd': '$DATA/proj', 'purpose': 'p'})
        (self.home / 'scontrol.txt').write_text('JobId=4242 JobName=smoke JobState=RUNNING Partition=zen4_0768_h100x4 '
                                                'QOS=zen4_0768_h100x4 NumNodes=1 TimeLimit=03:30:00\n')
        n = len(self.slurm_log().splitlines())
        r = self.call('update_job', {'job_id': '4242', 'time_limit': '04:30:00', 'approval': 'Alex: ok'})
        self.assertFalse(r['updated'])
        self.assertIn('only while the job is pending', r['error'])
        self.assertTrue(any('checkpoint' in o for o in r['options']))
        self.assertNotIn('scontrol update', '\n'.join(self.slurm_log().splitlines()[n:]))   # refused before scontrol
        r = self.call('update_job', {'job_id': '4242', 'time_limit': '02:00:00'})            # lowering works
        self.assertTrue(r['updated'], r)
        self.assertIn('scontrol update JobId=4242 TimeLimit=02:00:00', self.slurm_log())

    # ------------------------------------------------------------ node_exec

    def running(self, user='tester', state='RUNNING', nodes='n3010-[014-015]'):
        (self.home / 'scontrol.txt').write_text(
            f'JobId=4242 JobName=smoke UserId={user}(1) JobState={state} Partition=zen4_0768_h100x4 '
            f'QOS=zen4_0768_h100x4 NumNodes=2 NodeList={nodes} TimeLimit=01:00:00\n')

    def test_node_exec_runs_on_the_jobs_node(self):
        self.running()
        r = self.call('node_exec', {'job_id': '4242', 'command': 'echo on-node; hostname >/dev/null'})
        self.assertEqual(r['exit_code'], 0, r)
        self.assertEqual(r['stdout'].strip(), 'on-node')
        self.assertEqual(r['node'], 'n3010-014')                          # the first node by default
        log = self.slurm_log()
        self.assertIn('srun --jobid=4242 --overlap -N1 -n1 --cpus-per-task=1 --nodelist=n3010-014 timeout 60 bash -c',
                      log)
        r = self.call('node_exec', {'job_id': '4242', 'command': 'top -b -n1 >/dev/null; echo ok', 'node': 'n3010-015',
                                    'timeout_s': 999})
        self.assertEqual(r['node'], 'n3010-015')
        self.assertIn('--nodelist=n3010-015 timeout 120 bash', self.slurm_log())   # timeout capped
        self.assertIn('error', self.call('node_exec', {'job_id': '4242', 'command': 'ps', 'node': 'n3011-001'}))
        self.assertIn('error', self.call('node_exec', {'job_id': '4242', 'command': 'ps', 'node': 'n1;rm'}))
        (self.home / 'srun.fail').write_text('')
        r = self.call('node_exec', {'job_id': '4242', 'command': 'ps'})
        self.assertIn('could not start a step', r['note'])

    def test_node_exec_only_in_own_running_jobs(self):
        self.running(user='someone')
        self.assertIn('belongs to someone', self.call('node_exec', {'job_id': '4242', 'command': 'ps'})['error'])
        self.running(state='PENDING', nodes='(null)')
        self.assertIn('RUNNING', self.call('node_exec', {'job_id': '4242', 'command': 'ps'})['error'])
        self.assertTrue(self.call_raw('node_exec', {'job_id': '42; ls', 'command': 'ps'}).is_error)
        self.assertNotIn('srun', self.slurm_log())

    def test_node_exec_array_task(self):
        (self.home / 'scontrol.txt').write_text(
            'JobId=501 ArrayJobId=500 ArrayTaskId=1 UserId=tester(1) JobState=RUNNING NodeList=n3010-001\n'
            'JobId=502 ArrayJobId=500 ArrayTaskId=2 UserId=tester(1) JobState=RUNNING NodeList=n3010-002\n')
        self.assertIn('500_1', self.call('node_exec', {'job_id': '500', 'command': 'ps'})['error'])
        r = self.call('node_exec', {'job_id': '500_2', 'command': 'echo x'})
        self.assertEqual(r['node'], 'n3010-002', r)
        self.assertIn('srun --jobid=502 ', self.slurm_log())

    def test_node_exec_policy(self):
        self.running()
        for cmd in ('sleep 100 &', 'nohup python x.py', 'setsid foo', 'bash -c "sleep 9 &"', 'srun hostname',
                    'scancel 4242', 'top', 'watch nvidia-smi', 'cat ~/.ssh/id_ed25519', 'echo "unbalanced'):
            r = self.call('node_exec', {'job_id': '4242', 'command': cmd})
            self.assertIn('error', r, cmd)
            self.assertNotIn('approval', r.get('note', ''), cmd)
        for cmd in ('kill -STOP 1234', 'pkill -f vllm', 'timeout 5 kill 1', 'bash -c "kill 1"', 'rm -rf /local/x',
                    # wrapped forms (review): wrapper options and values are not the command word
                    'timeout 5s pkill -f vllm', 'pgrep python | xargs -n 1 kill', 'nice -n 5 kill 123',
                    'env -u X kill 1', 'timeout -s 9 5 kill 1', 'bash -lc "kill 1"', 'bash -ec "pkill vllm"',
                    'timeout 10s rm -rf $SCRATCH/x', 'bash -lc "rm -rf $SCRATCH/x"', 'fuser -k /dev/nvidia0'):
            r = self.call('node_exec', {'job_id': '4242', 'command': cmd})
            self.assertIn('approval', r['error'], cmd)
        self.assertNotIn('srun', self.slurm_log())
        r = self.call('node_exec', {'job_id': '4242', 'command': 'kill -0 $$ && echo alive',
                                    'approval': 'Alex: stop the stuck worker'})
        self.assertEqual(r['exit_code'], 0, r)
        entries = [json.loads(l) for l in (self.tmp / 'jobs.jsonl').read_text().splitlines()]
        self.assertEqual(entries[-1]['event'], 'node_exec')
        for cmd in ('timeout 10s nvidia-smi', 'ps aux | grep kill', 'fuser -v /dev/nvidia0', 'nice -n 5 free -g'):
            from amadeus import server
            self.assertEqual(server.node_exec_policy(cmd)['needs_approval'], [], cmd)
        # pipes, redirects and 2>&1 are fine
        r = self.call('node_exec', {'job_id': '4242', 'command': 'echo a 2>&1 | grep -c a'})
        self.assertEqual(r['stdout'].strip(), '1')

    def test_run_command_points_to_node_exec(self):
        r = self.call('run_command', {'command': 'timeout 60 srun --jobid=1 --overlap nvidia-smi'})
        self.assertIn('node_exec', r['note'])
        r = self.call('run_command', {'command': 'ssh -o BatchMode=yes n3010-014 nvidia-smi'})
        self.assertIn('node_exec', r['error'])
        self.assertEqual(self.slurm_log(), '')

    def test_job_details_and_log(self):
        logs = self.home / 'data' / 'proj' / 'logs'
        logs.mkdir()
        (logs / 'smoke-4242.out').write_text('line1\nERROR boom\nline3\n')
        wd = str(self.home / 'data' / 'proj')
        (self.home / 'sacct.txt').write_text(
            f'4242|4242|smoke|tester|zen4_0768_h100x4|dev|COMPLETED|0:0|s|s|e|600|1|n1|gres/np_zen4_0768_h100x4=100|'
            f'logs/%x-%j.out||{wd}|10\n')
        r = self.call('job_details', {'job_id': '4242'})
        self.assertEqual(r['rows'][0]['tokens'], 0.667)
        self.assertIn('ERROR boom', r['logs'])
        r = self.call('job_log', {'job_id': '4242', 'grep': 'ERROR'})
        self.assertIn('ERROR boom', r['text'])
        self.assertNotIn('line1', r['text'])

    def test_log_paths_are_never_shell_code(self):
        wd = str(self.home / 'data' / 'proj')
        (self.home / 'sacct.txt').write_text(
            f'4242|4242|x|tester|zen4_0768_h100x4|dev|COMPLETED|0:0|s|s|e|60|1|n1||'
            f'logs/$(touch {self.home}/pwned).out||{wd}|10\n')
        self.call('job_details', {'job_id': '4242'})
        self.assertFalse((self.home / 'pwned').exists())

    def test_array_log_patterns(self):
        from amadeus.server import _resolve
        row = {'JobID': '100_3', 'JobIDRaw': '103', 'JobName': 'arr', 'User': 'u', 'NodeList': 'n3010-[002-003]',
               'WorkDir': '/w'}
        self.assertEqual(_resolve('logs/%x-%A_%a-%j.out', row), '/w/logs/arr-100_3-103.out')
        self.assertEqual(_resolve('%x-%4a-%N.out', row), '/w/arr-0003-n3010-002.out')

    # ------------------------------------------------------------ log reading and diagnosis (findings 10, 32, 38, 42, 43, 64, 102)

    def test_job_log_cleans_greps_and_reads_progress(self):
        self.jobs(('4242', 'RUNNING', 600))
        self.write_log('4242', '\x1b[32mINFO\x1b[0m start\x00\x01\n'
                               'Loading weights:  50%|#####     | 361/723 [00:10<00:10, 36.1it/s]\r'
                               'Loading weights: 100%|##########| 723/723 [00:20<00:00, 36.1it/s]\n'
                               'train:   1%|          | 10/4858 [00:40<5:23:00, 4.0s/it, loss=0.91]\r'
                               'train:   1%|          | 12/4858 [00:48<5:23:00, 4.0s/it, loss=0.87]\n'
                               'srun: forcing job termination\n'
                               'OSError: Could not find a suitable TLS CA certificate bundle, invalid path: /etc/pki\n')
        r = self.call('job_log', {'job_id': '4242'})
        text = r['text']
        self.assertNotIn('\x1b', text)
        self.assertNotIn('\x00', text)
        self.assertIn('INFO start', text)
        self.assertIn('723/723', text)
        self.assertNotIn('361/723', text)                       # \r updates collapsed to the last state
        # grep sees every \r update and ignores colour codes; the cleaned match is shown
        r = self.call('job_log', {'job_id': '4242', 'grep': 'loss=0\\.91|INFO start'})
        self.assertIn('10/4858', r['text'])
        self.assertIn('INFO start', r['text'])
        self.assertNotIn('723/723', r['text'])
        self.assertIn('\x1b[32m', self.call('job_log', {'job_id': '4242', 'raw': True})['text'])
        r = self.call('job_log', {'job_id': '4242', 'progress': True})
        (path, bars), = r['progress'].items()
        self.assertEqual([(b['label'], b['n'], b['total'], b['kind']) for b in bars],
                         [('Loading weights', 723, 723, 'loading'), ('train', 12, 4858, 'work')])
        self.assertNotIn('text', r)
        r = self.call('job_log', {'job_id': '4242', 'hide_benign': True})
        self.assertNotIn('forcing job termination', r['text'])
        self.assertNotIn('TLS CA', r['text'])
        self.assertEqual(sum(r['benign_hidden'].values()), 2, r['benign_hidden'])
        self.assertIn('train', r['text'])

    def test_job_log_files_and_missing_logs(self):
        self.jobs(('4242', 'RUNNING', 30))
        r = self.call('job_log', {'job_id': '4242'})
        self.assertIn('node prolog', r['text'])                 # not "(not there)": normal in the first minutes
        d = self.home / 'data' / 'proj' / 'logs'
        (d / 'smoke-4242-r0.log').write_text('run 0 ok\n')
        (d / 'smoke-4242-r1.log').write_text('run 1 ok\n')
        r = self.call('job_log', {'job_id': '4242', 'file': 'logs/%x-%j-r*.log'})
        self.assertIn('run 0 ok', r['text'])
        self.assertIn('run 1 ok', r['text'])
        r = self.call('job_log', {'job_id': '4242', 'file': 'logs/none-%j-*.log'})
        self.assertIn('none-4242-*.log', r['text'])
        for bad in ('/etc/passwd', '../x', '~/x'):
            self.assertTrue(self.call_raw('job_log', {'job_id': '4242', 'file': bad}).is_error, bad)
        self.call('job_log', {'job_id': '4242', 'file': f'x$(touch {self.home}/p1)*', 'grep': f"'; touch {self.home}/p2"})
        self.call('job_log', {'job_id': '4242', 'file': f'`touch {self.home}/p3`', 'progress': True})
        self.assertEqual([p for p in ('p1', 'p2', 'p3') if (self.home / p).exists()], [])

    def test_job_details_diagnosis_limit_and_submit_line(self):
        self.jobs(('4242', 'FAILED', 3, '1:0', 30))
        self.write_log('4242', 'start\n/var/spool/slurmd/job4242/slurm_script: line 12: EESSI_X: unbound variable\n')
        (self.home / 'sacct_submit.txt').write_text(
            '4242|sbatch --export=ALL,HF_TOKEN=hf_secret123,RECORDS=data/r.jsonl --time=00:30:00 jobs/x.slurm\n')
        r = self.call('job_details', {'job_id': '4242'})
        self.assertEqual(r['rows'][0]['time_limit'], '00:30:00')
        self.assertIn('RECORDS=data/r.jsonl', r['submit_line'])
        self.assertNotIn('hf_secret123', r['submit_line'])
        self.assertEqual(r['diagnosis']['category'], 'script', r['diagnosis'])
        self.assertIn('set -u', r['diagnosis']['summary'])
        self.assertIn('unbound variable', r['logs'])
        # a clean COMPLETED job: no diagnosis
        self.jobs(('4242', 'COMPLETED', 300))
        self.write_log('4242', 'all done\n')
        self.assertNotIn('diagnosis', self.call('job_details', {'job_id': '4242'}))
        # COMPLETED, but the log shows a real error
        self.write_log('4242', 'Traceback (most recent call last):\n  File "c.py", line 3\nValueError: bad record\n')
        r = self.call('job_details', {'job_id': '4242'})
        self.assertIn('ValueError: bad record', r['diagnosis']['exceptions'][0])
        self.assertIn('COMPLETED, but', r['diagnosis']['note'])
        # pending: reason, start and dependency
        self.jobs(('4243', 'PENDING', 0))
        (self.home / 'squeue_dep.txt').write_text('4243|Dependency|N/A|afterok:4242(unfulfilled)\n')
        r = self.call('job_details', {'job_id': '4243'})
        self.assertEqual(r['pending'], [{'job_id': '4243', 'reason': 'Dependency', 'start': 'N/A',
                                         'dependency': 'afterok:4242(unfulfilled)'}])
        self.assertIn('pending', r['logs'])

    def test_wait_for_one_failed_job_diagnoses_it_once(self):
        self.jobs(('4242', 'NODE_FAIL', 63, '0:0'))
        r = self.call('wait_for_job', {'job_id': '4242', 'max_wait_s': 10})
        self.assertTrue(r['finished'])
        self.assertEqual(list(r['diagnosis']), ['4242'])
        self.assertEqual(r['diagnosis']['4242']['category'], 'infrastructure', r['diagnosis'])
        self.assertTrue(r['diagnosis']['4242']['findings'][0]['transient'])
        self.assertIn("Cancel, don't salvage", r['hint'])

    # ------------------------------------------------------------ check_job / submit_job: inputs on the cluster (29, 63)

    ASSETS = GOOD.replace('echo hi\n', 'MODEL=$DATA/models/M\nSIF=$DATA/containers/v.sif\n'
                                     'apptainer exec --nv $SIF python3 -c "print(1)" "$MODEL"\n')

    def test_check_and_submit_test_the_paths_a_job_reads(self):
        r = self.call('check_job', {'script_text': self.ASSETS, 'cwd': '$DATA/proj'})
        self.assertEqual(r['missing_paths'], ['$DATA/models/M', '$DATA/containers/v.sif'])
        self.assertFalse(r['ready'])
        self.assertTrue(any('submit_job refuses' in w for w in r['warnings']))
        self.assertNotIn('__amadeus', r['test_only']['output'])
        (self.home / 'data' / 'proj' / 'assets.slurm').write_text(self.ASSETS)
        r = self.call('submit_job', {'script': 'assets.slurm', 'cwd': '$DATA/proj', 'purpose': 'x'})
        self.assertFalse(r['submitted'])
        self.assertTrue(any('does not exist' in e or 'do not exist' in e for e in r['errors']), r)
        self.assertNotIn('--parsable', self.slurm_log())       # refused before sbatch, in the same SSH call
        # a job that fetches its inputs itself: allow_missing makes it a warning
        r = self.call('submit_job', {'script': 'assets.slurm', 'cwd': '$DATA/proj', 'purpose': 'x',
                                     'allow_missing': True})
        self.assertTrue(r['submitted'], r)
        self.assertTrue(any('allow_missing' in w for w in r['warnings']), r['warnings'])
        # with a dependency an earlier job may fetch them: a warning only
        r = self.call('check_job', {'script_text': self.ASSETS, 'cwd': '$DATA/proj',
                                    'sbatch_args': ['--dependency=afterok:1']})
        self.assertTrue(r['ready'], r)
        self.assertTrue(any('--dependency' in w for w in r['warnings']))
        (self.home / 'data' / 'models' / 'M').mkdir(parents=True)
        (self.home / 'data' / 'containers').mkdir()
        (self.home / 'data' / 'containers' / 'v.sif').write_text('')
        r = self.call('check_job', {'script_text': self.ASSETS, 'cwd': '$DATA/proj'})
        self.assertTrue(r['ready'], r)
        self.assertNotIn('missing_paths', r)
        r = self.call('submit_job', {'script': 'assets.slurm', 'cwd': '$DATA/proj', 'purpose': 'x'})
        self.assertTrue(r['submitted'], r)

    def test_check_job_cold_vllm_cache_on_dev(self):
        script = GOOD.replace('echo hi\n', 'export VLLM_CACHE_ROOT=$SCRATCH/cache-smoke/vllm\n'
                                         'vllm serve m --api-key k &\n')
        cold = __import__('amadeus.guard', fromlist=['COLD_START']).COLD_START
        r = self.call('check_job', {'script_text': script, 'cwd': '$DATA/proj'})
        self.assertIn(cold, r['warnings'])
        (self.home / 'scratch' / 'cache-smoke' / 'vllm').mkdir(parents=True)
        r = self.call('check_job', {'script_text': script, 'cwd': '$DATA/proj'})
        self.assertNotIn(cold, r['warnings'])

    def test_check_job_reports_asc_rewrites_and_expected_time(self):
        (self.home / 'asc.txt').write_text('sbatch: applying job settings for >> gpu_full_node <<\n'
                                           'sbatch: setting --cpus-per-task=22\n')
        r = self.call('check_job', {'script': 'job.slurm', 'cwd': '$DATA/proj', 'expected_minutes': 5,
                                    'sbatch_args': ['--cpus-per-task=8', '--qos=zen4_0768_h100x4', '--time=01:00:00']})
        self.assertEqual(r['asc']['mode'], 'gpu_full_node')
        self.assertEqual(r['asc']['changed'], ['--cpus-per-task 8 -> 22'])
        self.assertTrue(any('expected 5 min' in w for w in r['warnings']), r['warnings'])

    def test_heads_up_needs_the_node_count_named(self):
        args = {'script': 'job.slurm', 'cwd': '$DATA/proj', 'purpose': 'big',
                'sbatch_args': ['--qos=zen4_0768_h100x4', '-N', '16', '--time=00:20:00'],
                'approval': 'Alex: 500 credits for the pump study'}
        r = self.call('submit_job', args)
        self.assertFalse(r['submitted'])
        self.assertTrue(r['not_covered'][0].startswith('heads-up'))
        self.assertIn('naming the node count', r['note'])
        self.assertNotIn('--parsable', self.slurm_log())
        (self.home / 'scontrol.txt').write_text('JobId=4242 JobName=smoke JobState=PENDING Partition=zen4_0768_h100x4 '
                                                'QOS=zen4_0768_h100x4 NumNodes=16 TimeLimit=00:20:00\n')
        r = self.call('submit_job', {**args, 'approval': 'Alex: 16 nodes for 20 min are OK'})
        self.assertTrue(r['submitted'], r)

    def test_draft_job_new_parameters(self):
        r = self.call('draft_job', {'kind': 'gpu-node', 'name': 'pack', 'time': '30', 'dev': False,
                                    'runs': ['python a.py', 'python b.py'], 'run_minutes': [20, 22]})
        self.assertNotIn('error', r, r)
        self.assertIn('$RUN', r['script'])
        r = self.call('draft_job', {'kind': 'cpu', 'name': 'arr', 'time': '5', 'array_args': ['--seed 1', '--seed 2']})
        self.assertNotIn('error', r, r)
        self.assertIn('--array=0-1', r['script'])
        r = self.call('draft_job', {'kind': 'gpu-node', 'name': 'v', 'time': '30', 'dev': False,
                                    'vllm_preset': 'qwen38-fp8', 'load_cap_s': 600, 'command': 'echo client'})
        self.assertNotIn('error', r, r)
        self.assertIn('600', r['script'])
        self.assertIn('error', self.call('draft_job', {'kind': 'gpu-node', 'name': 'v', 'vllm_preset': 'nope'}))

    def test_wait_for_unknown_job_and_ssh_failure(self):
        r = self.call('wait_for_job', {'job_id': '999', 'max_wait_s': 40, 'poll_s': 5})
        self.assertIn('no job', r.get('error', ''))
        os.environ['AMADEUS_SSH'] = 'bash -c "echo Permission denied >&2; exit 255"'
        r = self.call('wait_for_job', {'job_id': '999', 'max_wait_s': 60, 'poll_s': 5})
        self.assertIn('cannot reach', r.get('error', ''))

    def test_wait_for_finished_job(self):
        self.jobs(('4242', 'COMPLETED', 600))
        self.write_log('4242', 'line1\ndone\n')
        r = self.call('wait_for_job', {'job_id': '4242', 'max_wait_s': 10, 'poll_s': 10})
        self.assertTrue(r['finished'])
        self.assertEqual(r['events'], ['ENDED 4242: COMPLETED (exit 0:0, 600 s)'])
        self.assertEqual(r['rows'][0]['state'], 'COMPLETED')          # one job: its details, as before
        self.assertIn('done', r['logs'])
        self.assertNotIn('hint', r)

    # ------------------------------------------------------------ certificate warnings (findings 0, 9, 36, 55, 65)

    def cert(self, minutes):
        from amadeus import auth
        p = self.home / 'cert'
        p.unlink(missing_ok=True) if minutes is None else p.write_text(str(minutes))
        auth.forget_certificate()

    def checks(self):
        p = self.home / 'auth-calls.log'
        return len(p.read_text().splitlines()) if p.exists() else 0

    def test_no_certificate_field_with_plenty_left(self):
        self.assertNotIn('certificate', self.call('ledger_entries'))
        self.assertNotIn('certificate_warning', self.call('list_jobs'))

    def test_every_dict_result_warns_under_60_min(self):
        self.cert(30)
        for name in ('ledger_entries', 'list_jobs', 'job_templates'):
            w = self.call(name).get('certificate', '')
            self.assertRegex(w, r'^expires \d\d:\d\d[+-]\d\d:\d\d \((29|30) min\): if you are answering a message from the human', name)
        a = self.call('access_status', {'probe': False})        # reports it itself, as a dict
        self.assertEqual(a['certificate']['minutes_left'], 30)

    def test_banner_once_and_error_sound(self):
        from amadeus import server, sound
        server._greeted = False
        first = self.call('access_status', {'probe': False})
        self.assertIn('MUSICA · ASC · Wien', first['banner'])
        self.assertNotIn('banner', self.call('access_status', {'probe': False}))
        with mock.patch.object(sound, 'play') as play:
            self.call('run_command', {'command': 'vim x'})       # refused: an `error` result
        play.assert_called_once_with('error')

    def test_missing_certificate_warns(self):
        self.cert(None)
        w = self.call('ledger_entries')['certificate']
        self.assertIn('expired or missing', w)
        self.assertIn('call start_login', w)
        self.assertIn('running jobs are not affected', w)

    def test_expiry_is_cached_and_reread_after_forget(self):
        self.cert(30)
        self.call('ledger_entries')
        n = self.checks()
        (self.home / 'cert').write_text('900')      # renewed outside: the cache still says 30 min
        self.assertIn('certificate', self.call('ledger_entries'))
        self.assertIn('certificate', self.call('job_templates'))
        self.assertEqual(self.checks(), n)          # no new --check within 5 min
        self.call('access_status', {'probe': False})  # a full check refreshes the cache
        self.assertNotIn('certificate', self.call('ledger_entries'))

    def test_failing_or_slow_check_never_breaks_a_tool(self):
        from amadeus import auth
        self.cert(30)
        os.environ['FAKE_CHECK_S'] = '5'
        with mock.patch.object(auth, 'CERT_QUICK_S', 0.5):
            t0 = time.monotonic()
            r = self.call('ledger_entries')
            self.assertLess(time.monotonic() - t0, 4)
        self.assertNotIn('certificate', r)          # unknown: no warning, no error
        os.environ['FAKE_CHECK_S'] = '0'
        with mock.patch.object(auth, 'expiry_warning', side_effect=RuntimeError('boom')):
            self.assertIn('entries', self.call('ledger_entries'))

    def test_list_jobs_warns_when_a_job_outlasts_the_certificate(self):
        self.cert(90)
        (self.home / 'squeue.txt').write_text('11|long|RUNNING|10:00|3:00:00|1|zen4_0768_h100x4|q|n1|N/A|x\n'
                                              '12|short|RUNNING|1:00|20:00|1|zen4_0768_h100x4|q|n1|N/A|x\n')
        r = self.call('list_jobs')
        self.assertNotIn('certificate', r)          # 90 min: no general warning yet
        w = r['certificate_warning']
        self.assertIn('11 (ends by', w)
        self.assertNotIn('12 (', w)
        self.assertIn('Running jobs are not affected', w)
        self.assertIn('early renewal (start_login)', w)

    def test_list_jobs_pending_start_estimate_counts(self):
        self.cert(120)
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        start = (datetime.now(ZoneInfo('Europe/Vienna')) + timedelta(minutes=100)).strftime('%Y-%m-%dT%H:%M:%S')
        (self.home / 'squeue.txt').write_text(f'13|p|PENDING|0:00|30:00|1|zen4_0768_h100x4|q|Priority|{start}|x\n')
        self.assertIn('13 (ends by', self.call('list_jobs')['certificate_warning'])
        (self.home / 'squeue.txt').write_text('13|p|PENDING|0:00|30:00|1|zen4_0768_h100x4|q|Priority|N/A|x\n')
        self.assertNotIn('certificate_warning', self.call('list_jobs'))

    def test_submit_warns_when_the_limit_ends_after_expiry(self):
        self.cert(5)
        r = self.call('submit_job', {'script': 'job.slurm', 'cwd': '$DATA/proj', 'purpose': 'p'})
        self.assertTrue(r['submitted'], r)
        self.assertIn(f'{r["job_id"]} (ends by', r['certificate_warning'])
        self.assertIn('(5 min)', r['certificate'])
        self.cert(900)
        r = self.call('submit_job', {'script': 'job.slurm', 'cwd': '$DATA/proj', 'purpose': 'p'})
        self.assertNotIn('certificate_warning', r)

    def test_wait_for_job_warns_from_the_recorded_limit(self):
        self.cert(90)
        self.jobs(('4242', 'RUNNING', 120))
        with open(self.tmp / 'jobs.jsonl', 'a') as f:
            f.write(json.dumps({'event': 'submit', 'job_id': '4242', 'purpose': 'p',
                                'estimate': {'time_limit_min': 180}}) + '\n')
        r = self.call('wait_for_job', {'job_id': '4242', 'until': 'start'})
        self.assertIn('4242 (ends by', r['certificate_warning'])
        self.jobs(('4242', 'COMPLETED', 120))      # ended: nothing left to land
        self.assertNotIn('certificate_warning', self.call('wait_for_job', {'job_id': '4242'}))

    def test_ssh_auth_failure_names_the_certificate(self):
        shim = self.tmp / 'denied.sh'
        shim.write_text('#!/bin/bash\necho "user1@musica: Permission denied (publickey,keyboard-interactive)." >&2\n'
                        'exit 255\n')
        os.environ['AMADEUS_SSH'] = f'bash {shim}'
        r = self.call('run_command', {'command': 'hostname'})
        self.assertIn('certificate expired or missing', r['note'])
        r = self.call_raw('list_jobs')
        self.assertTrue(r.is_error)
        self.assertIn('call start_login', r.content[0].text)
        shim.write_text('#!/bin/bash\necho "Connection timed out during banner exchange" >&2\nexit 255\n')
        r = self.call('run_command', {'command': 'hostname'})
        self.assertIn('tunnel stall', r['note'])

    # ------------------------------------------------------------ wait_for_job

    def jobs(self, *specs):
        """sacct rows (SACCT_FMT): (job_id, state, elapsed_s[, exit_code[, limit_min]]); started elapsed_s ago, logs in proj/logs."""
        wd = self.home / 'data' / 'proj'
        (wd / 'logs').mkdir(exist_ok=True)
        now = int(time.time())
        lines = []
        for spec in specs:
            jid, state, el = spec[:3]
            ec = spec[3] if len(spec) > 3 else '0:0'
            limit = spec[4] if len(spec) > 4 else 10
            start = 'Unknown' if state == 'PENDING' else str(now - el)
            lines.append(f'{jid}|{jid}|smoke|tester|zen4_0768_h100x4|dev|{state}|{ec}|{now - el - 5}|{start}|Unknown|'
                         f'{el if state != "PENDING" else 0}|1|n3010-014|gres/np_zen4_0768_h100x4=100|'
                         f'logs/%x-%j.out|logs/%x-%j.out|{wd}|{limit}')
        (self.home / 'sacct.txt').write_text('\n'.join(lines) + '\n')

    def write_log(self, jid, text, mode='w', age=0):
        p = self.home / 'data' / 'proj' / 'logs' / f'smoke-{jid}.out'
        with open(p, mode) as f:
            f.write(text)
        if age:
            t = time.time() - age
            os.utime(p, (t, t))
        return p

    def test_wait_failed_job_gets_no_outputs_alarm(self):
        self.jobs(('4243', 'FAILED', 90, '1:0'))
        self.write_log('4243', 'boom\n')
        r = self.call('wait_for_job', {'job_ids': ['4243'], 'expect': ['results/out-%j.txt'], 'max_wait_s': 60})
        self.assertIn('FAILED 4243: FAILED (exit 1:0, 90 s)', r['events'])
        self.assertFalse(any(e.startswith('OUTPUTS_MISSING') for e in r['events']))

    def test_wait_cancelled_job_has_no_salvage_hint(self):
        self.jobs(('4243', 'CANCELLED', 90, '0:15'))
        self.write_log('4243', 'working\n')
        r = self.call('wait_for_job', {'job_ids': ['4243'], 'until': 'any_fail', 'max_wait_s': 60})
        self.assertIn('CANCELLED 4243: CANCELLED (exit 0:15, 90 s)', r['events'])
        self.assertNotIn('hint', r)

    def test_wait_several_jobs_any_fail_with_diagnosis(self):
        self.jobs(('4242', 'RUNNING', 120), ('4243', 'FAILED', 90, '1:0'))
        self.write_log('4242', 'working\n')
        self.write_log('4243', 'Traceback (most recent call last):\n  File "x.py", line 1\n'
                               'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB\n')
        r = self.call('wait_for_job', {'job_ids': ['4242', '4243'], 'until': 'any_fail', 'max_wait_s': 60})
        self.assertFalse(r['finished'])
        self.assertLess(r['waited_s'], 10)
        self.assertIn('FAILED 4243: FAILED (exit 1:0, 90 s)', r['events'])
        self.assertIn('STARTED 4242: RUNNING', r['events'])
        self.assertIn('4243', r['diagnosis'])
        self.assertIn("Cancel, don't salvage", r['hint'])
        self.assertEqual({j['job_id']: j['state'] for j in r['jobs']}, {'4242': 'RUNNING', '4243': 'FAILED'})
        self.assertIn('working', '\n'.join(r['new_lines'].values()))
        # until='end' keeps waiting for the running job; the events it saw come back at the time limit
        r = self.call('wait_for_job', {'job_ids': ['4242', '4243'], 'max_wait_s': 10, 'poll_s': 5})
        self.assertFalse(r['finished'])
        self.assertGreaterEqual(r['waited_s'], 10)
        self.assertIn('call wait_for_job again', r['note'])

    def test_wait_one_ssh_command_per_poll(self):
        calls = self.home / 'ssh_calls'
        wrap = self.tmp / 'wrap.sh'
        wrap.write_text(f'echo x >> {calls}\nexec bash {HERE / "fake_ssh.sh"} "$1"\n')
        os.environ['AMADEUS_SSH'] = f'bash {wrap}'
        self.jobs(('4242', 'RUNNING', 60), ('4243', 'RUNNING', 60), ('4244', 'PENDING', 0))
        for j in ('4242', '4243'):
            self.write_log(j, 'x\n')
        r = self.call('wait_for_job', {'job_ids': ['4242', '4243', '4244'], 'max_wait_s': 10, 'poll_s': 5,
                                       'milestones': [{'pattern': 'never', 'within_s': 3600}]})
        self.assertFalse(r['finished'])
        # 3 polls, one catch-up probe when the running jobs' files first became known, one read of the new lines
        self.assertEqual(len(calls.read_text().splitlines()), 5)

    def test_wait_until_start_and_log_cursor(self):
        self.jobs(('4242', 'RUNNING', 30))
        self.write_log('4242', 'a\nloading 10%\rloading 50%\rloading 90%\nb\npart')
        r = self.call('wait_for_job', {'job_id': '4242', 'until': 'start'})
        self.assertFalse(r['finished'])
        self.assertEqual(r['events'], ['STARTED 4242: RUNNING'])
        (path, text), = r['new_lines'].items()
        self.assertEqual(text, 'a\nloading 90%\nb\npart')            # \r updates collapsed, partial line shown
        cur = r['next_cursor']
        self.assertEqual(cur['seen'], ['STARTED 4242'])
        self.write_log('4242', 'ial\nc\n', mode='a')
        self.jobs(('4242', 'COMPLETED', 40))
        r = self.call('wait_for_job', {'job_id': '4242', 'log_cursor': cur})
        self.assertTrue(r['finished'])
        self.assertEqual(r['events'], ['ENDED 4242: COMPLETED (exit 0:0, 40 s)'])
        self.assertEqual(r['new_lines'][path], 'partial\nc')           # only what is new since the cursor
        self.assertNotIn('logs', r)                                     # no second copy of the tail
        r = self.call('wait_for_job', {'job_id': '4242', 'log_cursor': r['next_cursor']})
        self.assertTrue(r['finished'])
        self.assertNotIn('events', r)
        self.assertNotIn('new_lines', r)

    def test_wait_milestones(self):
        self.jobs(('4242', 'RUNNING', 700))
        self.write_log('4242', 'staged 173G in 21s\nloading\n')
        ms = [{'pattern': r'staged \S+ in \d+s'}, {'pattern': r'ready after \d+s', 'within_s': 600}]
        r = self.call('wait_for_job', {'job_id': '4242', 'milestones': ms, 'max_wait_s': 60})
        self.assertFalse(r['finished'])
        self.assertIn(r'MILESTONE 4242: /staged \S+ in \d+s/ at line 1: staged 173G in 21s', r['events'])
        self.assertIn(r'LATE 4242: /ready after \d+s/ not in out after 700 s (due within 600 s)', r['events'])
        self.assertIn('cancel_job', r['hint'])
        self.assertEqual([m['matched'] for m in r['milestones']], [True, False])
        self.write_log('4242', 'server ready after 650s\n', mode='a')
        r = self.call('wait_for_job', {'job_id': '4242', 'milestones': ms, 'log_cursor': r['next_cursor'],
                                       'max_wait_s': 60})
        self.assertEqual(r['events'], [r'MILESTONE 4242: /ready after \d+s/ at line 3: server ready after 650s'])
        self.assertNotIn('hint', r)
        # a milestone in another file under the working directory
        (self.home / 'data' / 'proj' / 'logs' / 'server-4242.log').write_text('Application startup complete\n')
        r = self.call('wait_for_job', {'job_id': '4242', 'max_wait_s': 60, 'milestones': [
            {'pattern': 'startup complete', 'file': 'logs/server-%j.log'}]})
        self.assertIn('startup complete', r['events'][-1])

    def test_wait_recall_without_cursor_does_not_return_at_once(self):   # review: after a compaction
        self.jobs(('4242', 'RUNNING', 700))
        self.write_log('4242', 'staged 173G in 21s\n')
        ms = [{'pattern': r'staged \S+ in \d+s'}, {'pattern': r'ready after \d+s', 'within_s': 600}]
        r = self.call('wait_for_job', {'job_id': '4242', 'milestones': ms, 'max_wait_s': 60})
        self.assertLess(r['waited_s'], 10)                              # first call: the alarms, at once
        r = self.call('wait_for_job', {'job_id': '4242', 'milestones': ms, 'max_wait_s': 10, 'poll_s': 5})
        self.assertGreaterEqual(r['waited_s'], 10)                      # no cursor: no tight loop on old alarms
        self.assertTrue(any(e.startswith('MILESTONE') and e.endswith('(reported before)') for e in r['events']))
        self.write_log('4242', 'server ready after 650s\n', mode='a')    # a new alarm still returns at once
        r = self.call('wait_for_job', {'job_id': '4242', 'milestones': ms, 'max_wait_s': 60})
        self.assertLess(r['waited_s'], 10)
        self.assertIn(r'MILESTONE 4242: /ready after \d+s/ at line 2: server ready after 650s', r['events'])

    def test_wait_milestone_on_the_cleaned_log(self):                  # review: ANSI codes, \r updates
        self.jobs(('4242', 'RUNNING', 60))
        self.write_log('4242', 'x\n\x1b[32mINFO\x1b[0m: Application startup complete.\nloading 10%\rready 100%\n')
        r = self.call('wait_for_job', {'job_id': '4242', 'max_wait_s': 60, 'milestones': [
            {'pattern': '^INFO: Application startup'}, {'pattern': '^ready 100%'}]})
        self.assertEqual([m['matched'] for m in r['milestones']], [True, True], r)

    def test_wait_names_jobs_beyond_the_watched_rows(self):            # review: more than 8 jobs
        ids = [str(5000 + i) for i in range(10)]
        self.jobs(*[(j, 'RUNNING', 60) for j in ids])
        r = self.call('wait_for_job', {'job_ids': ids, 'max_wait_s': 10, 'poll_s': 5, 'quiet_alarm_s': 600})
        self.assertEqual(r['unwatched']['job_ids'], ids[8:])
        r = self.call('wait_for_job', {'job_ids': ids[:3], 'until': 'start'})
        self.assertNotIn('unwatched', r)

    def test_wait_quiet_alarm(self):
        self.jobs(('4242', 'RUNNING', 1000))
        self.write_log('4242', 'step 10\n', age=400)
        r = self.call('wait_for_job', {'job_id': '4242', 'quiet_alarm_s': 300, 'max_wait_s': 60})
        self.assertEqual(len(r['events']), 2, r['events'])
        self.assertRegex(r['events'][1], r'^QUIET 4\d\d s: 4242 is RUNNING')
        self.assertIn('cancel_job', r['hint'])
        # once seen, the same quiet period does not fire again; new output starts a new one
        cur = r['next_cursor']
        self.write_log('4242', 'step 11\n', mode='a', age=350)
        r = self.call('wait_for_job', {'job_id': '4242', 'quiet_alarm_s': 300, 'log_cursor': cur, 'max_wait_s': 60})
        self.assertRegex(r['events'][0], r'^QUIET 3\d\d s')
        self.assertEqual(r['new_lines'][next(iter(r['new_lines']))], 'step 11')

    def test_wait_expected_outputs(self):
        self.jobs(('4242', 'COMPLETED', 100))
        old = self.home / 'data' / 'proj' / 'results' / '4242'
        old.mkdir(parents=True)
        (old / 'stale.json').write_text('{}')
        os.utime(old / 'stale.json', (time.time() - 3600,) * 2)
        exp = ['results/%j/metrics.json', 'results/%j/*.json']
        r = self.call('wait_for_job', {'job_id': '4242', 'expect': exp})
        self.assertTrue(r['finished'])
        self.assertTrue(any(e.startswith('OUTPUTS_MISSING 4242: COMPLETED but missing') for e in r['events']))
        o = r['jobs'][0]['outputs']
        self.assertFalse(o['ok'])
        self.assertTrue(any('only older files' in m for m in o['missing']))
        (old / 'metrics.json').write_text('{"acc": 1}')
        self.jobs(('4242', 'TIMEOUT', 100))
        r = self.call('wait_for_job', {'job_id': '4242', 'expect': exp})
        o = r['jobs'][0]['outputs']
        self.assertTrue(o['ok'], o)
        self.assertIn('expected outputs are there', r['jobs'][0]['note'])
        self.assertEqual(r['events'], ['FAILED 4242: TIMEOUT (exit 0:0, 100 s)'])

    def test_wait_arguments(self):
        for args in ({}, {'job_id': '1', 'until': 'never'}, {'job_id': '1', 'expect': ['/etc/passwd']},
                     {'job_id': '1', 'expect': ['../x']}, {'job_id': '1', 'milestones': [{'pattern': '(('}]},
                     {'job_id': '1', 'milestones': [{'pattern': 'x', 'file': '/tmp/x'}]}):
            r = self.call_raw('wait_for_job', args)
            self.assertTrue(r.is_error or 'error' in json.loads(r.content[0].text), args)
        self.assertTrue(self.call_raw('wait_for_job', {'job_ids': ['1; ls']}).is_error)

    def test_wait_paths_and_patterns_are_never_shell_code(self):
        self.jobs(('4242', 'RUNNING', 30))
        self.write_log('4242', 'x\n')
        self.call('wait_for_job', {'job_id': '4242', 'max_wait_s': 10, 'poll_s': 5, 'expect': [f'x$(touch {self.home}/p1)'],
                                   'milestones': [{'pattern': f"'; touch {self.home}/p2; '"},
                                                  {'pattern': 'x', 'file': f'`touch {self.home}/p3`'}]})
        self.jobs(('4242', 'COMPLETED', 30))
        self.call('wait_for_job', {'job_id': '4242', 'expect': [f'x$(touch {self.home}/p1)*', 'a b/*']})
        self.assertEqual([p for p in ('p1', 'p2', 'p3') if (self.home / p).exists()], [])

    def test_push_and_fetch(self):
        src = self.tmp / 'repo'
        src.mkdir()
        subprocess.run(['git', 'init', '-q'], cwd=src, check=True)
        (src / 'a.py').write_text('print(1)\n')
        (src / '.gitignore').write_text('big/\n')
        (src / 'big').mkdir()
        (src / 'big' / 'x.bin').write_text('no')
        (src / 'new.txt').write_text('untracked but not ignored\n')
        subprocess.run(['git', 'add', 'a.py', '.gitignore'], cwd=src, check=True)
        subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'i'], cwd=src, check=True)
        r = self.call('push_files', {'local_dir': str(src), 'remote_dir': '$DATA/repo'})
        self.assertNotIn('error', r)
        dest = self.home / 'data' / 'repo'
        self.assertTrue((dest / 'a.py').exists() and (dest / 'new.txt').exists() and (dest / 'REVISION').exists())
        self.assertFalse((dest / 'big').exists())
        self.assertIn('error', self.call('push_files', {'local_dir': str(src), 'remote_dir': '~/.ssh'}))
        self.assertIn('error', self.call('push_files', {'local_dir': str(src), 'remote_dir': '/etc'}))
        r = self.call('fetch_files', {'remote_path': '$DATA/repo', 'local_dir': str(self.tmp / 'back')})
        self.assertTrue((self.tmp / 'back' / 'repo' / 'a.py').exists(), r)
        self.assertIn('error', self.call('fetch_files', {'remote_path': '$DATA/repo', 'local_dir': '/etc/x'}))
        self.assertIn('error', self.call('fetch_files', {'remote_path': '$DATA/nothing', 'local_dir': str(self.tmp)}))

    def test_push_warns_about_jobs_dirty_and_ignored_sources(self):
        src = self.tmp / 'repo'
        (src / 'pkg' / 'worlds').mkdir(parents=True)
        (src / 'worlds').mkdir()
        (src / 'results').mkdir()
        subprocess.run(['git', 'init', '-q'], cwd=src, check=True)
        (src / '.gitignore').write_text('worlds/\nresults/\n')
        (src / 'pkg' / 'core.py').write_text('x = 1\n')
        (src / 'pkg' / 'worlds' / 'gen.py').write_text('y = 2\n')        # caught by 'worlds/' by mistake
        (src / 'worlds' / 'pool.py').write_text('z\n')                    # the root pool: meant to be ignored
        (src / 'results' / 'r.py').write_text('z\n')
        subprocess.run(['git', 'add', '.'], cwd=src, check=True)
        subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'i'], cwd=src, check=True)
        data = self.home / 'data'
        (data / 'repo').mkdir()
        (data / 'other').mkdir()
        (self.home / 'squeue_wd.txt').write_text(
            f'11|PENDING|eval|{data}/repo|{data}/repo/jobs/eval.slurm\n'
            f'12|RUNNING|train|{data}/repo/sub|{data}/repo/sub/t.slurm\n'
            f'13|RUNNING|unrelated|{data}/other|{data}/other/x.slurm\n'
            f'14|PENDING|elsewhere|{data}/other|{data}/repo/jobs/y.slurm\n')   # only the script is inside
        r = self.call('push_files', {'local_dir': str(src), 'remote_dir': '$DATA/repo'})
        self.assertNotIn('error', r)
        self.assertEqual([j['job_id'] for j in r['jobs_using_target']], ['11', '12', '14'])
        self.assertTrue(any('pending jobs will run the pushed code' in w for w in r['warnings']), r)
        self.assertEqual(r['ignored_sources'], ['pkg/worlds/gen.py'])
        self.assertFalse(any('-dirty' in w for w in r['warnings']))
        (src / 'pkg' / 'core.py').write_text('x = 3\n')
        (self.home / 'squeue_wd.txt').unlink()
        r = self.call('push_files', {'local_dir': str(src), 'remote_dir': '$DATA/repo/pkg'})
        self.assertTrue(r['revision'].endswith('-dirty'))
        self.assertNotIn('jobs_using_target', r)
        self.assertTrue(any('commit first' in w for w in r['warnings']), r)
        # a job working one level above the target counts too
        (self.home / 'squeue_wd.txt').write_text(f'15|RUNNING|top|{data}/repo|{data}/repo/x.slurm\n')
        r = self.call('push_files', {'local_dir': str(src), 'remote_dir': '$DATA/repo/pkg'})
        self.assertEqual([j['job_id'] for j in r['jobs_using_target']], ['15'])


    def test_wrapped_tools_keep_their_schema(self):
        from mcp import Client
        from amadeus.server import mcp

        async def go():
            async with Client(mcp) as c:
                return {t.name: t.input_schema for t in (await c.list_tools()).tools}
        schemas = asyncio.run(go())
        self.assertNotIn('ctx', schemas['wait_for_job']['properties'])
        self.assertNotIn('required', schemas['wait_for_job'])          # job_id or job_ids
        self.assertEqual(schemas['submit_job']['properties']['sbatch_args'].get('anyOf', [{}])[0].get('type'), 'array')

    def test_report_issue(self):
        r = self.call('report_issue', {'title': 'fetch_files ignores max_mb', 'kind': 'bug', 'severity': 'low',
                                       'what_happened': 'used Bearer abcdefghijklmnop and https://auth.asc.ac.at/device?code=123456',
                                       'expected': 'refusal', 'project': 'pkg', 'tool_name': 'fetch_files'})
        self.assertTrue(r['filed'])
        text = Path(r['path']).read_text()
        self.assertIn('status: open', text)
        self.assertIn('project: pkg', text)
        self.assertNotIn('abcdefghijklmnop', text)
        self.assertNotIn('code=123456', text)
        listed = self.call('list_issues')['issues']
        self.assertEqual([i['id'] for i in listed], [r['id']])
        self.assertEqual(self.call('list_issues', {'status': 'fixed'})['issues'], [])

    def test_unexpected_errors_are_filed_once(self):
        (self.home / 'sacct.txt').write_text('4242|4242|smoke|tester|zen4_0768_h100x4|dev|COMPLETED|0:0|s|s|e|60|1|'
                                             'n1|x|||/tmp|10\n')
        for _ in range(2):
            with mock.patch('amadeus.server.S.credits_from_tres', side_effect=ValueError('boom')):
                r = self.call_raw('job_details', {'job_id': '4242', 'log_lines': 0})
            self.assertTrue(r.is_error)
            self.assertIn('filed automatically', r.content[0].text)
        files = list((self.tmp / 'issues').glob('*.md'))
        self.assertEqual(len(files), 1)
        text = files[0].read_text()
        self.assertIn('source: automatic', text)
        self.assertIn('occurrences: 2', text)
        self.assertIn('ValueError', text)

    def test_cluster_failures_are_not_filed(self):
        os.environ['AMADEUS_SSH'] = 'false'
        r = self.call_raw('job_details', {'job_id': '4242'})
        self.assertTrue(r.is_error)
        self.assertIn('sacct failed', r.content[0].text)
        self.assertFalse((self.tmp / 'issues').exists() and list((self.tmp / 'issues').glob('*.md')))


if __name__ == '__main__':
    unittest.main()
