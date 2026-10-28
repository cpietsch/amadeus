import unittest

from amadeus import guard, remote, templates
from amadeus import slurm as S

GOOD = """#!/bin/bash
#SBATCH --job-name=t
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=dev_zen4_0768_h100x4   # dev first
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=00:10:00
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/sh
set -euo pipefail
echo hi
#SBATCH --time=99:00:00
"""
PROD = ['--qos=zen4_0768_h100x4']


def with_lines(*lines, base=GOOD):
    head, rest = base.split('source', 1)
    return head + '\n'.join(lines) + '\nsource' + rest


def approval(script=GOOD, args=(), in_use=None):
    return guard.evaluate(script, list(args), in_use)['needs_approval']


class Parsing(unittest.TestCase):
    def test_args(self):
        o, e = S.parse_args(['-N4', '-p', 'x', '--time=1:00:00', '--exclusive', '--qos', 'q', '-c=8'])
        self.assertEqual(e, [])
        self.assertEqual(o, {'nodes': '4', 'partition': 'x', 'time': '1:00:00', 'exclusive': '', 'qos': 'q',
                             'cpus-per-task': '8'})

    def test_optional_value_options_do_not_swallow_the_next(self):     # review finding 1
        o, e = S.parse_args(['--no-kill', '--nodes=8'])
        self.assertEqual((o['nodes'], o['no-kill']), ('8', ''))
        o, e = S.parse_args(['--no-kill', '-N', '8'])
        self.assertEqual(o['nodes'], '8')
        o, _ = S.script_options('#!/bin/bash\n#SBATCH --get-user-env\n#SBATCH --nodes=8\n')
        self.assertEqual(o['nodes'], '8')

    def test_prefixes_clusters_and_aliases(self):                       # findings 10, 17
        o, e = S.parse_args(['--arr=0-99', '--part=zen5_2304_b200x8', '--qo=zen5_2304_b200x8'])
        self.assertEqual((o['array'], o['partition'], o['qos']), ('0-99', S.B200, S.B200))
        o, e = S.parse_args(['-ON16'])
        self.assertEqual((o['overcommit'], o['nodes']), ('', '16'))
        o, e = S.parse_args(['-vN', '16'])
        self.assertEqual(o['nodes'], '16')
        o, e = S.parse_args(['--tasks-per-node=4'])
        self.assertEqual(o['ntasks-per-node'], '4')
        _, e = S.parse_args(['--n=4'])
        self.assertTrue(e and 'ambiguous' in e[0])
        _, e = S.parse_args(['--frobnicate=1'])
        self.assertTrue(e)

    def test_cli_tokens_must_be_options(self):                          # findings 17, 18
        _, e = S.parse_args(['big.slurm'], cli=True)
        self.assertTrue(e)
        _, e = S.parse_args(['-N 16'], cli=True)
        self.assertTrue(any('whitespace' in x for x in e))

    def test_script_options_as_sbatch_reads_them(self):                 # finding 11
        o, e = S.script_options(GOOD)
        self.assertEqual(o['time'], '00:10:00')           # the #SBATCH after `echo` is ignored, as by sbatch
        self.assertEqual(o['qos'], 'dev_zen4_0768_h100x4')
        o, _ = S.script_options('#!/bin/bash\n#SBATCH --nodes=8\n    #SBATCH --nodes=1\n')
        self.assertEqual(o['nodes'], '8')
        _, e = S.script_options('#!/bin/bash\n#PBS -l nodes=8\n#SBATCH -N 1\n')
        self.assertTrue(e)

    def test_minutes(self):
        for s, m in [('10', 10), ('10:30', 10.5), ('1:00:00', 60), ('2-00', 2880), ('1-02:30', 1590),
                     ('1-00:00:30', 1440.5)]:
            self.assertAlmostEqual(S.minutes(s), m)
        self.assertIsNone(S.minutes(None))
        self.assertEqual(S.minutes('0'), float('inf'))                 # finding 3: 0 means no limit
        self.assertEqual(S.minutes('00:00:00'), float('inf'))

    def test_node_count(self):                                          # finding 2
        self.assertEqual(S.node_count({'nodes': '2,16'}), 16)
        self.assertEqual(S.node_count({'nodes': '2-8'}), 8)
        self.assertEqual(S.node_count({'partition': S.GPU, 'gpus': '16'}), 4)
        self.assertEqual(S.node_count({'ntasks': '16', 'ntasks-per-node': '4'}), 4)
        self.assertEqual(S.node_count({'nodelist': 'n3010-[001-008]'}), 8)

    def test_hostlist(self):
        self.assertEqual(S.expand_hostlist('n3010-[019,027],n3012-034'), ['n3010-019', 'n3010-027', 'n3012-034'])
        self.assertEqual(S.expand_hostlist('n3010-[002-004]'), ['n3010-002', 'n3010-003', 'n3010-004'])

    def test_array(self):
        self.assertEqual(S.array_size('0-3'), (4, 4))
        self.assertEqual(S.array_size('0-9%2'), (10, 2))
        self.assertEqual(S.array_size('1,3,5-6'), (4, 4))
        self.assertEqual(S.array_size('0-10:5'), (3, 3))

    def test_estimate(self):
        e = S.estimate({'partition': S.GPU, 'nodes': '2', 'time': '1:00:00'})
        self.assertEqual(e['max_tokens'], 8.0)
        e = S.estimate({'partition': S.GPU, 'gres': 'gpu:1', 'time': '60'})
        self.assertEqual(e['max_tokens'], 1.0)
        e = S.estimate({'partition': S.GPU, 'gres': 'gpu:0', 'time': '60'})   # finding 6: not free
        self.assertEqual(e['max_tokens'], 4.0)
        e = S.estimate({'partition': S.CPU, 'nodes': '1', 'time': '30', 'array': '0-3%2'})
        self.assertEqual((e['max_tokens'], e['nodes_at_once']), (1.92, 2))

    def test_credits_from_tres(self):
        self.assertEqual(S.credits_from_tres(S.GPU, 3600, 'billing=1,gres/np_zen4_0768_h100x4=200,node=2'), 8.0)
        self.assertIsNone(S.credits_from_tres(S.B200, 3600, 'gres/np_zen5_2304_b200x8=100'))

    def test_parse_scontrol(self):
        j = S.parse_scontrol('JobId=5 JobName=a b JobState=PENDING Partition=zen4_0768_h100x4 NumNodes=2-8 '
                             'TimeLimit=01:00:00 TRES=cpu=88,node=1,gres/gpu=4 ArrayTaskId=0-9%2')
        self.assertEqual((j['JobName'], j['NumNodes'], j['TRES'], j['ArrayTaskId']),
                         ('a b', '2-8', 'cpu=88,node=1,gres/gpu=4', '0-9%2'))

    def test_remote_quote(self):
        self.assertEqual(remote.q('~/a b'), "~/'a b'")
        self.assertEqual(remote.q('$DATA/x'), '$DATA/x')
        self.assertEqual(remote.q('~'), '~')
        self.assertEqual(remote.q('/abs/p;rm'), "'/abs/p;rm'")


class Guard(unittest.TestCase):
    def test_good_job(self):
        r = guard.evaluate(GOOD, [])
        self.assertEqual(r['errors'], [])
        self.assertEqual(r['needs_approval'], [])
        self.assertIn('--account=p123456', r['inject'])
        self.assertTrue(any(i.startswith('--exclude=') for i in r['inject']))

    def test_missing_basics(self):
        r = guard.evaluate('#!/bin/bash\n#SBATCH --account=fs123456\necho\n', [])
        text = ' '.join(r['errors'])
        for word in ('account', 'partition', '--time'):
            self.assertIn(word, text)

    def test_no_time_limit(self):
        r = guard.evaluate(GOOD, ['--time=0'])
        self.assertTrue(any('no time limit' in e for e in r['errors']))

    def test_dev_limits(self):
        r = guard.evaluate(GOOD, ['--time=00:20:00', '-N', '3'])
        self.assertEqual(len([e for e in r['errors'] if 'dev QoS' in e]), 2)

    def test_qos_mismatch(self):
        r = guard.evaluate(GOOD, ['--qos=zen4_0768'])
        self.assertTrue(any('does not match' in e for e in r['errors']))

    def test_approval(self):
        self.assertTrue(any('4 nodes' in a for a in approval(args=PROD + ['-N', '4', '--time=01:00:00'])))
        self.assertTrue(any('tokens' in a for a in approval(args=PROD + ['-N', '2', '--time=03:00:00'])))
        self.assertTrue(any('at once' in a for a in approval(args=PROD + ['--time=00:30:00'],
                                                             in_use={'nodes': 3, 'tokens': 0})))
        self.assertEqual(approval(args=PROD + ['--time=00:30:00'], in_use={'nodes': 1, 'tokens': 0}), [])
        self.assertTrue(any('B200' in a for a in approval(args=['--partition=zen5_2304_b200x8',
                                                                '--qos=zen5_2304_b200x8'])))
        self.assertTrue(any('B200' in a for a in approval(args=['--part=zen5_2304_b200x8', '--qo=zen5_2304_b200x8'])))
        self.assertTrue(any('4 nodes' in a for a in approval(args=PROD + ['--gpus=16', '--time=00:30:00'])))

    def test_series_tokens_count_together(self):                       # finding 5
        a = approval(args=PROD + ['--time=04:00:00'], in_use={'nodes': 2, 'tokens': 16})
        self.assertTrue(any('committed' in x for x in a))

    def test_check_actual(self):
        job = {'Partition': S.GPU, 'QOS': S.GPU, 'NumNodes': '16', 'TimeLimit': '00:10:00'}
        self.assertTrue(guard.check_actual(job)['needs_approval'])
        job = {'Partition': S.GPU, 'QOS': S.GPU, 'NumNodes': '1', 'TimeLimit': '00:10:00', 'ArrayTaskId': '0-99%10'}
        self.assertTrue(any('at once' in a for a in guard.check_actual(job)['needs_approval']))
        job = {'Partition': S.GPU, 'QOS': 'dev_' + S.GPU, 'NumNodes': '1', 'TimeLimit': '00:10:00'}
        r = guard.check_actual(job)
        self.assertEqual((r['errors'], r['needs_approval']), ([], []))

    def test_cpu_limits(self):
        r = guard.evaluate(GOOD, ['--threads-per-core=1', '--cpus-per-task=190'])
        self.assertTrue(any('176' in e for e in r['errors']))
        r = guard.evaluate(GOOD, ['--threads-per-core=1', '--cpus-per-task=176'])
        self.assertEqual(r['errors'], [])
        cpu = GOOD.replace('zen4_0768_h100x4', 'zen4_0768').replace('#SBATCH --gres=gpu:4\n', '')
        r = guard.evaluate(cpu, ['--cpus-per-task=192'])             # finding 12: comfort mode
        self.assertTrue(any('comfort mode' in e for e in r['errors']))

    def test_comfort_mode_cpu_nodes(self):                              # sbatch --test-only, 2026-10-14
        cpu = GOOD.replace('zen4_0768_h100x4', 'zen4_0768').replace('#SBATCH --gres=gpu:4\n', '')
        r = guard.evaluate(cpu, ['--cpus-per-task=8', '--ntasks=1'])         # -N1 -n1 -c8: rejected by ASC
        self.assertTrue(any('whole nodes of 190 single-core tasks' in e for e in r['errors']), r['errors'])
        r = guard.evaluate(cpu, ['--ntasks=1', '--mem=4G'])                  # -N1 -n1: a whole node, billed in full
        self.assertTrue(any('billed in full' in w for w in r['warnings']), r['warnings'])
        self.assertEqual(guard.evaluate(cpu, ['--exclusive'])['warnings'], [])  # whole node on purpose
        no_nodes = cpu.replace('#SBATCH --nodes=1\n', '')
        self.assertEqual(guard.evaluate(no_nodes, ['--ntasks=1', '--mem=4G'])['warnings'], [])   # shared
        r = guard.evaluate(no_nodes, ['--cpus-per-task=8'])                  # comfort mode sets -c back to 1
        self.assertTrue(any('get one core' in e for e in r['errors']), r['errors'])
        vanilla = cpu.replace('#!/bin/bash\n', '#!/bin/bash\n#ASC --vanilla\n')
        r = guard.evaluate(vanilla, ['--cpus-per-task=8', '--threads-per-core=1'])
        self.assertEqual((r['errors'], r['warnings']), ([], []))

    def test_dev_qos_nodes_per_user(self):                              # QOSMaxNodePerUserLimit, 2026-10-14
        busy = {'nodes': 2, 'tokens': 0.1, 'dev_nodes': {'dev_zen4_0768_h100x4': 2}}
        self.assertTrue(any('QOSMaxNodePerUserLimit' in w for w in guard.evaluate(GOOD, [], busy)['warnings']))
        other = {'nodes': 2, 'tokens': 0.1, 'dev_nodes': {'dev_zen4_0768': 2}}     # the CPU dev QoS is separate
        self.assertEqual(guard.evaluate(GOOD, [], other)['warnings'], [])
        one = {'nodes': 1, 'tokens': 0.1, 'dev_nodes': {'dev_zen4_0768_h100x4': 1}}
        self.assertEqual(guard.evaluate(GOOD, [], one)['warnings'], [])

    def test_subshell_scope(self):                                      # draft_job's vLLM client
        script = with_lines('(', 'set -e', 'python client.py', ')', 'RC=$?', base=GOOD.replace('set -euo pipefail', 'set -uo pipefail'))
        self.assertEqual(guard.lint_script(script)[1], [])
        inside = with_lines('(', 'set -e', 'cmd', 'RC=$?', ')', base=GOOD.replace('set -euo pipefail', 'set -uo pipefail'))
        self.assertTrue(guard.lint_script(inside)[1])

    def test_nounset_tracking(self):                                    # finding 8
        self.assertTrue(any('Lmod' in e for e in guard.evaluate(with_lines('set -u'), [])['errors']))
        self.assertFalse(any('Lmod' in e for e in guard.evaluate(with_lines('set -u', 'set +u'), [])['errors']))
        self.assertTrue(any('Lmod' in e for e in guard.evaluate(with_lines('set -o nounset'), [])['errors']))
        shebang = GOOD.replace('#!/bin/bash\n', '#!/bin/bash -eu\n')
        self.assertTrue(any('Lmod' in e for e in guard.evaluate(shebang, [])['errors']))

    def test_errexit_retry_lint(self):                                  # finding 9
        warns = lambda body: [w for w in guard.evaluate(GOOD.replace('echo hi', body), [])['warnings'] if '$?' in w]
        self.assertTrue(warns('srun hostname\nRC=$?'))
        self.assertTrue(warns('srun hostname; RC=$?'))
        self.assertFalse(warns('srun hostname && RC=0 || RC=$?'))
        self.assertFalse(warns('set +e\nsrun hostname; RC=$?\nset -e'))
        self.assertFalse(warns("srun bash -c '\n  python x.py\n  rc=$?\n  exit $rc' && RC=0 || RC=$?"))

    def test_exclude_hostlist(self):                                    # finding 13
        r = guard.evaluate(GOOD, ['--exclude=n3010-[019,027],n3012-034'])
        self.assertFalse(any('bad nodes' in w for w in r['warnings']))
        r = guard.evaluate(GOOD, ['--exclude=n3010-019'])
        self.assertFalse(any(i.startswith('--exclude') for i in r['inject']))
        self.assertTrue(any('bad nodes' in w for w in r['warnings']))

    def test_pattern_log_dir_warning(self):                             # finding 44
        r = guard.evaluate(GOOD, ['--output=logs/%j/out.txt'])
        self.assertTrue(any('% pattern' in w for w in r['warnings']))


CPU = GOOD.replace('zen4_0768_h100x4', 'zen4_0768').replace('#SBATCH --gres=gpu:4\n', '')
VANILLA = '#!/bin/bash\n#ASC --vanilla\n'


def body(*lines, base=GOOD):
    """GOOD with `echo hi` replaced by these lines."""
    return base.replace('echo hi', '\n'.join(lines))


def has(items, word):
    return any(word in x for x in items)


class ScriptRules(unittest.TestCase):                                   # AGENTS.md §8
    def check(self, script, args=()):
        return guard.evaluate(script, list(args))

    def test_heads_up_for_16_nodes(self):                               # [3]
        a = approval(args=PROD + ['-N', '16', '--time=00:20:00'])
        heads = [x for x in a if x.startswith(guard.HEADS_UP)]
        self.assertEqual(len(heads), 1, a)
        for word in ('16 nodes', 'tokens', '00:20:00', '4-node run'):
            self.assertIn(word, heads[0])
        self.assertFalse(any(x.startswith(guard.HEADS_UP) for x in approval(args=PROD + ['-N', '8', '--time=00:20:00'])))
        arr = approval(args=PROD + ['-N', '4', '--array=0-3', '--time=00:20:00'])     # 4 tasks x 4 nodes at once
        self.assertTrue(any(x.startswith(guard.HEADS_UP) for x in arr))
        job = {'Partition': S.GPU, 'QOS': S.GPU, 'NumNodes': '16', 'TimeLimit': '00:10:00'}
        self.assertTrue(any(x.startswith(guard.HEADS_UP) for x in guard.check_actual(job)['needs_approval']))

    def test_heads_up_needs_the_node_count_in_the_approval(self):       # [3]
        a = approval(args=PROD + ['-N', '16', '--time=00:20:00'])
        self.assertEqual(guard.approval_gaps(a, ''), a)
        self.assertEqual(len(guard.approval_gaps(a, 'Alex: 500 credits for the pump study')), 1)
        self.assertEqual(len(guard.approval_gaps(a, 'Alex: the 16 credits are fine')), 1)
        for ok in ('Alex: yes, go with 16 nodes', 'Alex: OK for the 16-node run', 'Alex: -N16 approved',
                   'Alex: nodes=16 fine'):
            self.assertEqual(guard.approval_gaps(a, ok), [], ok)
        small = approval(args=PROD + ['-N', '4', '--time=00:20:00'])
        self.assertEqual(guard.approval_gaps(small, 'Alex: 500 credits for the pump study'), [])

    def test_budget_wording_is_not_a_node_count(self):                  # review: "16 node-hours"
        a = approval(args=PROD + ['-N', '16', '--time=00:20:00'])
        for budget in ('500 credits approved for the pump study, about 16 node-hours', 'Alex: 16 node hours max',
                       'go ahead with 8 nodes, budget 16 nodes x 1h max',
                       '~400-600 credits approved for the main study; ladder + heads-up before 16 nodes',
                       'approved budget of 16 node-hours', 'approve 4 nodes, not 16 nodes', '500 credits; up to 16 N'):
            self.assertEqual(len(guard.approval_gaps(a, budget)), 1, budget)
        for ok in ('Alex: 16 nodes', 'Alex: the 16-node run is fine', 'Alex: -N16 ok', 'Alex: go with 16 nodes, '
                   'about 20 min'):
            self.assertEqual(guard.approval_gaps(a, ok), [], ok)

    def test_cpu_ntasks_without_srun(self):                             # [12]
        no_nodes = CPU.replace('#SBATCH --nodes=1\n', '')
        r = self.check(no_nodes, ['-n', '32', '--mem=64G'])
        self.assertTrue(has(r['warnings'], '32 single-core tasks'), r['warnings'])
        self.assertFalse(has(self.check(body('srun ./worker', base=no_nodes), ['-n', '32'])['warnings'],
                             'single-core'))
        self.assertFalse(has(self.check(no_nodes, ['-n', '1'])['warnings'], 'single-core'))
        vanilla = no_nodes.replace('#!/bin/bash\n', VANILLA)
        self.assertFalse(has(self.check(vanilla, ['-n', '32'])['warnings'], 'single-core'))

    def test_asc_rewrites(self):                                         # [12]
        out = ('sbatch: applying job settings for >> shared partial cpu node <<\nsbatch: setting --ntasks-per-node=32\n'
               'sbatch: setting --cpus-per-task=1\nSubmitted batch job 4242\n')
        r = guard.asc_rewrites(out, {'cpus-per-task': '8', 'ntasks-per-node': '32'})
        self.assertEqual((r['mode'], r['accepted']), ('shared partial cpu node', True))
        self.assertEqual(r['settings'], {'ntasks-per-node': '32', 'cpus-per-task': '1'})
        self.assertEqual(r['changed'], ['--cpus-per-task 8 -> 1'])
        bad = ('sbatch: error: applying job settings for >> full cpu node(s) <<\nsbatch: error: setting --ntasks=190\n'
               'sbatch: error: enabled exclusive node access\nsbatch: error: allocation failure: Requested node '
               'configuration is not available\n')
        r = guard.asc_rewrites(bad)
        self.assertEqual((r['mode'], r['exclusive'], r['accepted'], r['settings']),
                         ('full cpu node(s)', True, False, {'ntasks': '190'}))
        self.assertIsNone(guard.asc_rewrites('')['accepted'])

    def test_variables_in_path_options(self):                           # [14]
        for args in (['--output=$DATA/logs/%x.out'], ['-e', '~/logs/x.err'], ['--chdir=${SCRATCH}/run']):
            self.assertTrue(has(self.check(GOOD, args)['errors'], 'expands neither'), args)
        script = GOOD.replace('#SBATCH --output=logs/%x-%j.out', '#SBATCH --output=$DATA/logs/%x-%j.out')
        self.assertTrue(has(self.check(script)['errors'], 'expands neither'))
        ok = self.check(GOOD, ['--output=/data/fs123456/user1/logs/%x-%j.out'])
        self.assertFalse(has(ok['errors'], 'expands neither'))

    def test_export_of_cluster_variables(self):                         # [34]
        r = self.check(GOOD, ['--export=ALL,ADAPTERS=a,DATA=data/infer-256'])
        self.assertTrue(has(r['errors'], '--export sets DATA'), r['errors'])
        script = GOOD.replace('#SBATCH --job-name=t', '#SBATCH --job-name=t\n#SBATCH --export=ALL,SLURM_FOO=1')
        self.assertTrue(has(self.check(script, ['--export=ALL,X=1'])['errors'], 'SLURM_FOO'))  # both sources count
        for ok in (['--export=ALL,SET=data/infer-256'], ['--export=ALL,HOME']):
            self.assertFalse(has(self.check(GOOD, ok)['errors'], '--export'), ok)
        self.assertTrue(has(self.check(body('DATA=data/infer-256', 'python x.py --data $DATA'))['warnings'],
                            'assigns DATA'))
        self.assertTrue(has(self.check(body('export SCRATCH=/tmp/x'))['warnings'], 'assigns SCRATCH'))
        for ok in ('export PATH=$DATA/envs/x/bin:$PATH', 'HOME=$SCRATCH apptainer exec x.sif true', 'DATA_DIR=x'):
            self.assertFalse(has(self.check(body(ok))['warnings'], 'assigns'), ok)

    def test_vllm_memory_utilisation(self):                             # [35]
        serve = 'apptainer exec --nv x.sif vllm serve m --host 127.0.0.1 --gpu-memory-utilization {} &'
        self.assertTrue(has(self.check(body(serve.format('0.98')))['errors'], 'refuses to start'))
        self.assertTrue(has(self.check(body(serve.format('0.99').replace(' 0.99', '=0.99')))['errors'], 'refuses to start'))
        self.assertFalse(has(self.check(body(serve.format('0.975')))['errors'], 'refuses to start'))

    def test_vllm_servers_per_node(self):                               # [35]
        loop = ['for g in 0 1 2 3; do', '  CUDA_VISIBLE_DEVICES=$g vllm serve m --host 127.0.0.1 --port $((8000+g)) &',
                'done', 'sleep 60']
        self.assertTrue(has(self.check(body(*loop))['warnings'], 'OMP_NUM_THREADS'))
        self.assertFalse(has(self.check(body('export OMP_NUM_THREADS=1', *loop))['warnings'], 'OMP_NUM_THREADS'))
        one = 'CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve m --host 127.0.0.1 --tensor-parallel-size 4 &'
        self.assertFalse(has(self.check(body(one))['warnings'], 'OMP_NUM_THREADS'))
        router = body('vllm-router --worker-urls a b --host 127.0.0.1 &')
        self.assertTrue(has(self.check(router)['warnings'], 'ulimit'))
        self.assertFalse(has(self.check(router.replace('vllm-router', 'ulimit -n 65536\nvllm-router'))['warnings'],
                             'ulimit'))

    def test_vllm_api_key_and_cold_start(self):                          # [54], [35]
        self.assertTrue(has(self.check(body('vllm serve m --port 8000 &'))['warnings'], '--api-key'))
        self.assertFalse(has(self.check(body('vllm serve m --host 0.0.0.0 --api-key "$KEY" &'))['warnings'],
                             '--api-key'))
        self.assertTrue(has(self.check(body('llama-server -m x --host 0.0.0.0 &'))['warnings'], '--api-key'))
        cold = body('vllm serve m --host 127.0.0.1 &')
        self.assertTrue(has(self.check(cold)['warnings'], 'dev QoS'))                    # GOOD is on the dev QoS
        self.assertFalse(has(self.check(cold, PROD)['warnings'], 'dev QoS'))
        warm = body('export VLLM_CACHE_ROOT=$SCRATCH/cache-$SLURM_JOB_NAME/vllm', 'vllm serve m --host 127.0.0.1 &')
        self.assertFalse(has(self.check(warm)['warnings'], 'dev QoS'))
        per_job = body('export VLLM_CACHE_ROOT=$SCRATCH/cache/$SLURM_JOB_ID', 'vllm serve m --host 127.0.0.1 &')
        self.assertTrue(has(self.check(per_job)['warnings'], 'dev QoS'))
        self.assertEqual(guard.vllm_cache_dirs(warm, 't'), ['$SCRATCH/cache-t/vllm'])
        self.assertEqual(guard.vllm_cache_dirs(per_job, 't'), [])

    def test_syntax_check(self):                                         # [41]
        if not guard._bash_ok():
            self.skipTest('no bash >= 4 here')
        self.assertTrue(has(self.check(body('if true; then', 'echo'))['errors'], 'bash -n'))
        self.assertFalse(has(self.check(GOOD)['errors'], 'bash -n'))

    def test_syntax_check_accepts_extglob(self):                        # review: extglob under bash -n
        if not guard._bash_ok():
            self.skipTest('no bash >= 4 here')
        ext = body('shopt -s extglob', 'rm -f out/!(keep).json', 'case ${1:-a} in @(a|b)) echo ab ;; esac')
        self.assertFalse(has(self.check(ext)['errors'], 'bash -n'))

    def test_arithmetic_shift_is_not_a_here_document(self):             # review: `$(( 1 << k ))`
        rest = ('hf download r --include a b', 'DATA=/x', 'vllm serve $DATA/models/missing')
        for shift in ('k=2; n=$(( 1 << k ))', '(( m = 1 << k ))'):
            r = self.check(body(shift, *rest))
            self.assertTrue(has(r['warnings'], 'one --include per pattern'), shift)
            self.assertTrue(has(r['warnings'], 'assigns DATA'), shift)
            self.assertIn('$DATA/models/missing', guard.referenced_paths(body(shift, 'vllm serve $DATA/models/missing')))
        heredoc = body('cat > x.py <<EOF', 'hf download r --include a b', 'EOF')
        self.assertFalse(has(self.check(heredoc)['warnings'], '--include'))

    def test_declare_without_init_under_set_u(self):                    # [41], [54]
        self.assertTrue(has(self.check(body('declare -A STATE SINCE'))['warnings'], 'unbound'))
        self.assertTrue(has(self.check(body('local -a X'))['warnings'], 'unbound'))
        self.assertFalse(has(self.check(body('declare -A STATE=() SINCE=()'))['warnings'], 'unbound'))
        no_u = GOOD.replace('set -euo pipefail', 'set -eo pipefail')
        self.assertFalse(has(self.check(body('declare -A STATE', base=no_u))['warnings'], 'unbound'))
        self.assertFalse(has(self.check(body('declare -p X'))['warnings'], 'unbound'))

    def test_module_before_lmod(self):                                   # [41]
        before = GOOD.replace('source /cvmfs', 'module load CUDA\nsource /cvmfs')
        self.assertTrue(has(self.check(before)['warnings'], 'Lmod init'))
        self.assertFalse(has(self.check(body('module load CUDA/12.6.0'))['warnings'], 'Lmod init'))
        no_init = GOOD.replace('source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/sh\n', '')
        self.assertTrue(has(self.check(body('ml CUDA', base=no_init))['warnings'], 'Lmod init'))
        eessi = body('source /cvmfs/software.eessi.io/versions/2025.06/init/bash', 'module load CUDA', base=no_init)
        self.assertFalse(has(self.check(eessi)['warnings'], 'Lmod init'))

    def test_hf_include(self):                                           # [41]
        bad = 'hf download org/m --include "*.json" "*.safetensors" --local-dir $DATA/models/m'
        self.assertTrue(has(self.check(body(bad))['warnings'], 'one --include per pattern'))
        good = 'hf download org/m --include "*.json" --include "*.safetensors" --local-dir $DATA/models/m > log 2>&1'
        self.assertFalse(has(self.check(body(good))['warnings'], '--include'))
        repo_after = 'hf download --include "*.json" Qwen/Qwen3-4B --local-dir $DATA/models/q'
        self.assertFalse(has(self.check(body(repo_after))['warnings'], '--include'))
        for bad in ('hf download --quiet org/m --include a b', 'hf download --revision main org/m --include a b'):
            self.assertTrue(has(self.check(body(bad))['warnings'], 'one --include per pattern'), bad)

    def test_apptainer_tmpdir(self):                                     # [41]
        pull = 'apptainer pull $DATA/containers/x.sif docker://x'
        self.assertTrue(has(self.check(body(pull))['warnings'], 'APPTAINER_TMPDIR'))
        self.assertTrue(has(self.check(body('export APPTAINER_TMPDIR=$SCRATCH/t', pull))['warnings'], 'APPTAINER_TMPDIR'))
        for ok in ('export APPTAINER_TMPDIR=/local/$USER/apptainer_tmp',
                   'export APPTAINER_TMPDIR=$([ -d /local ] && echo /local/$USER/t || echo $SCRATCH/t)'):
            self.assertFalse(has(self.check(body(ok, pull))['warnings'], 'APPTAINER_TMPDIR'), ok)

    def test_multi_node_launch(self):                                    # [41]
        one = body('srun --ntasks-per-node=1 python x.py')
        self.assertTrue(has(self.check(one, PROD + ['-N', '4'])['warnings'], 'no retry'))
        self.assertFalse(has(self.check(one, PROD + ['-N', '2'])['warnings'], 'no retry'))
        retry = body('for attempt in 1 2 3; do', '  srun --ntasks-per-node=1 python x.py && RC=0 || RC=$?',
                     '  (( RC == 0 )) && break; sleep 30', 'done')
        self.assertFalse(has(self.check(retry, PROD + ['-N', '4'])['warnings'], 'no retry'))
        burst = body('for n in $NODES; do', '  srun -N1 -n1 -w $n serve.sh &', 'done', 'wait')
        self.assertTrue(has(self.check(burst)['warnings'], 'without a pause'))
        paced = body('for n in $NODES; do', '  srun -N1 -n1 -w $n serve.sh &', '  sleep 0.5', 'done', 'wait')
        self.assertFalse(has(self.check(paced)['warnings'], 'without a pause'))

    def test_waits_and_caches(self):                                     # [54]
        bare = body('vllm serve m --host 127.0.0.1 &', 'python client.py &', 'wait')
        self.assertTrue(has(self.check(bare)['warnings'], 'bare `wait`'))
        stopped = body('vllm serve m --host 127.0.0.1 &', 'SERVER=$!', 'stop() { kill $SERVER; }', 'python c.py',
                       'stop', 'build a &', 'build b &', 'wait')
        self.assertFalse(has(self.check(stopped)['warnings'], 'bare `wait`'))
        self.assertFalse(has(self.check(body('python a.py &', 'python b.py &', 'wait'))['warnings'], 'bare `wait`'))
        loop = ['for i in 1 2 3; do', '  curl -sf http://$HOST:8000/health && break', '  sleep 5', 'done']
        self.assertTrue(has(self.check(body(*loop))['warnings'], 'max-time'))
        self.assertFalse(has(self.check(body(*[l.replace('-sf', '-sf -m 5') for l in loop]))['warnings'], 'max-time'))
        self.assertFalse(has(self.check(body(*[l.replace('$HOST', '127.0.0.1') for l in loop]))['warnings'],
                             'max-time'))
        self.assertFalse(has(self.check(body('curl -s http://x/y'))['warnings'], 'max-time'))
        self.assertTrue(has(self.check(body('export DG_JIT_CACHE_DIR=$SCRATCH/dg'))['warnings'], 'DG_CACHE_DIR'))
        self.assertTrue(has(self.check(body('export HF_HOME=$HOME/.cache/hf'))['warnings'], 'under $HOME'))
        self.assertTrue(has(self.check(body('export TRITON_CACHE_DIR=~/.triton'))['warnings'], 'under $HOME'))
        self.assertFalse(has(self.check(body('export HF_HOME=$DATA/hf_cache'))['warnings'], 'under $HOME'))
        shared = ['export TRITON_CACHE_DIR=$SCRATCH/triton', 'for g in 0 1 2 3; do',
                  '  CUDA_VISIBLE_DEVICES=$g python train.py &', 'done', 'wait']
        self.assertTrue(has(self.check(body(*shared))['warnings'], 'share one TRITON_CACHE_DIR'))
        own = shared[:2] + ['  TRITON_CACHE_DIR=$SCRATCH/triton/g$g CUDA_VISIBLE_DEVICES=$g python train.py &'] + shared[3:]
        self.assertFalse(has(self.check(body(*own))['warnings'], 'share one'))

    def test_gpu_comfort_mode(self):                                     # [69], [79]
        r = self.check(GOOD, ['--ntasks-per-node=1', '--cpus-per-task=176', '--threads-per-core=1'])
        self.assertTrue(has(r['warnings'], '4 tasks x 22 CPUs'), r['warnings'])
        self.assertFalse(has(r['errors'], 'comfort'))
        self.assertTrue(has(self.check(body('srun python x.py'))['warnings'], '4 copies per node'))
        self.assertFalse(has(self.check(body('srun -n 1 python x.py'))['warnings'], '4 copies'))
        vanilla = GOOD.replace('#!/bin/bash\n', VANILLA)
        r = self.check(body('srun python x.py', base=vanilla), ['--ntasks-per-node=1', '--cpus-per-task=352', '--mem=0'])
        self.assertFalse(has(r['warnings'] + r['errors'], 'comfort mode'), r)
        self.assertTrue(has(self.check(GOOD, ['--mem=100G'])['errors'], 'memory via gpu count'))
        one = GOOD.replace('--gres=gpu:4', '--gres=gpu:1')
        self.assertTrue(has(self.check(one)['errors'], 'Cannot specify number of GPUs'))
        self.assertFalse(has(self.check(one.replace('#SBATCH --nodes=1\n', ''))['errors'], 'Cannot specify'))

    def test_time_far_above_the_expected_runtime(self):                 # [45]
        self.assertTrue(guard.time_vs_expected('01:00:00', 20))
        self.assertIsNone(guard.time_vs_expected('00:35:00', 20))
        self.assertIsNone(guard.time_vs_expected('00:19:00', 4))         # expected + 15 min is always fine
        stated = GOOD.replace('#SBATCH --job-name=t\n', '#SBATCH --job-name=t\n# ~35-40 min expected\n')
        self.assertEqual(guard.stated_minutes(stated), 40)
        self.assertTrue(has(self.check(stated, PROD + ['--time=03:00:00'])['warnings'], 'expected 40 min'))
        self.assertFalse(has(self.check(stated, PROD + ['--time=01:15:00'])['warnings'], 'expected'))
        self.assertEqual(guard.stated_minutes(GOOD.replace('\n#SBATCH --job-name', '\n# Expected runtime: 2 h\n'
                                                                                    '#SBATCH --job-name')), 120)
        self.assertIsNone(guard.stated_minutes(GOOD.replace('echo hi', '# expected 5 min')))   # body comments don't
        r = guard.evaluate(GOOD, PROD + ['--time=02:00:00'], expected_minutes=30)
        self.assertTrue(has(r['warnings'], 'expected 30 min'))

    def test_b200_cores_and_array_size(self):                           # [100]
        b200 = GOOD.replace('zen4_0768_h100x4', 'zen5_2304_b200x8').replace('dev_zen5_2304_b200x8', 'zen5_2304_b200x8')
        r = self.check(b200, ['--threads-per-core=1', '--cpus-per-task=190'])
        self.assertTrue(has(r['errors'], '176'), r['errors'])
        self.assertEqual(S.array_max_index('0-99999%10'), 99999)
        self.assertEqual(S.array_max_index('1,5,200:7'), 200)
        self.assertTrue(has(self.check(GOOD, ['--array=0-100000'])['errors'], 'MaxArraySize'))
        self.assertFalse(has(self.check(GOOD, ['--array=0-99999%4'])['errors'], 'MaxArraySize'))


class ReferencedPaths(unittest.TestCase):
    def test_assets_not_outputs(self):
        script = body('SIF=${SIF:-$DATA/containers/vllm.sif}', 'MODEL=$DATA/models/Qwen3.5-4B',
                      'PY=$DATA/envs/train/bin/python', 'OUT=$DATA/results/run-$SLURM_JOB_ID', 'mkdir -p "$OUT" logs',
                      'export HF_HOME=$DATA/hf_cache TRITON_CACHE_DIR=$SCRATCH/cache-x/triton',
                      'STAGE=/local/$USER/m', 'cp -r "$MODEL" "$STAGE"',
                      'apptainer exec --nv -B "$DATA,$SCRATCH,/local" "$SIF" "$PY" x.py --data /data/fs123456/user1/'
                      'sets/a.jsonl --out $DATA/results/x > $DATA/logs/x.log',
                      'for g in 0 1; do CUDA_VISIBLE_DEVICES=$g "$PY" y.py --model $DATA/models/m-$g & done; wait')
        self.assertEqual(guard.referenced_paths(script),
                         ['$DATA/containers/vllm.sif', '$DATA/models/Qwen3.5-4B', '$DATA/envs/train/bin/python',
                          '/data/fs123456/user1/sets/a.jsonl'])

    def test_created_by_the_script(self):
        script = body('VENV=$DATA/envs/hf', 'MODEL_DIR=$DATA/models/X', 'SIF=$DATA/containers/x.sif',
                      '[ -x "$VENV/bin/hf" ] || python3 -m venv "$VENV"', '"$VENV/bin/hf" download r --local-dir "$MODEL_DIR"',
                      'apptainer pull "$SIF.tmp" docker://x && mv "$SIF.tmp" "$SIF"', 'ls "$MODEL_DIR" "$SIF"')
        self.assertEqual(guard.referenced_paths(script), [])

    def test_fetched_then_used(self):                                   # review: wget -O, conda -p, git clone
        for lines in (('wget -q -O $DATA/models/q.gguf https://x/q.gguf', 'llama-server -m $DATA/models/q.gguf'),
                      ('conda env create -p $DATA/envs/new -f env.yml', '$DATA/envs/new/bin/python run.py'),
                      ('git clone https://x/y.git $DATA/src/y', 'source $DATA/src/y/venv/bin/activate'),
                      ('echo started | tee -a $DATA/data/run.log',)):
            self.assertEqual(guard.referenced_paths(body(*lines)), [], lines)
        for lines in (('rsync -a $DATA/models/src/ /local/x/',), ('cp -a $DATA/models/src /local/x',)):
            self.assertEqual(guard.referenced_paths(body(*lines)), ['$DATA/models/src'], lines)

    def test_staged_vllm_model_is_still_probed(self):                   # review: M=$MODEL, then mkdir "$M"
        from amadeus import drafts
        for stage in (True, False):
            s = drafts.draft(kind='gpu-node', name='t', vllm_model='$DATA/models/typo', stage_model=stage)['script']
            self.assertIn('$DATA/models/typo', guard.referenced_paths(s), stage)
        moved = body('M=$DATA/models/x', 'M=/local/m', 'mkdir -p "$M"', 'vllm serve $DATA/models/x')
        self.assertEqual(guard.referenced_paths(moved), ['$DATA/models/x'])

    def test_templates(self):
        for d in templates.dirs():
            for f in d.glob('*.slurm'):
                for p in guard.referenced_paths(f.read_text()):
                    self.assertRegex(p, r'^(\$DATA|\$SCRATCH|\$HOME|/data/|/scratch/|/home/|~/)', f.name)
                    self.assertNotIn('$SLURM', p)


class TemplatesStayClean(unittest.TestCase):
    """No false positives: every tested job script (job_templates) passes the rules without errors."""

    def test_no_errors_on_tested_scripts(self):
        n = 0
        for d in templates.dirs():
            for f in sorted(d.glob('*.slurm')):
                r = guard.evaluate(f.read_text(), [])
                self.assertEqual(r['errors'], [], f.name)
                n += 1
        self.assertGreater(n, 0)


class CommandPolicy(unittest.TestCase):                                 # findings 0, 21
    def refused(self, c):
        return guard.command_policy(c)['refuse']

    def approval(self, c):
        return guard.command_policy(c)['needs_approval']

    def test_slurm_changes_refused(self):
        for c in ('sbatch job.slurm', 'cd x && sbatch -N 16 j', 'FOO=1 sbatch j', 'nohup scancel 5 &',
                  'scontrol update JobId=1 TimeLimit=72:00:00', 'echo $(sbatch j)', "bash -c 'scancel 7'",
                  'ssh n3010-001 scancel 7', 'if sbatch j; then echo; fi', 'x\nsrun hostname'):
            self.assertTrue(self.refused(c), c)
        for c in ('squeue -u $USER', 'scontrol show job 5', 'sacct -j 5', 'ls $DATA', 'echo sbatch'):
            self.assertFalse(self.refused(c), c)

    def test_sacctmgr_read_only_allowed(self):
        for c in ('sacctmgr show qos', 'sacctmgr -n show qos format=Name,MaxWall', 'sacctmgr list assoc user=$USER',
                  'sacctmgr -n show user $USER', 'sacctmgr -P --noheader list qos', 'sacctmgr --parsable2 sho qos',
                  'sacctmgr -i show qos', 'sacctmgr show -n -P assoc', 'sacctmgr --readonly LIS assoc',
                  'sacctmgr show reservation', 'sacctmgr show r', 'sacctmgr list resource',
                  'sacctmgr -s show user', 'sacctmgr -h', 'sacctmgr --version', 'sacctmgr help',
                  'timeout 30 sacctmgr -n show qos | grep dev', 'env LC_ALL=C sacctmgr -p list qos',
                  "bash -c 'sacctmgr -n show qos'", 'squeue -u $USER && sacctmgr show assoc'):
            self.assertFalse(self.refused(c), c)

    def test_sacctmgr_changes_refused(self):
        for c in ('sacctmgr', 'sacctmgr -i', 'sacctmgr -n', 'sacctmgr add user x account=y',
                  'sacctmgr create account x', 'sacctmgr modify user x set qos=long',
                  'sacctmgr update qos dev set MaxWall=1-0', 'sacctmgr -i delete user x', 'sacctmgr remove user x',
                  'sacctmgr archive dump', 'sacctmgr load file=x.cfg', 'sacctmgr dump musica file=x.cfg',
                  'sacctmgr reconfigure', 'sacctmgr shutdown', 'sacctmgr sh qos', 'sacctmgr clear stats',
                  'sacctmgr show runawayjobs', 'sacctmgr -i list runaway',
                  "echo 'delete user x' | sacctmgr -i", 'timeout 30 sacctmgr -i delete user x',
                  "bash -c 'sacctmgr modify user x set qos=long'", 'sacctmgr show qos; sacctmgr delete qos x',
                  'squeue && sacctmgr add qos x', 'env sacctmgr -i remove user x', 'echo $(sacctmgr -i del user x)'):
            r = self.refused(c)
            self.assertTrue(r, c)
            self.assertIn('the human (account administration)', r, c)

    def test_sacctmgr_runaway_jobs_refused_in_every_spelling(self):
        # Slurm's _show_it: RunawayJobs abbreviated to 2+ letters, its aliases OrphanJobs and LostJobs to 1+; getopt
        # takes options after the command word too, and `--` ends them
        for c in ('sacctmgr -i show lostjobs', 'sacctmgr -i list orphanjobs', 'sacctmgr -i show ru',
                  'sacctmgr -i show -n runawayjobs', 'sacctmgr show -i -P RunawayJ', 'sacctmgr list O',
                  'sacctmgr show l', 'sacctmgr show -- runawayjobs', 'sacctmgr -i sho lost'):
            self.assertTrue(self.refused(c), c)

    def test_access_config_refused(self):
        for c in ('cat ~/.ssh/id_ed25519', 'echo x >> ~/.bashrc', 'cp k $HOME/.ssh/authorized_keys'):
            self.assertTrue(self.refused(c), c)

    def test_destructive_needs_approval(self):
        for c in ('rm -rf $SCRATCH/x', 'mv a b', 'find $DATA -name x -delete', 'rsync -a --delete a/ b/',
                  'git clean -fdx', 'cd x && rm y', 'timeout 10s rm -rf $SCRATCH/x', 'bash -lc "rm -rf x"',
                  'nice -n 5 rm x', 'env -u X rm y', 'xargs -n 1 rm < list'):
            self.assertTrue(self.approval(c), c)
        for c in ('du -sh $DATA/*', 'git status', 'tar czf - x | cat > /dev/null'):
            self.assertFalse(self.approval(c), c)


if __name__ == '__main__':
    unittest.main()
