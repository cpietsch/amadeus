"""The project config (amadeus/config.py): the file, env overrides, the refusal without an account, and what is built
from it (server instructions, drafts, guard messages, templates, budget)."""
import asyncio
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from amadeus import config, drafts, files, guard, ledger, templates

from . import TEST_CONFIG

OTHER = """
[project]
name = "other-project"
account = "p777777"
filesystem = "fs777777"
budget_tokens = 250
workdir = "$DATA/other"
template_dirs = ["@JOBS@"]

[limits]
approval_nodes = 2
approval_tokens = 5
headsup_nodes = 8

[files]
local_root = "@ROOT@"
hosts = ["box1", "box2"]
default_host = "box1"
"""


class Base(unittest.TestCase):
    ENV = ('AMADEUS_CONFIG', 'AMADEUS_ACCOUNT', 'AMADEUS_PROJECT', 'AMADEUS_FILESYSTEM', 'AMADEUS_BUDGET',
           'AMADEUS_WORKDIR', 'AMADEUS_TEMPLATE_DIRS', 'AMADEUS_LOCAL_ROOT', 'AMADEUS_HOSTS', 'AMADEUS_DEFAULT_HOST',
           'AMADEUS_LEDGER', 'XDG_STATE_HOME')

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.old = {k: os.environ.get(k) for k in self.ENV}

    def tearDown(self):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp)

    def use(self, text: str) -> Path:
        p = self.tmp / 'config.toml'
        (self.tmp / 'jobs').mkdir(exist_ok=True)
        p.write_text(text.replace('@JOBS@', str(self.tmp / 'jobs')).replace('@ROOT@', str(self.tmp)))
        os.environ['AMADEUS_CONFIG'] = str(p)
        return p


class Loading(Base):
    def test_test_config_is_in_use(self):
        self.assertEqual(config.path(), TEST_CONFIG)
        self.assertEqual((config.account(), config.filesystem(), config.budget()), ('p123456', 'fs123456', 4000.0))

    def test_file_values(self):
        self.use(OTHER)
        self.assertEqual((config.project(), config.account(), config.filesystem()),
                         ('other-project', 'p777777', 'fs777777'))
        self.assertEqual((config.budget(), config.workdir()), (250.0, '$DATA/other'))
        self.assertEqual(config.template_dirs(), [self.tmp / 'jobs'])
        self.assertEqual((config.approval_nodes(), config.approval_tokens(), config.headsup_nodes()), (2, 5.0, 8))
        self.assertEqual((config.local_root(), config.hosts(), config.default_host()),
                         (str(self.tmp), ['box1', 'box2'], 'box1'))

    def test_defaults_without_a_file(self):
        os.environ['AMADEUS_CONFIG'] = str(self.tmp / 'missing.toml')
        self.assertEqual((config.account(), config.project(), config.budget(), config.error()), ('', '', None, None))
        self.assertEqual((config.workdir(), config.local_root(), config.hosts()), ('$DATA', '~', []))
        self.assertEqual((config.approval_nodes(), config.approval_tokens(), config.headsup_nodes()), (4, 20.0, 16))
        self.assertEqual(config.template_dirs(), [])
        self.assertIsNone(config.notes())

    def test_reread_when_the_file_changes(self):
        p = self.use(OTHER)
        self.assertEqual(config.account(), 'p777777')
        time.sleep(0.01)
        p.write_text('[project]\naccount = "p888888"\n')
        self.assertEqual(config.account(), 'p888888')

    def test_broken_file(self):
        self.use('[project\naccount = ')
        self.assertEqual(config.account(), '')
        self.assertIn('config.toml', config.error())
        self.assertIn('config.toml', config.account_error())
        self.assertIn('could not be read', config.summary())

    def test_empty_env_counts_as_unset(self):
        self.use(OTHER)
        os.environ.update(AMADEUS_ACCOUNT='', AMADEUS_WORKDIR=' ', AMADEUS_BUDGET='', AMADEUS_LOCAL_ROOT='')
        self.assertEqual((config.account(), config.workdir(), config.budget()), ('p777777', '$DATA/other', 250.0))
        self.assertEqual(config.local_root(), str(self.tmp))

    def test_env_wins_over_the_file(self):
        self.use(OTHER)
        os.environ.update(AMADEUS_ACCOUNT='p111111', AMADEUS_PROJECT='env-project', AMADEUS_FILESYSTEM='fs111111',
                          AMADEUS_BUDGET='99', AMADEUS_WORKDIR='$SCRATCH/w', AMADEUS_HOSTS='h1, h2',
                          AMADEUS_DEFAULT_HOST='h2', AMADEUS_LOCAL_ROOT='/srv/x',
                          AMADEUS_TEMPLATE_DIRS=os.pathsep.join(['/a/jobs', '/b/jobs']))
        self.assertEqual((config.account(), config.project(), config.filesystem()),
                         ('p111111', 'env-project', 'fs111111'))
        self.assertEqual((config.budget(), config.workdir()), (99.0, '$SCRATCH/w'))
        self.assertEqual((config.hosts(), config.default_host(), config.local_root()), (['h1', 'h2'], 'h2', '/srv/x'))
        self.assertEqual(config.template_dirs(), [Path('/a/jobs'), Path('/b/jobs')])
        os.environ['AMADEUS_TEMPLATE_DIRS'] = ''          # set but empty: no extra directories
        self.assertEqual(config.template_dirs(), [])

    def test_ledger_default_and_override(self):
        os.environ.pop('AMADEUS_LEDGER')
        os.environ.pop('XDG_STATE_HOME', None)
        self.assertEqual(ledger.path(), Path.home() / '.local' / 'state' / 'amadeus' / 'jobs.jsonl')
        os.environ['XDG_STATE_HOME'] = str(self.tmp)
        self.assertEqual(ledger.path(), self.tmp / 'amadeus' / 'jobs.jsonl')
        os.environ['AMADEUS_LEDGER'] = str(self.tmp / 'l.jsonl')
        self.assertEqual(ledger.path(), self.tmp / 'l.jsonl')


class BuiltFromConfig(Base):
    GOOD = ('#!/bin/bash\n#SBATCH --partition=zen4_0768_h100x4\n#SBATCH --qos=dev_zen4_0768_h100x4\n'
            '#SBATCH --gres=gpu:4\n#SBATCH --time=00:10:00\necho hi\n')

    def test_instructions(self):
        from amadeus import server
        text = server.instructions()
        self.assertIn('project test-project-2026, Slurm account p123456, filesystem fs123456', text)
        self.assertIn('Budget: 4,000 tokens in total', text)
        self.assertNotIn('@', text.replace('amadeus://', ''))
        self.assertEqual(server.mcp.instructions, text)     # the test config was in place at import
        self.use(OTHER)
        text = server.instructions()
        self.assertIn('Slurm account p777777', text)
        self.assertIn('Budget: 250 tokens', text)
        self.assertIn('for 2 or more nodes at once', text)
        self.assertIn('more than 5 tokens', text)
        self.assertIn('except 8 or more nodes', text)
        for v in ('p123456', 'fs123456', 'nvidia-hackathon', '4,000', 'musica-hackathon'):
            self.assertNotIn(v, text)
        os.environ['AMADEUS_CONFIG'] = str(self.tmp / 'missing.toml')
        text = server.instructions()
        self.assertIn('No project is configured yet', text)
        self.assertIn('INSTALL.md', text)
        self.assertIn('ASC projects portal', text)

    def test_guard_uses_the_account_and_limits(self):
        self.use(OTHER)
        r = guard.evaluate(self.GOOD, [])
        self.assertIn('--account=p777777', r['inject'])
        r = guard.evaluate(self.GOOD.replace('#!/bin/bash\n', '#!/bin/bash\n#SBATCH --account=fs777777\n'), [])
        self.assertTrue(any('fs777777 is the filesystem' in e for e in r['errors']), r['errors'])
        r = guard.evaluate(self.GOOD.replace('dev_', '').replace('--gres', '--nodes=2\n#SBATCH --gres'), [])
        self.assertTrue(any('2 nodes' in a for a in r['needs_approval']), r['needs_approval'])

    def test_guard_without_account(self):
        os.environ['AMADEUS_CONFIG'] = str(self.tmp / 'missing.toml')
        r = guard.evaluate(self.GOOD, [])
        self.assertFalse(any(i.startswith('--account') for i in r['inject']))
        self.assertTrue(any('no Slurm account configured' in w for w in r['warnings']))

    def test_drafts(self):
        self.use(OTHER)
        d = drafts.draft(kind='cpu', name='t', command='echo')
        self.assertIn('#SBATCH --account=p777777', d['script'])
        os.environ['AMADEUS_CONFIG'] = str(self.tmp / 'missing.toml')
        d = drafts.draft(kind='cpu', name='t', command='echo')
        self.assertNotIn('--account', d['script'])
        self.assertTrue(any('No Slurm account is configured' in n for n in d['notes']))

    def test_files_filesystem_and_hosts(self):
        self.use(OTHER)
        self.assertEqual(files.normalize_remote('/data/fs777777/u/x'), ('/data/fs777777/u/x', None))
        self.assertIsNone(files.normalize_remote('/data/fs123456/u/x')[0])
        self.assertEqual(files.hosts(), ['', 'box1', 'box2'])
        self.assertEqual(files.default_host(), 'box1')

    def test_templates_from_config(self):
        self.use(OTHER)
        (self.tmp / 'jobs' / 'mine.slurm').write_text('#!/bin/bash\n# my job\n#SBATCH --time=5\necho\n')
        names = [e['name'] for e in templates.index()]
        self.assertIn(f'{self.tmp.name}/mine', names)
        self.assertIn(f'{templates.ROOT.name}/vllm_multinode', names)


class Refusal(Base):
    """Without an account check_job and submit_job refuse; read-only tools still work."""

    def call(self, name, args=None):
        import json

        from mcp import Client

        from amadeus.server import mcp

        async def go():
            async with Client(mcp) as c:
                r = await c.call_tool(name, args or {})
                return r.structured_content if r.structured_content is not None else json.loads(r.content[0].text)
        return asyncio.run(go())

    def test_refused_without_account(self):
        os.environ['AMADEUS_CONFIG'] = str(self.tmp / 'missing.toml')
        os.environ['AMADEUS_SSH'] = 'false'                 # nothing may reach a cluster
        try:
            for tool, args in (('check_job', {'script_text': BuiltFromConfig.GOOD}),
                               ('submit_job', {'script': 'job.slurm', 'purpose': 'p'})):
                r = self.call(tool, args)
                self.assertIn('no Slurm account configured', r['error'], tool)
                self.assertIn(str(self.tmp / 'missing.toml'), r['error'])
                self.assertIn('INSTALL.md', r['error'])
            self.assertIn('script', self.call('draft_job', {'kind': 'cpu', 'name': 't', 'command': 'echo'}))
            self.assertIn('templates', self.call('job_templates'))
        finally:
            os.environ.pop('AMADEUS_SSH', None)

    def test_project_notes_resource(self):
        from mcp import Client

        from amadeus.server import mcp
        notes = self.tmp / 'PROJECT.md'
        notes.write_text('# Our project\nbudget approvals\n')
        self.use(OTHER)
        p = self.tmp / 'config.toml'
        p.write_text(p.read_text().replace('[limits]', f'notes = "{notes}"\n\n[limits]'))

        async def go():
            async with Client(mcp) as c:
                return (await c.read_resource('amadeus://project')).contents[0].text
        self.assertIn('budget approvals', asyncio.run(go()))
        os.environ['AMADEUS_CONFIG'] = str(self.tmp / 'missing.toml')
        self.assertIn('No project notes configured', asyncio.run(go()))


if __name__ == '__main__':
    unittest.main()
