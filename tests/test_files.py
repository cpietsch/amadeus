"""push/fetch against a fake cluster (tests/fake_ssh.sh) and a fake client machine (tests/fake_host.sh)."""
import shlex
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from amadeus import files

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def git(cwd, *args):
    subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t', *args], cwd=cwd, check=True,
                   capture_output=True)


def make_repo(path: Path, content: dict[str, str | bytes]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git(path, 'init', '-q')
    for name, text in content.items():
        p = path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text if isinstance(text, bytes) else text.encode())
    git(path, 'add', '-A')
    git(path, 'commit', '-qm', 'init')
    return path


class Env(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cluster = self.tmp / 'cluster'          # the fake cluster's $HOME; $DATA is cluster/data
        (self.cluster / 'data').mkdir(parents=True)
        (self.cluster / 'scratch').mkdir()
        ssh = self.cluster / '.ssh'
        ssh.mkdir()
        (ssh / 'authorized_keys').write_text('ssh-ed25519 ORIGINAL\n')
        (ssh / 'id_ed25519_musica').write_text('PRIVATE KEY\n')
        self.local = self.tmp / 'local'              # AMADEUS_LOCAL_ROOT on the server's machine
        self.local.mkdir()
        self.gpubox = self.tmp / 'gpubox'            # the fake workstation's $HOME
        self.gpubox.mkdir()
        self.count = self.tmp / 'count'
        self.env = {'AMADEUS_SSH': f'bash {HERE / "fake_ssh.sh"}', 'FAKE_HOME': str(self.cluster),
                    'AMADEUS_RETRY_DELAY': '0', 'AMADEUS_LOCAL_ROOT': str(self.local),
                    'AMADEUS_HOSTS': 'gpubox, -oProxyCommand=evil', 'FAKE_HOST_HOME': str(self.gpubox),
                    'AMADEUS_HOST_SSH_GPUBOX': f'bash {HERE / "fake_host.sh"}', 'AMADEUS_DEFAULT_HOST': None,
                    'FAKE_HOST_TAR_FAIL': None, 'FLAKY_MATCH': None, 'FLAKY_MSG': None, 'FLAKY_FAILS': None,
                    'FLAKY_PASS_BYTES': None, 'FLAKY_COUNT': str(self.count)}
        self.old = {k: os.environ.get(k) for k in [*self.env, 'HOME']}
        self.setenv(**self.env)

    def setenv(self, **kv):
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def tearDown(self):
        self.setenv(**self.old)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def flaky(self, match, msg, fails=1000, pass_bytes=None):
        self.setenv(AMADEUS_SSH=f'bash {HERE / "flaky_ssh.sh"}', FLAKY_MATCH=match, FLAKY_MSG=msg,
                    FLAKY_FAILS=str(fails), FLAKY_PASS_BYTES=None if pass_bytes is None else str(pass_bytes))

    def keys_untouched(self):
        self.assertEqual((self.cluster / '.ssh' / 'authorized_keys').read_text(), 'ssh-ed25519 ORIGINAL\n')


class RemotePaths(unittest.TestCase):
    """Findings 7 and 16: the static check normalises before it decides."""

    def test_hidden_and_home_paths_refused(self):
        for p in ['~//.ssh', '~/./.ssh', '~/./.bashrc', '$HOME//.ssh/authorized_keys', '~/./.bashrc.d',
                  '~/./.config/wireguard', '~/.', '~', '~/', '$HOME', '${HOME}/./', '~/.local/bin', '~/.step',
                  '~/proj/../.ssh', '$DATA/../x', '$DATA/a/..', '/etc', '$DATAX/x', '/data/fs123456/../../etc',
                  '~/proj/.git/hooks', 'relative/path', '', '$DATA/a\nb', '/data/fs123456']:
            self.assertIsNotNone(files.check_remote(p), p)

    def test_normalised(self):
        for p, want in [('$DATA/x', '$DATA/x'), ('${DATA}/x/', '$DATA/x'), ('$SCRATCH//a/./b/', '$SCRATCH/a/b'),
                        ('~/musica-hackathon', '~/musica-hackathon'), ('$HOME//proj', '$HOME/proj'),
                        ('/data//fs123456/user1/./t', '/data/fs123456/user1/t'), ('$DATA', '$DATA'),
                        ('$DATA/.cache', '$DATA/.cache')]:
            self.assertEqual(files.normalize_remote(p), (want, None), p)


class Excludes(unittest.TestCase):
    """Finding 41: one matching rule for both modes."""

    def test_patterns(self):
        pats = files._patterns(['data/raw/', './build', '/top/', '*.ckpt'])
        self.assertEqual(pats, ['data/raw', 'build', 'top', '*.ckpt'])
        for rel in ['data/raw/huge.bin', 'build/x/y.o', 'top/a', 'a/b/m.ckpt']:
            self.assertTrue(files._excluded(rel, pats), rel)
        for rel in ['data/raw2/x', 'mydata/raw/x', 'src/top.py', 'data/keep.txt']:
            self.assertFalse(files._excluded(rel, pats), rel)
        self.assertTrue(files._excluded('src/raw/x.py', ['raw']))
        self.assertTrue(files._excluded('data/raw/x', ['data/raw/*']))


class Push(Env):
    def push(self, src, dest, **kw):
        return files.push(str(src), dest, **kw)

    def test_excludes_same_in_both_modes(self):
        src = make_repo(self.local / 'repo', {'a.py': 'a\n', 'data/keep.txt': 'k\n'})
        (src / 'data' / 'raw').mkdir()
        (src / 'data' / 'raw' / 'huge.bin').write_text('big\n')          # untracked, not ignored
        for i, pat in enumerate(['data/raw', 'data/raw/', 'raw/', 'data/raw/*', './data/raw']):
            for mode in ('tracked', 'all'):
                dest = f'$DATA/ex{i}{mode}'
                r = self.push(src, dest, mode=mode, exclude=[pat])
                self.assertNotIn('error', r, r)
                d = self.cluster / 'data' / f'ex{i}{mode}'
                self.assertTrue((d / 'data' / 'keep.txt').exists(), (pat, mode))
                self.assertFalse((d / 'data' / 'raw').exists(), (pat, mode))

    def test_all_mode_default_excludes(self):
        src = self.local / 'plain'
        for n in ['a.txt', '.git/config', 'sub/__pycache__/x.pyc', 'img.sif', '.venv/bin/python', 'sub/b.txt']:
            (src / n).parent.mkdir(parents=True, exist_ok=True)
            (src / n).write_text(n)
        r = self.push(src, '$DATA/plain', mode='all')
        self.assertNotIn('error', r, r)
        self.assertEqual(r['files'], 2)
        d = self.cluster / 'data' / 'plain'
        self.assertEqual(sorted(str(p.relative_to(d)) for p in d.rglob('*') if p.is_file()), ['a.txt', 'sub/b.txt'])

    def test_secrets_never_pushed(self):
        """Finding 15 (3): key files and .env files stay home in both modes."""
        src = make_repo(self.local / 'repo', {'a.py': 'a\n'})
        for n in ['.env', 'conf/id_ed25519', 'tls/server.key', 'tls/server.pem']:
            (src / n).parent.mkdir(parents=True, exist_ok=True)
            (src / n).write_text('SECRET\n')
        for mode in ('tracked', 'all'):
            r = self.push(src, f'$DATA/s{mode}', mode=mode)
            self.assertNotIn('error', r, r)
            self.assertEqual(r['skipped_secrets'], ['.env', 'conf/id_ed25519', 'tls/server.key', 'tls/server.pem'])
            d = self.cluster / 'data' / f's{mode}'
            self.assertEqual(sorted(p.name for p in d.rglob('*') if p.is_file()),
                             ['REVISION', 'a.py'] if mode == 'tracked' else ['a.py'])

    def test_revision_dirty_with_untracked_files(self):
        src = make_repo(self.local / 'repo', {'a.py': 'a\n', '.gitignore': 'ignored/\n'})
        rev = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=src, capture_output=True,
                             text=True).stdout.strip()
        (src / 'ignored').mkdir()
        (src / 'ignored' / 'x').write_text('x')
        r = self.push(src, '$DATA/repo')
        self.assertEqual(r['revision'], rev)                                  # clean: ignored files don't count
        self.assertEqual((self.cluster / 'data' / 'repo' / 'REVISION').read_text(), rev + '\n')
        (src / 'new_module.py').write_text('x = 1\n')                          # finding 40
        r = self.push(src, '$DATA/repo')
        self.assertEqual(r['revision'], rev + '-dirty')
        self.assertTrue((self.cluster / 'data' / 'repo' / 'new_module.py').exists())
        self.assertEqual((self.cluster / 'data' / 'repo' / 'REVISION').read_text(), rev + '-dirty\n')
        r = self.push(src, '$DATA/repo', exclude=['new_module.py'])
        self.assertEqual(r['revision'], rev)                                  # excluded, so not pushed
        (src / 'a.py').write_text('changed\n')
        self.assertEqual(self.push(src, '$DATA/repo', exclude=['new_module.py'])['revision'], rev + '-dirty')

    def test_failed_push_leaves_remote_tree_unchanged(self):
        """Finding 39: a connection that drops mid-upload changes nothing; the staging directory is removed."""
        dest = self.cluster / 'data' / 'repo'
        dest.mkdir()
        dest.chmod(0o750)
        (dest / 'a_model.py').write_text('old model\n')
        (dest / 'REVISION').write_text('oldrev\n')
        src = make_repo(self.local / 'repo', {'a_model.py': os.urandom(600_000), 'b.py': 'b\n'})
        self.flaky('tar xzf', 'client_loop: send disconnect: Broken pipe', pass_bytes=200_000)
        r = self.push(src, '$DATA/repo')
        self.assertIn('error', r)
        self.assertEqual(r['attempts'], 3)
        self.assertEqual((dest / 'a_model.py').read_text(), 'old model\n')
        self.assertEqual((dest / 'REVISION').read_text(), 'oldrev\n')
        self.assertFalse((dest / 'b.py').exists())
        self.assertEqual([p.name for p in dest.iterdir() if p.name.startswith('.push.')], [])
        self.setenv(AMADEUS_SSH=f'bash {HERE / "fake_ssh.sh"}')
        r = self.push(src, '$DATA/repo')
        self.assertNotIn('error', r, r)
        self.assertEqual((dest / 'a_model.py').read_bytes(), (src / 'a_model.py').read_bytes())
        self.assertEqual((dest / 'REVISION').read_text(), r['revision'] + '\n')
        self.assertEqual(stat.S_IMODE(dest.stat().st_mode), 0o750)            # staging dir's 0700 not copied over
        self.assertEqual([p.name for p in dest.iterdir() if p.name.startswith('.push.')], [])
        self.assertEqual(r['remote_path'], str(dest))

    def test_failed_producer_commits_nothing(self):
        """A source that fails after writing a complete archive (e.g. a file vanished) must not land."""
        make_repo(self.gpubox / 'proj', {'a.py': 'new\n'})
        dest = self.cluster / 'data' / 'proj'
        dest.mkdir()
        (dest / 'a.py').write_text('old\n')
        self.setenv(FAKE_HOST_TAR_FAIL='1')
        r = files.push('~/proj', '$DATA/proj', host='gpubox')
        self.assertIn('error', r)
        self.assertEqual(r['attempts'], 1)
        self.assertEqual((dest / 'a.py').read_text(), 'old\n')
        self.assertFalse((dest / 'REVISION').exists())

    def test_retries_only_transient_ssh_failures(self):
        """Finding 45: 'Permission denied' fails at once; transient failures get 3 attempts, 2 sleeps."""
        src = make_repo(self.local / 'repo', {'a.py': 'a\n'})
        with mock.patch('amadeus.files.time.sleep') as sleep:
            self.flaky('tar xzf', 'user1@musica: Permission denied (publickey,keyboard-interactive).')
            r = self.push(src, '$DATA/repo')
            self.assertEqual((r['attempts'], sleep.call_count), (1, 0), r)
            self.assertIn('Permission denied', r['error'])
            self.assertEqual(r['exit_codes'][1], 255)
            self.count.unlink()
            self.flaky('tar xzf', 'Connection timed out during banner exchange')
            r = self.push(src, '$DATA/repo')
            self.assertEqual((r['attempts'], sleep.call_count), (3, 2), r)  # no sleep after the last attempt
            self.count.unlink()
            sleep.reset_mock()
            self.flaky('tar xzf', 'Connection closed by 193.170.79.21 port 22', fails=1)
            r = self.push(src, '$DATA/repo')
            self.assertNotIn('error', r, r)
            self.assertEqual((r['attempts'], sleep.call_count), (2, 1))

    def test_dot_paths_and_symlinks_refused_on_the_cluster(self):
        """Findings 7 and 16: '~/./.ssh' is refused before SSH; a symlink into ~/.ssh is refused on the cluster."""
        src = self.local / 'evil'
        src.mkdir()
        (src / 'authorized_keys').write_text('ssh-ed25519 ATTACKER\n')
        for dest in ['~/./.ssh', '~//.ssh', '$HOME/./.ssh', '~/.']:
            self.assertIn('error', self.push(src, dest, mode='all'), dest)
        (self.cluster / 'data' / 'link').symlink_to(self.cluster / '.ssh')
        (self.cluster / 'innocent').symlink_to(self.cluster / '.ssh')
        for dest in ['$DATA/link', '~/innocent', '$DATA/link/sub']:
            r = self.push(src, dest, mode='all')
            self.assertIn('not below', r.get('error', ''), (dest, r))
        self.keys_untouched()
        self.assertEqual(sorted(p.name for p in (self.cluster / '.ssh').iterdir()),
                         ['authorized_keys', 'id_ed25519_musica'])
        (self.cluster / 'data' / 'home').symlink_to(self.cluster)        # $DATA/home -> $HOME itself
        self.assertIn('error', self.push(src, '$DATA/home', mode='all'))
        self.assertFalse((self.cluster / 'authorized_keys').exists())

    def test_normalised_destination(self):
        src = make_repo(self.local / 'repo', {'a.py': 'a\n'})
        r = self.push(src, '$DATA//proj/./x/')
        self.assertEqual(r['remote_dir'], '$DATA/proj/x')
        self.assertTrue((self.cluster / 'data' / 'proj' / 'x' / 'a.py').exists())
        self.assertIn('error', self.push(src, '$DATA'))
        self.assertIn('error', self.push(src, '$DATA/x', mode='everything'))

    def test_push_limit(self):
        src = make_repo(self.local / 'repo', {'a.bin': os.urandom(300_000)})
        with mock.patch.object(files, 'MAX_PUSH_MB', 0.1):
            r = self.push(src, '$DATA/repo')
        self.assertIn('limit', r['error'])
        self.assertFalse((self.cluster / 'data' / 'repo').exists())

    def test_relay_stops_at_the_limit(self):
        out = self.tmp / 'out'
        r = files._relay(['head', '-c', '3000000', '/dev/urandom'], ['bash', '-c', f'cat > {out}'], 30,
                         limit=1_000_000)
        self.assertTrue(r.over)
        self.assertLessEqual(out.stat().st_size, 1_000_000)


class LocalSide(Env):
    """Findings 15 and 22: the local root, hidden directories, the server's own files, other hosts."""

    def put(self, rel, text='x\n'):
        p = self.cluster / 'data' / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def test_default_root_is_home(self):
        home = self.tmp / 'devhome'
        (home / '.config' / 'wireguard').mkdir(parents=True)
        (home / '.config' / 'wireguard' / 'asc.conf').write_text('PrivateKey = SECRET\n')
        (home / '.bashrc').write_text('# original\n')
        (home / 'proj' / '.git').mkdir(parents=True)
        self.setenv(HOME=str(home), AMADEUS_LOCAL_ROOT=None)
        self.put('evil/.bashrc', 'touch /tmp/PWNED\n')
        self.put('pkg/a.txt')
        self.put('hooks/post-commit', 'evil\n')
        self.assertIn('error', files.push('~/.config/wireguard', '$DATA/loot', mode='all'))
        self.assertIn('error', files.push(str(home / '.config' / 'wireguard'), '$DATA/loot', mode='all'))
        self.assertFalse((self.cluster / 'data' / 'loot').exists())
        self.assertIn('error', files.fetch('$DATA/evil/.bashrc', '~'))
        self.assertIn('error', files.fetch('$DATA/pkg', '~/../'))
        self.assertIn('error', files.fetch('$DATA/hooks', '~/proj/.git'))
        self.assertIn('error', files.fetch('$DATA/pkg', 'proj'))                     # relative: refused
        self.assertEqual((home / '.bashrc').read_text(), '# original\n')
        self.assertFalse((home / 'pkg').exists())
        self.assertFalse((home / 'proj' / '.git' / 'hooks').exists())
        r = files.fetch('$DATA/evil', '~')                                         # hidden files are skipped
        self.assertEqual(r.get('skipped_hidden'), 1, r)
        self.assertFalse((home / 'evil' / '.bashrc').exists())
        self.assertEqual((home / '.bashrc').read_text(), '# original\n')
        r = files.fetch('$DATA/pkg', '~')
        self.assertEqual(r.get('local_path'), str(home / 'pkg'), r)
        self.assertTrue((home / 'pkg' / 'a.txt').exists())

    def test_server_files_protected(self):
        self.setenv(AMADEUS_LOCAL_ROOT=str(REPO.parent))
        for name in ['amadeus', 'state', 'scripts', REPO.name]:
            self.put(f'{name}/zz-test-marker.txt')
        self.put('AGENTS.md/zz-test-marker.txt')
        try:
            for remote_path, local_dir in [('$DATA/amadeus', REPO), ('$DATA/state', REPO), ('$DATA/x', REPO / 'state'),
                                           ('$DATA/scripts', REPO), (f'$DATA/{REPO.name}', REPO.parent),
                                           ('$DATA/x', REPO / 'amadeus' / 'sub'), ('$DATA/AGENTS.md', REPO)]:
                r = files.fetch(remote_path, str(local_dir))
                self.assertIn("server's own", r.get('error', ''), (remote_path, local_dir, r))
        finally:
            for p in REPO.rglob('zz-test-marker.txt'):
                p.unlink()
        self.assertEqual(list(REPO.rglob('zz-test-marker.txt')), [])

    def test_hidden_entries_skipped_and_escaping_links_refused(self):
        self.put('res/a.txt')
        self.put('res/.git/hooks/post-checkout', 'evil\n')
        self.put('res/.envrc', 'evil\n')
        r = files.fetch('$DATA/res', str(self.local / 'out'))
        self.assertNotIn('error', r, r)
        out = self.local / 'out' / 'res'
        self.assertTrue((out / 'a.txt').exists())
        self.assertFalse((out / '.git').exists() or (out / '.envrc').exists())
        self.assertEqual(r['skipped_hidden'], 2)
        outside = self.tmp / 'outside'
        outside.mkdir()
        (out / 'sub').symlink_to(outside)
        r = files.fetch('$DATA/res', str(self.local / 'out'))
        self.assertIn('links to a directory outside', r.get('error', ''), r)
        self.assertIn('error', files.fetch('$DATA/res/.git', str(self.local / 'out')))

    def test_push_from_other_host(self):
        make_repo(self.gpubox / 'proj', {'a.py': 'on gpubox\n'})
        make_repo(self.local / 'proj', {'a.py': 'on the server\n'})
        r = files.push('~/proj', '$DATA/proj', host='gpubox')
        self.assertNotIn('error', r, r)
        self.assertEqual((r['host'], r['local_dir']), ('gpubox', str(self.gpubox / 'proj')))
        self.assertEqual((self.cluster / 'data' / 'proj' / 'a.py').read_text(), 'on gpubox\n')
        r = files.push(str(self.local / 'proj'), '$DATA/proj2')
        self.assertEqual((r['host'], (self.cluster / 'data' / 'proj2' / 'a.py').read_text()), ('', 'on the server\n'))
        self.setenv(AMADEUS_DEFAULT_HOST='gpubox')
        self.assertEqual(files.default_host(), 'gpubox')
        r = files.push('~/proj', '$DATA/proj3')
        self.assertEqual(r['host'], 'gpubox')
        self.assertEqual((self.cluster / 'data' / 'proj3' / 'a.py').read_text(), 'on gpubox\n')
        # the server's AMADEUS_LOCAL_ROOT does not apply on gpubox: its root is the home directory there
        self.assertIn('outside', files.push(str(self.local / 'proj'), '$DATA/x', host='gpubox')['error'])

    def test_fetch_to_other_host(self):
        self.put('res/r.json', '{}\n')
        r = files.fetch('$DATA/res', '~/results', host='gpubox')
        self.assertNotIn('error', r, r)
        self.assertEqual((r['host'], r['local_path']), ('gpubox', str(self.gpubox / 'results' / 'res')))
        self.assertTrue((self.gpubox / 'results' / 'res' / 'r.json').exists())
        self.assertFalse((self.local / 'res').exists())
        self.assertIn('error', files.fetch('$DATA/res', '~/.ssh', host='gpubox'))

    def test_hosts(self):
        self.assertEqual(files.hosts(), ['', 'gpubox'])                     # '-oProxyCommand=evil' is dropped
        for h in ['evil', '-oProxyCommand=evil', 'GPUBOX']:
            r = files.push(str(self.local), '$DATA/x', host=h)
            self.assertIn('unknown host', r['error'])
            self.assertEqual(r['host'], h)
        self.assertIn('unknown host', files.fetch('$DATA/x', str(self.local), host='evil')['error'])
        self.assertEqual(files._host_argv('gpubox', 'echo hi')[-1], 'bash -c ' + shlex.quote(files.GNU_SHIM + 'echo hi'))
        self.setenv(AMADEUS_HOST_SSH_GPUBOX=None)
        self.assertEqual(files._host_argv('gpubox', 'x')[:4], ['ssh', '-o', 'BatchMode=yes', 'gpubox'])


class Fetch(Env):
    def test_symlinked_directory_and_file(self):
        """Finding 43: a symlink is fetched as its target's contents, under the requested name."""
        mh = self.cluster / 'data' / 'mh'
        mh.mkdir()
        (mh / 'big.txt').write_text('y' * 100_000)
        (mh / 'rel').symlink_to('big.txt')
        (self.cluster / 'musica-hackathon').symlink_to(mh)
        (self.cluster / 'data' / 'ckpt-1000.pt').write_text('weights\n')
        (self.cluster / 'data' / 'latest.ckpt').symlink_to('ckpt-1000.pt')
        r = files.fetch('~/musica-hackathon', str(self.local))
        self.assertNotIn('error', r, r)
        out = self.local / 'musica-hackathon'
        self.assertFalse(out.is_symlink())
        self.assertEqual((out / 'big.txt').read_text(), 'y' * 100_000)
        self.assertEqual(os.readlink(out / 'rel'), 'big.txt')               # links inside stay links
        self.assertGreaterEqual(r['megabytes'], 0.1)
        self.assertEqual(r['remote_path'], str(mh))
        r = files.fetch('$DATA/latest.ckpt', str(self.local))
        self.assertNotIn('error', r, r)
        self.assertFalse((self.local / 'latest.ckpt').is_symlink())
        self.assertEqual((self.local / 'latest.ckpt').read_text(), 'weights\n')

    def test_odd_names(self):
        odd = "run 1&2,\\x'y"
        (self.cluster / 'data' / odd).mkdir()
        (self.cluster / 'data' / odd / 'f g').write_text('ok')
        r = files.fetch(f'$DATA/{odd}', str(self.local / 'a b'))
        self.assertNotIn('error', r, r)
        self.assertEqual((self.local / 'a b' / odd / 'f g').read_text(), 'ok')
        src = self.local / 'src dir'
        (src / "it's").mkdir(parents=True)
        (src / "it's" / 'tab\there').write_text('t')
        (src / '--checkpoint=1').write_text('a file, not an option')
        r = files.push(str(src), f'$DATA/{odd}/up', mode='all')
        self.assertNotIn('error', r, r)
        self.assertEqual((self.cluster / 'data' / odd / 'up' / "it's" / 'tab\there').read_text(), 't')
        self.assertEqual((self.cluster / 'data' / odd / 'up' / '--checkpoint=1').read_text(), 'a file, not an option')

    def test_fetch_refuses_secrets(self):
        for p in ['~/./.ssh', '~//.ssh', '~/.']:
            self.assertIn('error', files.fetch(p, str(self.local)), p)
        (self.cluster / 'data' / 'keys').symlink_to(self.cluster / '.ssh')
        r = files.fetch('$DATA/keys', str(self.local))
        self.assertIn('not below', r.get('error', ''), r)
        self.assertEqual(list(self.local.iterdir()), [])

    def test_fetch_errors(self):
        self.assertIn('does not exist', files.fetch('$DATA/nothing', str(self.local))['error'])
        (self.cluster / 'data' / 'big').write_bytes(b'0' * 2_000_000)
        self.assertIn('max_mb', files.fetch('$DATA/big', str(self.local), max_mb=1)['error'])
        (self.local / 'afile').write_text('x')
        self.assertIn('not a directory', files.fetch('$DATA/big', str(self.local / 'afile'))['error'])

    def test_fetch_retries_only_transient(self):
        (self.cluster / 'data' / 'res').mkdir()
        (self.cluster / 'data' / 'res' / 'a').write_text('a')
        with mock.patch('amadeus.files.time.sleep') as sleep:
            self.flaky('tar czf', 'Permission denied (publickey).')
            r = files.fetch('$DATA/res', str(self.local))
            self.assertEqual((r['attempts'], sleep.call_count), (1, 0), r)
            self.count.unlink()
            self.flaky('tar czf', 'kex_exchange_identification: read: Connection reset by peer')
            r = files.fetch('$DATA/res', str(self.local))
            self.assertEqual((r['attempts'], sleep.call_count), (3, 2), r)
            self.assertIn('error', r)
            self.count.unlink()
            sleep.reset_mock()
            self.flaky('tar czf', 'Connection timed out during banner exchange', fails=2)
            r = files.fetch('$DATA/res', str(self.local))
            self.assertNotIn('error', r, r)
            self.assertEqual((r['attempts'], sleep.call_count), (3, 2))
        self.assertEqual((self.local / 'res' / 'a').read_text(), 'a')


if __name__ == '__main__':
    unittest.main()
