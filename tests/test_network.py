"""amadeus/network.py: Amadeus's own ssh config (no ~/.ssh edits) and the automatic network choice."""
import os
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from amadeus import network

ROOT = Path(__file__).resolve().parent.parent


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        keys = ('HOME', 'XDG_STATE_HOME', 'MUSICA_VPN', 'AMADEUS_USER', 'AMADEUS_SSH', 'MUSICA_SSH_CONFIG',
                'AMADEUS_CONFIG', 'AMADEUS_AUTH_SCRIPT', 'AMADEUS_STEP_DIR')
        self.old = {k: os.environ.get(k) for k in keys}
        os.environ.update(HOME=str(self.tmp), XDG_STATE_HOME=str(self.tmp / 'state'), AMADEUS_USER='asc_user',
                          AMADEUS_CONFIG=str(self.tmp / 'none.toml'),          # no [cluster] user from a file
                          AMADEUS_AUTH_SCRIPT='/nonexistent/musica_auth.sh',   # never a real login
                          AMADEUS_STEP_DIR=str(self.tmp / 'step'))
        for k in ('MUSICA_VPN', 'AMADEUS_SSH', 'MUSICA_SSH_CONFIG'):
            os.environ.pop(k, None)
        network._own.clear()
        network._probe.clear()

    def tearDown(self):
        for k, v in self.old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        network._own.clear()
        network._probe.clear()

    def own_entry(self, proxy=True):
        (self.tmp / '.ssh').mkdir(exist_ok=True)
        (self.tmp / '.ssh' / 'config').write_text(
            'Host musica\n    HostName musica.vie.asc.ac.at\n    User own_user\n'
            + ('    ProxyCommand nc -X 5 -x 127.0.0.1:1080 %h %p\n' if proxy else ''))
        network._own.clear()

    def ssh_G(self, *args):
        return subprocess.run(['ssh', *args, '-G', 'musica'], capture_output=True, text=True).stdout


class OwnConfig(Base):
    def test_writes_its_own_config_and_leaves_ssh_dir_alone(self):
        os.environ['MUSICA_VPN'] = 'wireproxy'
        args = network.ssh_args()
        self.assertEqual(args[0], '-F')
        p = Path(args[1])
        self.assertEqual(p, self.tmp / 'state' / 'amadeus' / 'ssh_config')
        self.assertEqual(p.stat().st_mode & 0o777, 0o600)
        self.assertFalse((self.tmp / '.ssh').exists())
        g = self.ssh_G('-F', str(p))          # the real OpenSSH reads it
        self.assertIn('hostname musica.vie.asc.ac.at', g)
        self.assertIn('user asc_user', g)
        self.assertIn(f'identityagent {self.tmp}/state/amadeus/agent.sock', g)
        self.assertIn(str(network.KNOWN_HOSTS), g)
        self.assertIn('proxycommand nc -X 5 -x 127.0.0.1:1080 %h %p', g)
        self.assertIn('batchmode yes', g)
        self.assertIn('hostname musica.lnz.asc.ac.at',
                      subprocess.run(['ssh', '-F', str(p), '-G', 'musica-lnz'], capture_output=True, text=True).stdout)

    def test_system_vpn_has_no_proxy_and_rewrites_on_change(self):
        os.environ['MUSICA_VPN'] = 'wireproxy'
        p = Path(network.ssh_args()[1])
        os.environ['MUSICA_VPN'] = 'system'
        network.ssh_args()
        self.assertNotIn('ProxyCommand', p.read_text())
        self.assertNotIn('proxycommand nc', self.ssh_G('-F', str(p)))

    def test_known_hosts_holds_the_asc_ca(self):
        self.assertTrue(network.KNOWN_HOSTS.read_text().startswith('@cert-authority *.asc.ac.at ecdsa-sha2-nistp256 '))

    def test_existing_ssh_config_entry_wins(self):
        self.own_entry()
        os.environ.pop('AMADEUS_USER')
        self.assertEqual(network.ssh_args(), [])
        self.assertEqual(network.user(), 'own_user')
        self.assertIsNone(network.setup_problem())
        self.assertEqual(network.vpn_mode(), 'wireproxy')       # from its ProxyCommand
        self.own_entry(proxy=False)
        self.assertEqual(network.vpn_mode(), 'system')
        self.assertEqual(network.script_env()['MUSICA_SSH_CONFIG'], '')    # the scripts then use plain ssh

    def test_missing_user_is_reported(self):
        os.environ.pop('AMADEUS_USER')
        self.assertIn('[cluster] user', network.setup_problem())

    def test_concurrent_writes(self):
        import threading
        errors = []

        def write():
            try:
                network.ensure_config()
            except Exception as e:
                errors.append(e)
        for _ in range(10):
            network.config_path().unlink(missing_ok=True)
            ts = [threading.Thread(target=write) for _ in range(6)]
            [t.start() for t in ts]
            [t.join() for t in ts]
        self.assertEqual(errors, [])
        self.assertEqual([p.name for p in network.state_dir().iterdir()], ['ssh_config'])   # no temp files left

    def test_login_refused_without_user(self):
        from amadeus import auth
        os.environ.pop('AMADEUS_USER')
        with mock.patch.object(auth, 'missing_tools', return_value=[]):
            r = auth.start_login(wait_s=0)
        self.assertEqual(r['login'], 'not started')
        self.assertIn('[cluster] user', r['error'])

    def test_remote_uses_it(self):
        from amadeus import remote
        base = remote.ssh_base()
        self.assertEqual(base[:3], ['ssh', '-o', 'BatchMode=yes'])
        self.assertEqual(base[3], '-F')
        self.assertEqual(base[-1], 'musica')
        os.environ['AMADEUS_SSH'] = 'echo hi'
        self.assertEqual(remote.ssh_base(), ['echo', 'hi'])


class NetworkChoice(Base):
    def probe_at(self, listening):
        s = socket.socket()
        s.bind(('127.0.0.1', 0))
        if listening:
            s.listen()
        self.addCleanup(s.close)
        return mock.patch.object(network, 'PROBE', ('127.0.0.1', s.getsockname()[1]))

    def wireproxy_conf(self):
        (self.tmp / '.config' / 'wireguard').mkdir(parents=True)
        (self.tmp / '.config' / 'wireguard' / 'wireproxy.conf').write_text('[Socks5]\n')

    def test_direct_path_wins_over_a_configured_wireproxy(self):
        self.wireproxy_conf()
        with self.probe_at(True), mock.patch.object(network, 'wireproxy_running', return_value=False):
            self.assertEqual(network.vpn_mode(), 'system')

    def test_wireproxy_when_not_reachable_directly(self):
        self.wireproxy_conf()
        with self.probe_at(False), mock.patch.object(network, 'wireproxy_running', return_value=False):
            self.assertEqual(network.vpn_mode(), 'wireproxy')

    def test_system_when_nothing_is_configured(self):
        with self.probe_at(False), mock.patch.object(network, 'wireproxy_running', return_value=False):
            self.assertEqual(network.vpn_mode(), 'system')

    def test_running_wireproxy_is_used_when_no_direct_path(self):
        with self.probe_at(False), mock.patch.object(network, 'wireproxy_running', return_value=True):
            self.assertEqual(network.vpn_mode(), 'wireproxy')

    def test_direct_path_wins_over_a_running_wireproxy(self):
        with self.probe_at(True), mock.patch.object(network, 'wireproxy_running', return_value=True):
            self.assertEqual(network.vpn_mode(), 'system')

    def test_a_machine_with_a_system_path_never_switches_to_wireproxy(self):
        """A stall or the WireGuard app not yet connected must not start a second tunnel with the same key."""
        self.wireproxy_conf()
        with self.probe_at(True), mock.patch.object(network, 'wireproxy_running', return_value=False):
            self.assertEqual(network.vpn_mode(), 'system')
        network.forget_probe()
        with self.probe_at(False), mock.patch.object(network, 'wireproxy_running', return_value=True):
            self.assertEqual(network.vpn_mode(), 'system')
        os.environ['MUSICA_VPN'] = 'wireproxy'                    # the explicit setting still wins
        self.assertEqual(network.vpn_mode(), 'wireproxy')

    def test_a_failed_probe_is_retried_sooner(self):
        with self.probe_at(False):
            self.assertFalse(network.direct_reachable())
        network._probe['at'] -= network.PROBE_TTL_BAD + 1
        with self.probe_at(True):
            self.assertTrue(network.direct_reachable())

    def test_explicit_setting_wins(self):
        with mock.patch.object(network, 'direct_reachable', side_effect=AssertionError('probed')):
            os.environ['MUSICA_VPN'] = 'system'
            self.assertEqual(network.vpn_mode(), 'system')
            os.environ['MUSICA_VPN'] = 'wireproxy'
            self.assertEqual(network.vpn_mode(), 'wireproxy')

    def test_probe_is_cached(self):
        with self.probe_at(True):
            self.assertTrue(network.direct_reachable())
        with self.probe_at(False):
            self.assertTrue(network.direct_reachable())      # cached
            network.forget_probe()
            self.assertFalse(network.direct_reachable())


class Scripts(Base):
    def access(self, env, expr):
        return subprocess.run(['bash', '-c', f'. {ROOT}/scripts/lib/access.sh; {expr}'], capture_output=True,
                              text=True, env={**os.environ, **env}).stdout.strip()

    def test_scripts_use_the_config_the_server_passes(self):
        out = self.access({'MUSICA_SSH_CONFIG': '/x/ssh_config', 'MUSICA_VPN': 'system'}, 'echo "${SSHF[*]}" $(access_vpn)')
        self.assertEqual(out, '-F /x/ssh_config system')

    def test_scripts_work_it_out_like_the_server(self):
        """Run by hand (no server env): the same decision through amadeus/network.py, and the file is written."""
        p = network.config_path()
        self.assertFalse(p.exists())
        out = self.access({'MUSICA_VPN': 'wireproxy'}, 'echo "${SSHF[*]}" $(access_vpn) $MUSICA_ASC_USER')
        self.assertEqual(out, f'-F {p} wireproxy asc_user')
        self.assertIn('ProxyCommand nc', p.read_text())
        self.own_entry()
        self.assertEqual(self.access({}, 'echo "[${SSHF[*]}]" $(access_vpn)'), '[] wireproxy')

    def test_paths_with_spaces(self):
        os.environ['XDG_STATE_HOME'] = str(self.tmp / 'sp ace')
        p = network.ensure_config()
        r = subprocess.run([str(ROOT / 'scripts' / 'musica_auth.sh'), '--check'], capture_output=True, text=True,
                           env={**os.environ, 'MUSICA_SSH_CONFIG': str(p), 'MUSICA_VPN': 'system',
                                'PATH': f'{self.stub_step()}:{os.environ["PATH"]}'})
        self.assertEqual(r.stdout.strip(), 'no certificate')
        ia = subprocess.run(['bash', '-c', f'IA=$(ssh -F "{p}" -G musica | awk \'/^identityagent /{{sub(/^identityagent /, ""); print}}\'); echo "$IA"'],
                            capture_output=True, text=True).stdout.strip()
        self.assertEqual(ia, str(self.tmp / 'sp ace' / 'amadeus' / 'agent.sock'))

    def stub_step(self):
        stub = self.tmp / 'bin'
        stub.mkdir(exist_ok=True)
        (stub / 'step').write_text('#!/bin/sh\nexit 0\n')     # no certificate
        (stub / 'step').chmod(0o755)
        return stub

    def test_auth_script_reads_user_and_agent_from_it(self):
        p = network.ensure_config()
        stub = self.stub_step()
        r = subprocess.run([str(ROOT / 'scripts' / 'musica_auth.sh'), '--check'], capture_output=True, text=True,
                           env={**os.environ, 'MUSICA_SSH_CONFIG': str(p), 'MUSICA_VPN': 'system',
                                'PATH': f'{stub}:{os.environ["PATH"]}'})
        self.assertEqual((r.returncode, r.stdout.strip()), (1, 'no certificate'))
        user = subprocess.run(['bash', '-c', f'. {ROOT}/scripts/lib/access.sh; ssh "${{SSHF[@]}}" -G musica'],
                              capture_output=True, text=True, env={**os.environ, 'MUSICA_SSH_CONFIG': str(p)}).stdout
        self.assertIn('user asc_user', user)


if __name__ == '__main__':
    unittest.main()
