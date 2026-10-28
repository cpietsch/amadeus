"""A stalled userspace tunnel (wireproxy) is restarted and safe commands run again (auth.heal_tunnel, remote.run).

The stall is simulated with an ssh stand-in that fails before login like a stalled tunnel until the fake restart
(auth._restart_tunnel, patched) has run, then hands over to tests/fake_ssh.sh. No real tunnel is touched.
"""
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from amadeus import auth, remote

HERE = Path(__file__).resolve().parent

# Fails like a stalled tunnel (exit 255 before login) until the file $STALL_HEALED exists; one line per call in
# $STALL_LOG. Then the fake cluster runs the command.
STALL_SSH = r'''#!/bin/bash
echo "$1" >> "$STALL_LOG"
if [ ! -e "$STALL_HEALED" ]; then
    echo "Connection timed out during banner exchange" >&2
    echo "Connection to UNKNOWN port 65535 timed out" >&2
    exit 255
fi
exec bash "$FAKE_SSH" "$@"
'''


class Stall(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        cluster = self.tmp / 'cluster'
        cluster.mkdir()
        shim = self.tmp / 'ssh'
        shim.write_text(STALL_SSH)
        self.healed = self.tmp / 'healed'
        self.log = self.tmp / 'calls'
        env = {'AMADEUS_SSH': f'bash {shim}', 'FAKE_SSH': str(HERE / 'fake_ssh.sh'), 'FAKE_HOME': str(cluster),
               'STALL_HEALED': str(self.healed), 'STALL_LOG': str(self.log), 'AMADEUS_RETRY_DELAY': '0',
               'MUSICA_VPN': 'wireproxy', 'XDG_STATE_HOME': str(self.tmp / 'state')}
        self.old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        self.restarts = []

        def fake_restart():
            self.restarts.append(time.time())
            self.healed.touch()
            return 'stopped pid 4242; started again'

        for patch in (mock.patch.object(auth, '_restart_tunnel', side_effect=fake_restart),
                      mock.patch.object(auth, 'cached_certificate', return_value={'valid': True, 'minutes_left': 600})):
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp)

    def calls(self) -> int:
        return len(self.log.read_text().splitlines()) if self.log.exists() else 0

    def test_restart_and_one_retry_for_a_safe_command(self):
        with mock.patch.object(remote, 'retry_delay', return_value=0) as delay:
            r = remote.run('echo hi', retry=True)
        self.assertTrue(r.ok, r)
        self.assertEqual((r.out.strip(), r.attempts, self.calls(), len(self.restarts)), ('hi', 2, 2, 1))
        self.assertEqual(delay.call_count, 0)          # the retry through the new tunnel goes at once
        self.assertIn('tunnel restarted after a stall', r.recovery)
        self.assertIn('run again', r.recovery)
        self.assertEqual(r.summary()['recovery'], r.recovery)

    def test_a_restart_grants_one_attempt_even_after_the_last(self):
        auth._restart_stamp().parent.mkdir(parents=True, exist_ok=True)
        auth._restart_stamp().write_text(f'{time.time():.3f}\n')    # another call restarted it just now
        calls = []
        real = auth.heal_tunnel

        def heal(err):   # the rate limit runs out during the third (last) attempt
            calls.append(err)
            if len(calls) == 3:
                auth._restart_stamp().write_text('0\n')
            return real(err)

        with mock.patch.object(auth, 'heal_tunnel', side_effect=heal):
            r = remote.run('echo hi', retry=True)
        self.assertTrue(r.ok, r)
        self.assertEqual((r.attempts, len(self.restarts)), (4, 1))

    def test_no_retry_for_a_command_not_safe_to_repeat(self):
        r = remote.run('echo submitted', retry=False)
        self.assertEqual((r.code, r.attempts, self.calls()), (255, 1, 1))
        self.assertEqual(len(self.restarts), 1)        # the tunnel is healed for the next call ...
        self.assertIn('not repeated', r.recovery)      # ... but the command is not run again
        self.assertIn('not repeated', r.summary()['recovery'])

    def test_nothing_in_system_mode(self):
        os.environ['MUSICA_VPN'] = 'system'
        r = remote.run('echo hi', retry=True)
        self.assertEqual((r.code, r.attempts, r.recovery, self.restarts), (255, 3, '', []))

    def test_nothing_without_a_valid_certificate_or_for_other_errors(self):
        with mock.patch.object(auth, 'cached_certificate', return_value={'valid': False}):
            self.assertIsNone(auth.heal_tunnel('Connection to UNKNOWN port 65535 timed out'))
        self.assertIsNone(auth.heal_tunnel('user1@musica: Permission denied (publickey,keyboard-interactive).'))
        self.assertIsNone(auth.heal_tunnel('Host key verification failed.'))
        self.assertEqual(self.restarts, [])

    def test_rate_limit_shared_across_calls(self):
        r = remote.run('echo one', retry=True)
        self.assertTrue(r.ok)
        self.healed.unlink()                           # the tunnel stalls again within the minute
        r = remote.run('echo two', retry=True)
        self.assertEqual((r.code, r.attempts, r.recovery, len(self.restarts)), (255, 3, '', 1))
        stamp = float(auth._restart_stamp().read_text())
        auth._restart_stamp().write_text(f'{stamp - 61:.3f}\n')   # a minute later
        r = remote.run('echo three', retry=True)
        self.assertTrue(r.ok, r)
        self.assertEqual(len(self.restarts), 2)

    def test_parallel_calls_restart_once(self):
        out = []
        ts = [threading.Thread(target=lambda: out.append(remote.run('echo hi', retry=True))) for _ in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertTrue(all(r.ok for r in out), out)
        self.assertEqual(len(self.restarts), 1)

    def test_rate_limit_shared_across_processes(self):
        code = ('import sys; from unittest import mock; from amadeus import auth\n'
                'with mock.patch.object(auth, "cached_certificate", return_value={"valid": True}), '
                'mock.patch.object(auth, "_restart_tunnel", return_value="x"):\n'
                '    print(auth.heal_tunnel("Connection timed out during banner exchange"))\n')
        run = lambda: subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                     cwd=HERE.parent, env=os.environ).stdout.strip()
        self.assertEqual(run(), 'tunnel restarted after a stall (x)')
        self.assertEqual(run(), 'None')                # the other process's restart counts
        self.assertIsNone(auth.heal_tunnel('Connection timed out during banner exchange'))
        self.assertEqual(self.restarts, [])

    def test_ssh_hint_mentions_the_restart(self):
        hint = auth.ssh_hint('Connection to UNKNOWN port 65535 timed out')
        self.assertIn('restarts the tunnel', hint)
        self.assertIn('(push_files/fetch_files) do not restart it', hint)   # transfers are not covered yet

    def test_no_restart_when_the_command_ran_and_exited_255(self):
        """A command that ran on the login node and exits 255 after a connection error of its own (rsync or ssh to
        another host) must not restart a healthy tunnel nor claim that nothing ran."""
        self.healed.touch()                            # the tunnel is fine
        ran = self.tmp / 'cluster' / 'ran.txt'
        for err in ('ssh: connect to host x port 22: Connection timed out',
                    'ssh: connect to host x port 22: Connection refused',
                    'kex_exchange_identification: Connection closed by remote host',   # even a stall message
                    'Connection to UNKNOWN port 65535 timed out'):
            for retry in (False, True):
                r = remote.run(f'echo ran >> $HOME/ran.txt; echo "{err}" >&2; exit 255', retry=retry)
                self.assertEqual((r.code, r.recovery, self.restarts), (255, '', []), (err, retry))
                self.assertTrue(r.started)
                self.assertIn(err, r.err)
                self.assertNotIn(remote.START_MARK, r.err)
                self.assertIn('started on the login node', r.summary()['note'])
        self.assertGreaterEqual(ran.read_text().count('ran'), 8)

    def test_no_restart_when_ssh_printed_output(self):
        # an ssh stand-in that prints a stall message but also stdout: something answered, so no restart
        shim = self.tmp / 'chatty'
        shim.write_text('#!/bin/bash\necho partial\necho "Connection timed out during banner exchange" >&2\nexit 255\n')
        os.environ['AMADEUS_SSH'] = f'bash {shim}'
        r = remote.run('true', retry=False)
        self.assertEqual((r.code, r.recovery, self.restarts), (255, '', []))

    def test_only_ssh_setup_messages_restart(self):
        for err in ('ssh: connect to host x port 22: Connection timed out', 'Connection refused',
                    'rsync error: unexplained error (code 255)'):
            self.assertIsNone(auth.heal_tunnel(err), err)
        self.assertEqual(self.restarts, [])
        self.assertTrue(auth.heal_tunnel('kex_exchange_identification: Connection closed by remote host'))

    def test_start_mark_hidden_from_normal_results(self):
        self.healed.touch()
        r = remote.run('echo out; echo warn >&2', retry=True)
        self.assertTrue(r.ok and r.started, r)
        self.assertEqual((r.out, r.err), ('out\n', 'warn'))


class RestartByPid(unittest.TestCase):
    """_restart_tunnel stops our wireproxy by PID (argv, not a pattern) and starts it with vpn.sh up."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.old = {k: os.environ.get(k) for k in ('HOME', 'AMADEUS_AUTH_SCRIPT')}
        os.environ['HOME'] = str(self.tmp)
        (self.tmp / '.config' / 'wireguard').mkdir(parents=True)
        self.conf = self.tmp / '.config' / 'wireguard' / 'wireproxy.conf'
        self.conf.write_text('[Socks5]\n')
        bindir = self.tmp / 'bin'
        bindir.mkdir()
        vpn = bindir / 'vpn.sh'
        vpn.write_text(f'#!/bin/bash\necho "$1" >> {self.tmp}/vpn-calls\n')
        vpn.chmod(0o755)
        os.environ['AMADEUS_AUTH_SCRIPT'] = str(bindir / 'musica_auth.sh')   # vpn.sh is looked up next to it
        self.procs = []

    def tearDown(self):
        for p in self.procs:              # each stand-in runs in its own session: its sleep goes too
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
            p.wait()
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp)

    def start(self, conf):
        """A stand-in with wireproxy's argv (`wireproxy -c <conf>`) that only sleeps."""
        p = subprocess.Popen(['bash', '-c', f'exec -a wireproxy bash -c "sleep 300; :" -c {conf}'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        self.procs.append(p)
        return p

    @unittest.skipUnless(auth.PROC, 'reads /proc')
    def test_stops_only_our_wireproxy_and_starts_it_again(self):
        ours = self.start(self.conf)
        other = self.start(self.tmp / 'other.conf')
        self.assertTrue(_wait(lambda: ours.pid in auth.wireproxy_pids()))
        self.assertNotIn(other.pid, auth.wireproxy_pids())
        t0 = time.monotonic()
        note = auth._restart_tunnel()
        self.assertFalse(Path(f'/proc/{ours.pid}').exists())   # stopped (and reaped by _restart_tunnel's wait)
        self.assertLess(time.monotonic() - t0, 4)       # by SIGTERM, not the SIGKILL after 5 s
        self.assertIsNone(other.poll())                # another wireproxy config is left alone
        self.assertIn(f'stopped pid {ours.pid}', note)
        self.assertIn('started again', note)
        self.assertEqual((self.tmp / 'vpn-calls').read_text(), 'up\n')


def _wait(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


if __name__ == '__main__':
    unittest.main()
