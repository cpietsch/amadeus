"""auth.start_login / status and remote.run retries, in a sandbox: a temp HOME, a fake musica_auth.sh (never the real
one, never step) and a flaky ssh stand-in. Nothing here talks to the cluster."""
import json
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

ROOT = Path(__file__).resolve().parents[1]

# Stand-in for scripts/musica_auth.sh. It writes where the real one lets step write ($HOME/.step/asc-login.log for
# --wait, asc-renew.log for --renew) and prints what the real wrapper prints. Behaviour from the environment:
#   $HOME/cert         minutes left of the certificate (no file: no certificate); FAKE_CHECK_S: seconds --check takes
#   FAKE_LINK_DELAY    seconds until "step" writes the device link to its log; FAKE_CODE: the code
#   FAKE_WRAPPER_LINK  1: the wrapper also prints the link once it is there (the real one greps once, after 5-10 s)
#   FAKE_STEP_EXTRA    a line "step" writes after the link
#   FAKE_OUTCOME       hang (default) | expire | ok, after FAKE_RUN_S seconds
FAKE_AUTH = r'''#!/bin/bash
echo "$$ $*" >> "$HOME/auth-calls.log"
STEP=$HOME/.step; mkdir -p "$STEP"
if [ "$1" = --check ]; then
    sleep "${FAKE_CHECK_S:-0}"
    [ -f "$HOME/cert" ] || { echo "no certificate"; exit 1; }
    left=$(cat "$HOME/cert")
    echo "certificate valid until 2026-10-15T08:00:00+02:00 ($left min left)"
    [ "$left" -ge 60 ]; exit
fi
if [ "$1" = --renew ]; then
    LOG=$STEP/asc-renew.log
    echo "Renewing (current certificate valid until 2026-10-15T08:00:00+02:00). Approve in a browser within about 60 s:"
else
    LOG=$STEP/asc-login.log
    echo "No valid certificate. Approve in a browser:"
fi
link="https://auth.asc.ac.at/device?code=${FAKE_CODE:-11111111}"
( sleep "${FAKE_LINK_DELAY:-0}"
  { echo "Visit $link and enter the code"; [ -n "$FAKE_STEP_EXTRA" ] && echo "$FAKE_STEP_EXTRA"; } > "$LOG" ) &
writer=$!
[ "${FAKE_WRAPPER_LINK:-0}" = 1 ] && { wait $writer; echo "$link"; }
case "${FAKE_OUTCOME:-hang}" in
    hang) while :; do sleep 0.2; done ;;
    expire) sleep "${FAKE_RUN_S:-1}"; kill $writer 2>/dev/null; wait $writer
            echo "error: device authorization grant expired" >> "$LOG"; echo "login failed:"; cat "$LOG"; exit 1 ;;
    ok) sleep "${FAKE_RUN_S:-1}"; kill $writer 2>/dev/null; wait $writer; echo "certificate valid until 2026-10-16T08:00:00+02:00"; exit 0 ;;
esac
'''

FAKE_ENV = ('FAKE_CHECK_S', 'FAKE_LINK_DELAY', 'FAKE_CODE', 'FAKE_WRAPPER_LINK', 'FAKE_STEP_EXTRA', 'FAKE_OUTCOME',
            'FAKE_RUN_S')


def wait_until(cond, timeout=10.0, step=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(step)
    return False


def state(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
    except OSError:
        return None


class AuthSandbox(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.home = self.tmp / 'home'
        self.home.mkdir()
        self.bin = self.tmp / 'bin'
        self.bin.mkdir()
        self.script = self.bin / 'musica_auth.sh'
        self.script.write_text(FAKE_AUTH)
        self.script.chmod(0o755)
        env = {'HOME': str(self.home), 'AMADEUS_AUTH_SCRIPT': str(self.script),
               'AMADEUS_STEP_DIR': str(self.home / '.step')}
        self.old = {k: os.environ.get(k) for k in [*env, *FAKE_ENV]}
        for k in FAKE_ENV:
            os.environ.pop(k, None)
        os.environ.update(env)
        auth._proc = auth._proc_login = None
        self.extra = []  # processes the test starts itself
        real = auth.missing_tools
        auth.missing_tools = lambda: []   # the fake auth script needs no local tools
        self.addCleanup(setattr, auth, 'missing_tools', real)

    def tearDown(self):
        pids = set()
        calls = self.home / 'auth-calls.log'
        if calls.exists():
            pids |= {int(l.split()[0]) for l in calls.read_text().splitlines() if l.strip()}
        for p in self.extra:  # our own children, not reaped yet: their pids are still theirs
            for kill in (lambda: os.killpg(p.pid, signal.SIGKILL), p.kill):
                try:
                    kill()
                except OSError:
                    pass
        for pid in pids:  # every fake login runs in its own session: kill the group (its loop and sleeps too)
            try:
                if str(self.tmp) in Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace'):
                    os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass
        for p in self.extra:
            p.wait()
        if auth._proc is not None:
            auth._proc.wait()
        auth._proc = auth._proc_login = None
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp)

    def calls(self, arg):
        p = self.home / 'auth-calls.log'
        return [l for l in (p.read_text().splitlines() if p.exists() else []) if l.split()[1:] == [arg]]

    def set_cert(self, minutes):
        (self.home / 'cert').write_text(str(minutes))

    def run_outside(self, *argv, **env):
        """A login a human started (not through the server), same HOME."""
        p = subprocess.Popen([*argv], env={**os.environ, **env}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        self.extra.append(p)
        return p


class Locking(AuthSandbox):
    """Findings 25 and 38: concurrent start_login calls must start one login, never two."""

    def test_parallel_threads_start_one_login(self):
        self.set_cert(600)  # valid: early renewal, the dangerous case (two agents on agent-renew.sock)
        os.environ['FAKE_CHECK_S'] = '1'
        results = []
        threads = [threading.Thread(target=lambda: results.append(auth.start_login(wait_s=5))) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(self.calls('--renew')), 1, results)
        self.assertEqual(sorted(r['login'] for r in results), ['already running'] * 3 + ['started'])
        self.assertEqual({r.get('link') for r in results}, {'https://auth.asc.ac.at/device?code=11111111'})
        self.assertEqual(len(self.calls('--check')), 1)  # the others saw the running login and did not check again

    def test_two_server_processes_start_one_login(self):
        self.set_cert(600)
        os.environ['FAKE_CHECK_S'] = '1'
        code = 'import json; from amadeus import auth; print(json.dumps(auth.start_login(wait_s=4)))'
        other = subprocess.Popen([sys.executable, '-c', code], cwd=ROOT, stdout=subprocess.PIPE, text=True)
        mine = auth.start_login(wait_s=4)
        theirs = json.loads(other.communicate(timeout=60)[0])
        self.assertEqual(len(self.calls('--renew')), 1, (mine, theirs))
        self.assertEqual(sorted([mine['login'], theirs['login']]), ['already running', 'started'])
        self.assertEqual(mine.get('link'), theirs.get('link'))

    def test_login_started_outside_the_server_is_not_doubled(self):
        """A human's `scripts/musica_auth.sh --renew` (AGENTS.md §3) runs: no second renewal, its link is returned."""
        self.set_cert(600)
        self.run_outside(str(self.script), '--renew')
        self.assertTrue(wait_until(lambda: (self.home / '.step' / 'asc-renew.log').exists()))
        r = auth.start_login(wait_s=3)
        self.assertEqual(r['login'], 'already running')
        self.assertIn('outside this server', r['started_by'])
        self.assertEqual(r['link'], 'https://auth.asc.ac.at/device?code=11111111')
        self.assertEqual(len(self.calls('--renew')), 1)
        self.assertEqual(self.calls('--check'), [])
        self.assertEqual(auth.status(probe=False)['login_in_progress'], 'started outside this server')

    def test_running_step_login_blocks_a_new_one(self):
        # step is a binary (argv[0] 'step'); a script would show its interpreter. Python runs ./ssh under argv
        # 'step ssh login …', so /proc/<pid>/cmdline looks like the real one.
        (self.bin / 'ssh').write_text('import time\nwhile True:\n    time.sleep(0.2)\n')
        p = subprocess.Popen(['step', 'ssh', 'login', 'tester', '--context', 'asc', '--console'], executable=sys.executable,
                             cwd=self.bin, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        self.extra.append(p)
        self.assertTrue(wait_until(lambda: auth._foreign_login() is not None))
        self.assertEqual(auth._foreign_login().pid, p.pid)
        r = auth.start_login(wait_s=0)
        self.assertEqual(r['login'], 'already running')
        self.assertEqual(self.calls('--wait'), [])

    def test_login_of_another_home_is_ignored(self):
        other = self.tmp / 'other-home'
        other.mkdir()
        self.run_outside(str(self.script), '--renew', HOME=str(other))
        self.assertTrue(wait_until(lambda: (other / '.step' / 'asc-renew.log').exists()))
        self.assertIsNone(auth._foreign_login())


class Liveness(AuthSandbox):
    """Finding 36: an exited login (zombie) or a reused pid is not 'already running'; no expired link comes back."""

    def test_zombie_login_is_replaced_and_its_link_never_returned(self):
        os.environ.update(FAKE_OUTCOME='expire', FAKE_RUN_S='2', FAKE_WRAPPER_LINK='1')
        r1 = auth.start_login(wait_s=5)
        self.assertEqual(r1['login'], 'started')
        self.assertEqual(r1['link'], 'https://auth.asc.ac.at/device?code=11111111')
        p = auth._proc
        auth._proc = None  # the old code dropped the Popen: nothing reaps the exited wrapper
        self.assertTrue(wait_until(lambda: state(p.pid) == 'Z'), state(p.pid))
        self.assertIsNone(auth.login_running())
        self.assertIsNone(state(p.pid))  # reaped with waitpid(WNOHANG)
        p.returncode = 1  # already reaped: keep Popen from waiting for it again

        os.environ['FAKE_CODE'] = '22222222'
        r2 = auth.start_login(wait_s=5)
        self.assertEqual(r2['login'], 'started')
        self.assertIn('device code expired', r2['previous_login'])
        self.assertIn('A new login with a new code was started', r2['note'])
        self.assertEqual(r2['link'], 'https://auth.asc.ac.at/device?code=22222222')
        self.assertNotIn('11111111', json.dumps(r2))
        self.assertEqual(len(self.calls('--wait')), 2)

    def test_exited_login_reaped_by_poll(self):
        os.environ.update(FAKE_OUTCOME='expire', FAKE_RUN_S='0.5')
        auth.start_login(wait_s=0)
        p = auth._proc
        self.assertTrue(wait_until(lambda: auth.login_running() is None))
        self.assertIsNotNone(p.returncode)
        self.assertIsNone(state(p.pid))

    def test_reused_pid_in_pid_file_is_not_a_login(self):
        sleeper = subprocess.Popen(['sleep', '30'])
        self.extra.append(sleeper)
        (self.home / '.step').mkdir()
        (self.home / '.step' / 'amadeus-login.pid').write_text(str(sleeper.pid))  # old format: pid only
        self.assertIsNone(auth.login_running())
        r = auth.start_login(wait_s=3)
        self.assertEqual(r['login'], 'started')
        self.assertEqual(len(self.calls('--wait')), 1)
        sleeper.kill()

    def test_pid_file_of_a_live_login_counts(self):
        """Another server process (or this one before a restart) started it: found through the pid file."""
        os.environ['FAKE_WRAPPER_LINK'] = '1'
        r1 = auth.start_login(wait_s=3)
        p = auth._proc
        auth._proc = None  # as seen from a fresh process
        self.assertEqual(auth.login_running(), p.pid)
        r2 = auth.start_login(wait_s=3)
        self.assertEqual(r2['login'], 'already running')
        self.assertEqual(r2['link'], r1['link'])
        self.assertEqual(len(self.calls('--wait')), 1)
        auth._proc = p  # for tearDown

    def test_login_that_ends_while_waiting_reports_the_outcome(self):
        os.environ.update(FAKE_OUTCOME='expire', FAKE_RUN_S='0.3', FAKE_LINK_DELAY='5')
        r = auth.start_login(wait_s=5)
        self.assertNotIn('link', r)
        self.assertEqual(r['result'], 'device code expired before it was approved')
        self.assertIn("Don't loop", r['next'])

    def test_successful_login_reports_the_certificate(self):
        os.environ.update(FAKE_OUTCOME='ok', FAKE_RUN_S='0.3', FAKE_LINK_DELAY='5')
        auth._cert.update(at=time.monotonic(), cert={'valid': True, 'minutes_left': 20}, expires=time.time() + 1200)
        r = auth.start_login(wait_s=5)
        self.assertNotIn('link', r)
        self.assertIn('certificate obtained', r['result'])
        self.assertEqual(auth._cert, {})           # the next tool result re-reads the expiry
        r2 = auth.start_login(wait_s=0)
        self.assertEqual(r2['login'], 'started')
        self.assertNotIn('note', r2)  # the previous one succeeded: nothing to warn about
        self.assertIn('certificate obtained', r2['previous_login'])


class Expiry(unittest.TestCase):
    def test_local_time_gets_this_process_offset(self):
        with mock.patch.dict(os.environ, {'TZ': 'Europe/Vienna'}):
            time.tzset()
            self.assertEqual(auth.with_offset('2026-10-15T12:06:36'), '2026-10-15T12:06:36+02:00')
        time.tzset()  # patch.dict has restored TZ

    def test_zone_kept_and_junk_passed_through(self):
        self.assertEqual(auth.with_offset('2026-10-15T08:00:00+02:00'), '2026-10-15T08:00:00+02:00')
        self.assertEqual(auth.with_offset('soon'), 'soon')

class CachedExpiry(AuthSandbox):
    """Findings 0, 9, 36, 65, 95: the cached expiry behind the warnings in tool results."""

    def setUp(self):
        super().setUp()
        auth.forget_certificate()

    def tearDown(self):
        auth.forget_certificate()
        super().tearDown()

    def test_clock_is_vienna_time_with_offset(self):
        from datetime import datetime, timezone
        cest = datetime(2026, 10, 15, 10, 0, tzinfo=timezone.utc).timestamp()
        cet = datetime(2026, 10, 26, 10, 0, tzinfo=timezone.utc).timestamp()   # after the change on 25 Oct
        self.assertEqual(auth.clock(cest, with_date=False), '12:00+02:00')
        self.assertEqual(auth.clock(cet, with_date=True), '2026-10-26 11:00+01:00')
        self.assertRegex(auth.clock(time.time()), r'^\d\d:\d\d[+-]\d\d:\d\d$')   # today: no date

    def test_minutes_counted_from_the_cached_expiry(self):
        self.set_cert(30)
        self.assertIn(auth.cached_certificate()['minutes_left'], (29, 30))
        auth._cert['expires'] = time.time() - 120       # it ran out since the last read
        c = auth.cached_certificate()
        self.assertFalse(c['valid'])
        self.assertIn('expired at', auth.expiry_warning())
        self.assertEqual(len(self.calls('--check')), 1)  # read once

    def test_warning_starts_no_login_without_the_human(self):          # AGENTS.md §3
        for left in (30, None):
            if left is None:
                (self.home / 'cert').unlink()
            else:
                self.set_cert(left)
            auth.forget_certificate()
            w = auth.expiry_warning()
            self.assertIn('not a cron, /loop or stop-hook prompt', w)
            self.assertIn('start no code', w)
            self.assertNotIn(' now', w)

    def test_short_cache_close_to_expiry(self):                         # a renewal elsewhere shows within 30 s
        self.set_cert(30)
        self.assertIn('min)', auth.expiry_warning())
        self.set_cert(900)
        self.assertIsNotNone(auth.expiry_warning())  # cached
        auth._cert['at'] -= auth.CERT_TTL_BAD + 1
        self.assertIsNone(auth.expiry_warning())
        auth._cert['at'] -= auth.CERT_TTL_BAD + 1     # valid with plenty left: the long TTL applies
        n = len(self.calls('--check'))
        auth.expiry_warning()
        self.assertEqual(len(self.calls('--check')), n)

    def test_failed_check_gives_no_warning(self):
        self.script.write_text('#!/bin/bash\nsleep 5\n')
        with mock.patch.object(auth, 'CERT_QUICK_S', 0.3):
            self.assertIsNone(auth.cached_certificate())
            self.assertIsNone(auth.expiry_warning())
            self.assertIsNone(auth.outlasting({'1': time.time() + 10 ** 6}))

    def test_outlasting_names_only_the_late_jobs(self):
        self.set_cert(120)
        now = time.time()
        note = auth.outlasting({'1': now + 3600, '2': now + 3 * 3600, '3': None})
        self.assertIn('2 (ends by', note)
        self.assertNotIn('1 (', note)
        self.assertIsNone(auth.outlasting({'1': now + 3600}))

    def test_ssh_hint_tells_certificate_from_tunnel(self):
        self.set_cert(300)
        auth.cached_certificate()
        self.assertIn('certificate expired', auth.ssh_hint('u@musica: Permission denied (publickey,keyboard-interactive).'))
        self.assertEqual(auth._cert, {})                # an auth failure drops the cached expiry
        self.assertIn('tunnel stall', auth.ssh_hint('Connection to UNKNOWN port 65535 timed out'))
        (self.home / 'cert').unlink()
        auth.forget_certificate()
        self.assertIn('certificate expired', auth.ssh_hint('Connection timed out during banner exchange'))
        self.assertIsNone(auth.ssh_hint('Permission denied (publickey,gssapi-keyex,gssapi-with-mic).'))
        self.assertIsNone(auth.ssh_hint('Host key verification failed.'))


class Links(AuthSandbox):
    """Finding 37: the link comes from step's own log too, only if fresh and not followed by expiry or failure."""

    def test_link_from_step_login_log_when_the_wrapper_has_not_printed_it(self):
        os.environ['FAKE_LINK_DELAY'] = '2'  # step prints it after the wrapper's one-shot grep
        t0 = time.monotonic()
        r = auth.start_login(wait_s=8)
        self.assertGreaterEqual(time.monotonic() - t0, 1.5)
        self.assertEqual(r['link'], 'https://auth.asc.ac.at/device?code=11111111')
        self.assertEqual(r['mode'], 'new login')
        self.assertIn('Show this link', r['instructions'])
        self.assertNotIn('11111111', r['log_tail'])  # links appear only in 'link'

    def test_link_from_step_renew_log(self):
        self.set_cert(30)
        os.environ['FAKE_LINK_DELAY'] = '1'
        r = auth.start_login(wait_s=6)
        self.assertEqual(len(self.calls('--renew')), 1)
        self.assertIn('early renewal', r['mode'])
        self.assertEqual(r['link'], 'https://auth.asc.ac.at/device?code=11111111')

    def test_stale_step_log_is_ignored(self):
        step = self.home / '.step'
        step.mkdir()
        old = step / 'asc-login.log'
        old.write_text('Visit https://auth.asc.ac.at/device?code=99999999 and enter the code\n')
        os.utime(old, (time.time() - 300, time.time() - 300))
        os.environ['FAKE_LINK_DELAY'] = '30'
        r = auth.start_login(wait_s=1.5)
        self.assertEqual(r['login'], 'started')
        self.assertNotIn('link', r)
        self.assertIn('no device link yet', r['next'])
        self.assertNotIn('99999999', json.dumps(r))

    def test_expired_link_in_a_fresh_log_is_not_returned(self):
        os.environ['FAKE_STEP_EXTRA'] = 'error: device authorization grant expired'
        r = auth.start_login(wait_s=1.5)
        self.assertNotIn('link', r)
        self.assertNotIn('11111111', json.dumps(r))

    def test_live_link_rules(self):
        link = 'https://auth.asc.ac.at/device?code=123'
        self.assertEqual(auth._live_link(f'Visit {link}\n'), link)
        self.assertIsNone(auth._live_link(f'Visit {link}\nerror: device authorization grant expired\n'))
        self.assertIsNone(auth._live_link(f'login failed:\nVisit {link}\n'))
        self.assertEqual(auth._live_link(f'old {link}0 expired\nVisit {link}\n'), link)
        self.assertIsNone(auth._live_link('nothing'))

    def test_status_shows_the_login_in_progress(self):
        auth.start_login(wait_s=0)
        st = auth.status(probe=False)
        self.assertIs(st['login_in_progress'], True)
        self.assertEqual(st['certificate'], {'valid': False, 'detail': 'no certificate'})
        self.assertIn('start_login', st['next'])


FLAKY = r'''#!/bin/bash
# ssh stand-in: the first $FLAKY_FAILS calls fail (print $FLAKY_ERR, exit 255; or hang with FLAKY_HANG=1)
n=$(( $(cat "$FLAKY_COUNT" 2>/dev/null || echo 0) + 1 )); echo $n > "$FLAKY_COUNT"
if [ "$n" -le "${FLAKY_FAILS:-0}" ]; then
    [ "${FLAKY_HANG:-0}" = 1 ] && exec sleep 30
    echo "$FLAKY_ERR" >&2; exit 255
fi
echo "ok $n"
'''


class Retries(unittest.TestCase):
    """Finding 35: keepalive timeouts and remote closes are transient; one retry after a timeout; no final sleep."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        shim = self.tmp / 'ssh'
        shim.write_text(FLAKY)
        env = {'AMADEUS_SSH': f'bash {shim}', 'AMADEUS_RETRY_DELAY': '0', 'FLAKY_COUNT': str(self.tmp / 'n'),
               'FLAKY_FAILS': '0', 'FLAKY_ERR': '', 'FLAKY_HANG': '0'}
        self.old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)

    def tearDown(self):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp)

    def flaky(self, fails, err='', hang=False):
        os.environ.update(FLAKY_FAILS=str(fails), FLAKY_ERR=err, FLAKY_HANG='1' if hang else '0')

    def test_transient_messages(self):
        for msg in ('Timeout, server musica.vie.asc.ac.at not responding.',
                    'Connection to musica.vie.asc.ac.at closed by remote host.',
                    'client_loop: send disconnect: Broken pipe',
                    'ssh_exchange_identification: read: Connection reset by peer',
                    'Connection timed out during banner exchange',
                    'kex_exchange_identification: Connection closed by remote host',
                    'Connection to UNKNOWN port 65535 timed out'):
            self.assertTrue(remote.TRANSIENT.search(msg), msg)
        for msg in ('user1@musica.vie.asc.ac.at: Permission denied (publickey,keyboard-interactive).',
                    'Host key verification failed.', 'ssh: Could not resolve hostname musica: Name or service not known'):
            self.assertFalse(remote.TRANSIENT.search(msg), msg)

    def test_keepalive_timeout_is_retried(self):
        self.flaky(1, 'Timeout, server musica.vie.asc.ac.at not responding.')
        r = remote.run('squeue', retry=True)
        self.assertTrue(r.ok, r)
        self.assertEqual((r.attempts, r.out.strip()), (2, 'ok 2'))

    def test_remote_close_is_retried(self):
        self.flaky(2, 'Connection to musica.vie.asc.ac.at closed by remote host.')
        r = remote.run('squeue', retry=True)
        self.assertEqual((r.code, r.attempts), (0, 3))

    def test_no_retry_without_retry_flag_or_for_auth_failures(self):
        self.flaky(1, 'Timeout, server musica.vie.asc.ac.at not responding.')
        r = remote.run('sbatch x', retry=False)
        self.assertEqual((r.code, r.attempts), (255, 1))
        (self.tmp / 'n').unlink()
        self.flaky(1, 'Permission denied (publickey,keyboard-interactive).')
        r = remote.run('squeue', retry=True)
        self.assertEqual((r.code, r.attempts), (255, 1))

    def test_timeout_retried_once(self):
        self.flaky(1, hang=True)
        r = remote.run('squeue', timeout=1, retry=True)
        self.assertEqual((r.code, r.attempts), (0, 2))
        (self.tmp / 'n').unlink()
        self.flaky(5, hang=True)
        r = remote.run('squeue', timeout=1, retry=True)
        self.assertTrue(r.timed_out)
        self.assertEqual(r.attempts, 2)  # one retry after a timeout, not three attempts
        (self.tmp / 'n').unlink()
        r = remote.run('squeue', timeout=1, retry=False)
        self.assertEqual((r.code, r.attempts), (124, 1))

    def test_no_sleep_after_the_last_attempt(self):
        self.flaky(5, 'Connection to musica.vie.asc.ac.at closed by remote host.')
        with mock.patch.object(remote, 'retry_delay', return_value=0) as delay:  # one call per sleep
            r = remote.run('squeue', retry=True)
        self.assertEqual((r.code, r.attempts), (255, 3))
        self.assertEqual(delay.call_count, 2)
        (self.tmp / 'n').unlink()
        self.flaky(5, hang=True)
        with mock.patch.object(remote, 'retry_delay', return_value=0) as delay:
            r = remote.run('squeue', timeout=1, retry=True)
        self.assertEqual((r.code, r.attempts, delay.call_count), (124, 2, 1))


if __name__ == '__main__':
    unittest.main()


class MissingTools(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.old = {k: os.environ.get(k) for k in ('PATH', 'HOME', 'MUSICA_VPN')}
        os.environ.update({'HOME': str(self.tmp), 'MUSICA_VPN': 'system'})

    def tearDown(self):
        for k, v in self.old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def test_reports_step_and_refuses_login_without_it(self):
        b = self.tmp / 'bin'
        b.mkdir()
        for t in ('ssh', 'ssh-add', 'ssh-keygen', 'pgrep'):   # stubs; ssh -G prints no proxycommand
            (b / t).write_text('#!/bin/sh\nexit 0\n')
            (b / t).chmod(0o755)
        os.environ['PATH'] = str(b)
        self.assertEqual(auth.missing_tools(), ['step'])
        r = auth.start_login(wait_s=0)
        self.assertEqual(r['login'], 'not started')
        self.assertIn('brew install step', r['error'])

    def test_wireproxy_mode_needs_wireproxy(self):
        os.environ['MUSICA_VPN'] = 'wireproxy'
        os.environ['PATH'] = '/nonexistent'
        self.assertIn('wireproxy', auth.missing_tools())

    def test_mac_without_gnu_tools(self):
        from unittest import mock
        with mock.patch.object(auth.sys, 'platform', 'darwin'), mock.patch.object(auth.Path, 'exists', return_value=False):
            self.assertIn('gnu-tools', auth.missing_tools())
            self.assertIn('brew install coreutils findutils gnu-tar', auth._missing_note(['gnu-tools']))
