"""The stdio transport: a bare `ping` (Crush, Go MCP SDK) must not get INVALID_PARAMS on a 2026-07-28 connection."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = {'io.modelcontextprotocol/protocolVersion': '2026-07-28', 'io.modelcontextprotocol/clientCapabilities': {}}
METHOD_NOT_FOUND, INVALID_PARAMS = -32601, -32602


class StdioServer:
    def __init__(self):
        env = dict(os.environ, AMADEUS_SOUND='off', AMADEUS_ISSUES='off', AMADEUS_SSH='false')
        self.p = subprocess.Popen([sys.executable, '-m', 'amadeus.server'], cwd=ROOT, env=env, text=True,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.n = 0

    def send(self, method, params=None, notify=False):
        msg = {'jsonrpc': '2.0', 'method': method}
        if not notify:
            self.n += 1
            msg['id'] = self.n
        if params is not None:
            msg['params'] = params
        self.p.stdin.write(json.dumps(msg) + '\n')
        self.p.stdin.flush()
        return None if notify else json.loads(self.p.stdout.readline())

    def close(self):
        self.p.stdin.close()
        self.p.stdout.close()
        try:
            self.p.wait(10)
        except subprocess.TimeoutExpired:
            self.p.kill()


class BarePingTest(unittest.TestCase):
    def setUp(self):
        self.s = StdioServer()
        self.addCleanup(self.s.close)

    def test_modern_bare_ping_gets_method_not_found(self):
        self.assertIn('2026-07-28', self.s.send('server/discover', {'_meta': META})['result']['supportedVersions'])
        for params in (None, {}, {'_meta': {}}):
            r = self.s.send('ping', params)
            self.assertEqual(r.get('error', {}).get('code'), METHOD_NOT_FOUND, (params, r))
        tools = self.s.send('tools/list', {'_meta': META})['result']['tools']   # the connection still works
        self.assertIn('access_status', {t['name'] for t in tools})

    def test_modern_other_bare_request_still_rejected(self):
        self.s.send('server/discover', {'_meta': META})
        self.assertEqual(self.s.send('tools/list')['error']['code'], INVALID_PARAMS)

    def test_legacy_ping_unchanged(self):
        r = self.s.send('initialize', {'protocolVersion': '2025-11-25', 'capabilities': {},
                                       'clientInfo': {'name': 'test', 'version': '0'}})
        self.assertIn('result', r)
        self.s.send('notifications/initialized', notify=True)
        self.assertEqual(self.s.send('ping'), {'jsonrpc': '2.0', 'id': self.s.n, 'result': {}})


if __name__ == '__main__':
    unittest.main()
