"""Stand-in for a vLLM server: /health, and /v1/chat/completions with the API key checked."""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, KEY = int(sys.argv[1]), sys.argv[2]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply(200 if self.path == '/health' else 404, {})

    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', 0)))
        if self.headers.get('Authorization') != f'Bearer {KEY}':
            return self.reply(401, {'error': 'bad key'})
        self.reply(200, {'choices': [{'message': {'content': f'hi from {PORT}'}}]})


srv = ThreadingHTTPServer(('127.0.0.1', PORT), H)
print('INFO:     Application startup complete.', flush=True)
srv.serve_forever()
