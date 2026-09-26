"""Tiny local recorder for code-switching test audio.

    python tools/recorder/server.py        # then open http://localhost:8765 in your browser

Serves index.html and saves POSTed recordings (16 kHz mono WAV + a JSON of sentence timestamps)
into data/user_recordings/. Binds to 127.0.0.1 only.
"""
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / "user_recordings"
NAME = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body=b"", ctype="text/plain"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/list":
            files = sorted(p.name for p in OUT.glob("*.wav")) if OUT.exists() else []
            return self._send(200, json.dumps(files).encode(), "application/json")
        self._send(404, b"not found")

    def do_POST(self):
        url = urlparse(self.path)
        name = parse_qs(url.query).get("name", [""])[0]
        if not NAME.match(name):
            return self._send(400, b"bad name")
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0 or n > 200 * 1024 * 1024:
            return self._send(400, b"bad size")
        data = self.rfile.read(n)
        OUT.mkdir(parents=True, exist_ok=True)
        if url.path == "/save_wav":
            if data[:4] != b"RIFF":
                return self._send(400, b"not a wav")
            (OUT / f"{name}.wav").write_bytes(data)
        elif url.path == "/save_json":
            json.loads(data)                                   # must be valid JSON
            (OUT / f"{name}.json").write_bytes(data)
        else:
            return self._send(404, b"not found")
        print(f"saved {name} ({url.path[6:]}, {n} bytes)")
        self._send(200, b"ok")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"recorder: http://localhost:8765  ->  saving to {OUT}")
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
