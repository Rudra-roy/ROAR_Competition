"""Preview docs/ the way GitHub Pages will serve it: under /ROAR_Competition/, with HTTP range requests (Safari will not
play a video without them) and the site's own 404.html. Development tool; nothing in the submission uses it.

  python tools/site_serve.py [--host 127.0.0.1] [--port 8090]

Bind --host to one address (for example a Tailscale IP) rather than 0.0.0.0, so only that network can reach it.
"""
import argparse
import http.server
import os
import re

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
PREFIX = "/ROAR_Competition/"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=DOCS, **k)

    def translate_path(self, path):
        path = path.split("?", 1)[0].split("#", 1)[0]
        if path in ("/", PREFIX.rstrip("/")):
            return os.path.join(DOCS, "index.html")
        if path.startswith(PREFIX):
            path = "/" + path[len(PREFIX):]
        else:
            return os.path.join(DOCS, "__outside_prefix__")  # Pages would 404 here
        return super().translate_path(path)

    def send_error(self, code, message=None, explain=None):
        if code == 404 and os.path.exists(os.path.join(DOCS, "404.html")):
            body = open(os.path.join(DOCS, "404.html"), "rb").read()
            self.send_response(404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
            return
        super().send_error(code, message, explain)

    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
        size = os.path.getsize(path)
        if not m or not (m.group(1) or m.group(2)):
            return super().send_head()
        a = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
        b = min(int(m.group(2)), size - 1) if m.group(1) and m.group(2) else size - 1
        if a >= size or a > b:
            self.send_response(416)
            self.send_header("Content-Range", "bytes */%d" % size)
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(a)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", "bytes %d-%d/%d" % (a, b, size))
        self.send_header("Content-Length", str(b - a + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._left = b - a + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "_left", None)
        self._left = None
        if left is None:
            return super().copyfile(source, outputfile)
        while left > 0:
            chunk = source.read(min(65536, left))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    srv = http.server.ThreadingHTTPServer((a.host, a.port), Handler)
    print("serving docs/ at http://%s:%d%s" % (a.host, a.port, PREFIX), flush=True)
    srv.serve_forever()
