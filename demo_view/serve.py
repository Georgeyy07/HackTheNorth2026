"""Lightweight optional standalone server for demo_view."""
import argparse
import http.server
import socketserver
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DEMO_DIR), **kwargs)

def main():
    parser = argparse.ArgumentParser(description="Serve demo_view locally.")
    parser.add_argument("--port", default=8888, type=int)
    args = parser.parse_args()

    with socketserver.TCPServer(("", args.port), Handler) as httpd:
        print(f"🌲 RoadScope Demo View serving at: http://localhost:{args.port}/")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")

if __name__ == "__main__":
    main()
