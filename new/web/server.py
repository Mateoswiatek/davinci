#!/usr/bin/env python3
"""Simple HTTP server to host the VR Stream Viewer."""

import http.server
import socketserver
import os

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)


def main():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serwer działa na http://0.0.0.0:{PORT}")
        print(f"Serwuję pliki z: {DIRECTORY}")
        print("Naciśnij Ctrl+C aby zatrzymać")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nZatrzymano serwer")


if __name__ == "__main__":
    main()