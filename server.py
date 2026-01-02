#!/usr/bin/env python3
"""
Simple HTTP server to serve the Experiment Viewer UI
"""
import http.server
import socketserver
import webbrowser
import os

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow loading local images
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

def main():
    handler = MyHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print("=" * 60)
        print(f"🚀 Experiment Viewer Server Starting...")
        print(f"📡 Server running at: http://localhost:{PORT}")
        print(f"📂 Serving from: {os.getcwd()}")
        print("=" * 60)
        print("\n✨ Opening browser...")
        print("\n⚠️  Press Ctrl+C to stop the server\n")
        
        # Open browser automatically
        webbrowser.open(f'http://localhost:{PORT}')
        
        # Serve forever
        httpd.serve_forever()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")





