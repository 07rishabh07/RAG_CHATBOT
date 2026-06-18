import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from .rag import answer_question, build_rag
except ImportError:
    from rag import answer_question, build_rag

BASE_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8000

CHUNKS, INDEX = build_rag()


class RAGRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        if self.path != "/ask":
            self.send_error(404, "Not found")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_body.decode("utf-8"))
            question = str(payload.get("question", "")).strip()
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON"}, status=400)
            return

        if not question:
            self.send_json({"error": "Question is required"}, status=400)
            return

        self.send_json(answer_question(question, CHUNKS, INDEX))

    def send_json(self, payload, status=200):
        response = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response)


def main():
    server = ThreadingHTTPServer((HOST, PORT), RAGRequestHandler)
    print(f"RAG chatbot running at http://{HOST}:{PORT}", flush=True)
    print("Press Ctrl+C to stop the server.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
