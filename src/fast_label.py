"""Keyboard-driven labelling page: a faster alternative to the Streamlit app, writing the same files.

Run:  python src/fast_label.py            then open http://localhost:8502
      http://localhost:8502/?task=relabel  for the blind consistency re-label
Keys: 1-9 intent · Y/N needs a human · A-F reason · U unsure · Enter save & next · arrow keys move
"""
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from labels_io import CANDIDATES, CODEBOOK, GOLDEN, MIN_LABELS_BEFORE_RELABEL, RELABEL, read_csv, relabel_items, upsert

PORT = 8502
PAGE = Path(__file__).with_name("fast_label.html")
OUTPUT = {"golden": GOLDEN, "relabel": RELABEL}

codebook = json.loads(CODEBOOK.read_text(encoding="utf-8"))
INTENT_IDS = [i["id"] for i in codebook["intents"]]
REASON_IDS = [r["id"] for r in codebook["escalation_reasons"]]
ALWAYS_ESCALATE = set(codebook.get("always_escalate_intents", []))


def task_items(task):
    return read_csv(CANDIDATES, "conv_id") if task == "golden" else relabel_items()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _task(self, url):
        task = parse_qs(url.query).get("task", ["golden"])[0]
        return task if task in OUTPUT else None

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            return self._send(200, PAGE.read_bytes(), "text/html")
        if url.path != "/api/state" or not (task := self._task(url)):
            return self._send(404, {"error": "not found"})
        items = task_items(task)
        if items is None:
            return self._send(200, {"error": f"Finish at least {MIN_LABELS_BEFORE_RELABEL} golden labels before the re-label check."})
        labels = read_csv(OUTPUT[task], "conv_id")
        self._send(200, {
            "task": task,
            "intents": [{"id": i["id"], "name": i["name"], "definition": i["definition"]} for i in codebook["intents"]],
            "reasons": codebook["escalation_reasons"],
            "rules": codebook["labelling_rules"],
            "always_escalate": sorted(ALWAYS_ESCALATE),
            "items": items[["conv_id", "context", "customer_text"]].to_dict("records"),
            "labels": labels.set_index("conv_id").to_dict("index") if len(labels) else {},
        })

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/label" or not (task := self._task(url)):
            return self._send(404, {"error": "not found"})
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        escalate = body.get("should_escalate")
        reason = body.get("escalation_reason") if escalate == "yes" else "none"
        if body.get("intent") not in INTENT_IDS or escalate not in ("yes", "no") or (escalate == "yes" and reason not in REASON_IDS):
            return self._send(400, {"error": "invalid label"})
        if body["intent"] in ALWAYS_ESCALATE and escalate != "yes":
            return self._send(400, {"error": "Codebook rule: this intent always needs a human. Press Y and pick a reason."})
        upsert(OUTPUT[task], "conv_id", {
            "conv_id": str(body["conv_id"]), "intent": body["intent"], "should_escalate": escalate,
            "escalation_reason": reason, "unsure": bool(body.get("unsure")), "notes": str(body.get("notes", "")),
            "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        self._send(200, {"ok": True, "count": len(read_csv(OUTPUT[task], "conv_id"))})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"Labelling page: http://localhost:{PORT}  (re-label: http://localhost:{PORT}/?task=relabel)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
