"""
Content-Length framing for MCP JSON-RPC over stdio.

This is transport only. It does not call a model and it does not know any tool names.
"""
import json
import sys


def encode_message(payload: dict) -> bytes:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _read_exact(stream, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        piece = stream.read(remaining)
        if not piece:
            raise EOFError("MCP stream closed before the body finished")
        chunks.append(piece)
        remaining -= len(piece)
    return b"".join(chunks)


def read_message(stream):
    """Read one JSON-RPC message. Returns None at clean EOF."""
    header = b""
    while b"\r\n\r\n" not in header:
        piece = stream.read(1)
        if not piece:
            return None if header == b"" else (_ for _ in ()).throw(EOFError("truncated MCP header"))
        header += piece
        if len(header) > 65536:
            raise ValueError("MCP header is too large")
    raw_header, leftover = header.split(b"\r\n\r\n", 1)
    length = None
    for line in raw_header.decode("ascii").split("\r\n"):
        if line.lower().startswith("content-length:"):
            length = int(line.split(":", 1)[1].strip())
    if length is None:
        raise ValueError(f"MCP header has no Content-Length: {raw_header!r}")
    body = leftover
    if len(body) < length:
        body += _read_exact(stream, length - len(body))
    return json.loads(body[:length].decode("utf-8"))


def write_message(stream, payload: dict) -> None:
    stream.write(encode_message(payload))
    stream.flush()


def serve(handler) -> None:
    """Read requests from stdin and write responses to stdout. Logs stay off stdout."""
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while True:
        try:
            message = read_message(stdin)
        except EOFError:
            return
        if message is None:
            return
        method = message.get("method")
        if method == "notifications/initialized":
            continue
        response = handler(message)
        if response is not None:
            write_message(stdout, response)
