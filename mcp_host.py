"""
MCP host. Tool names come from tools/list. Nothing here is a fixed tool catalogue.

Adding a server is a config change. This module does not need to change with it.
The language model is called by mcp_agent.py, never by a server.
"""
import os
import subprocess
import sys
import threading
from pathlib import Path

from mcp_framing import read_message, write_message

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "mcp_config.json"


class McpSession:
    """One stdio session with one MCP server."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.name = spec["name"]
        self.trace = []
        command = spec.get("command") or "python"
        if command == "python":
            command = sys.executable
        env = os.environ.copy()
        env.update(spec.get("env") or {})
        self.process = subprocess.Popen(
            [command, *spec.get("args", [])],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self._next_id = 1
        self._stderr = []
        thread = threading.Thread(target=self._drain_stderr, daemon=True)
        thread.start()

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self._stderr.append(line.decode("utf-8", errors="replace"))

    def request(self, method: str, params: dict | None = None, notify: bool = False) -> dict | None:
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if not notify:
            message["id"] = self._next_id
            self._next_id += 1
        self.trace.append({"direction": "client->server", "body": message})
        write_message(self.process.stdin, message)
        if notify:
            return None
        while True:
            reply = read_message(self.process.stdout)
            if reply is None:
                raise EOFError(f"{self.name} closed the MCP stream")
            self.trace.append({"direction": "server->client", "body": reply})
            if reply.get("id") == message["id"]:
                if "error" in reply:
                    raise RuntimeError(f"{self.name} {method} failed: {reply['error']}")
                return reply

    def initialize(self) -> dict:
        reply = self.request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "recipe-agent", "version": "1.0.0"},
        })
        self.request("notifications/initialized", notify=True)
        return reply

    def list_tools(self) -> list[dict]:
        reply = self.request("tools/list")
        return (reply.get("result") or {}).get("tools") or []

    def list_resources(self) -> list[dict]:
        try:
            reply = self.request("resources/list")
        except RuntimeError:
            return []
        return (reply.get("result") or {}).get("resources") or []

    def read_resource(self, uri: str) -> str:
        reply = self.request("resources/read", {"uri": uri})
        contents = (reply.get("result") or {}).get("contents") or []
        return "\n".join(item.get("text") or "" for item in contents)

    def call_tool(self, name: str, arguments: dict) -> dict:
        reply = self.request("tools/call", {"name": name, "arguments": arguments or {}})
        return reply.get("result") or {}

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


class McpHost:
    """Opens every server in the config and routes tools/call by discovered name."""

    def __init__(self, config_path: Path = None, servers: list[dict] = None):
        if servers is None:
            import json
            path = Path(config_path or CONFIG_PATH)
            servers = json.loads(path.read_text(encoding="utf-8"))["servers"]
        self.servers = servers
        self.sessions = []
        self.tools = []
        self.by_name = {}

    def open(self) -> None:
        for spec in self.servers:
            session = McpSession(spec)
            session.initialize()
            self.sessions.append(session)

    def discover(self) -> list[dict]:
        """tools/list on each open server. Names are whatever the server advertised."""
        self.tools = []
        self.by_name = {}
        for session in self.sessions:
            for tool in session.list_tools():
                record = {
                    "server": session.name,
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "inputSchema": tool.get("inputSchema") or {},
                }
                self.tools.append(record)
                self.by_name[tool["name"]] = session
        return self.tools

    def resources_text(self) -> str:
        """App-attached context. The model is not asked to fetch this on every turn."""
        blocks = []
        for session in self.sessions:
            for resource in session.list_resources():
                text = session.read_resource(resource["uri"])
                if text:
                    blocks.append(text)
        return "\n\n".join(blocks)

    def call(self, name: str, arguments: dict) -> tuple[str, str, bool]:
        session = self.by_name.get(name)
        if session is None:
            known = ", ".join(tool["name"] for tool in self.tools) or "(none discovered)"
            return f"Unknown tool {name!r}. Discovered tools: {known}.", "", True
        result = session.call_tool(name, arguments)
        parts = []
        for item in result.get("content") or []:
            if item.get("text"):
                parts.append(item["text"])
        return "\n".join(parts), session.name, bool(result.get("isError"))

    def close(self) -> None:
        for session in self.sessions:
            session.close()
        self.sessions = []


def load_host(config_path: Path = None, servers: list[dict] = None) -> McpHost:
    host = McpHost(config_path=config_path, servers=servers)
    host.open()
    host.discover()
    return host
