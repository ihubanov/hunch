"""A stdio MCP server exposing Hunch's `judge` as a tool, for agents that call tools rather than import
Python. The agent starts it on demand; it uses the agent's own LLM endpoint from the environment (see
hunch.client), so there is no separate server or settings file.

    claude mcp add hunch -- python -m hunch mcp          # Claude Code
    {"mcpServers": {"hunch": {"command": "python", "args": ["-m", "hunch", "mcp"]}}}

Implements the small part of MCP a tool server needs (initialize, tools/list, tools/call, ping) over
newline-delimited JSON-RPC 2.0, with no dependency beyond Hunch itself.
"""
from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from . import __version__
from .client import judge
from .engine import HunchError

PROTOCOL_VERSION = "2025-06-18"

JUDGE_TOOL = {
    "name": "judge",
    "title": "Calibrated judgment",
    "description": (
        "Ask one or more typed questions about some context (and optional images) and get calibrated "
        "probabilities back instead of text. Kinds: yesno -> p_yes; pick (1-300 options) -> pick, probs, "
        "confidence; scale (levels lowest first) -> value, probs, confidence. Name the look-alike case in "
        "no_if: it is worth 17-24 accuracy points. Act at p >= 0.9, review 0.5-0.9."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "context": {"description": "The data to judge: a string, or an object with named fields that "
                                       "questions refer to in backticks, e.g. `ticket.body`."},
            "checks": {"type": "object", "description": (
                "Your ids -> check. {kind: yesno, question, yes_if?, no_if?} | "
                "{kind: pick, question, options: {key: description|null}} | "
                "{kind: scale, question, levels: [lowest, ..., highest]}"),
                "additionalProperties": {"type": "object"}},
            "images": {"type": "array", "items": {"type": "string"}, "maxItems": 8,
                       "description": "Optional https:// or data:image/...;base64 URLs (vision models only), "
                                      "referred to as IMAGE 1..n."},
            "model": {"type": "string", "description": "Backend model id; default: the agent's configured model."},
            "effort": {"type": "string", "description": "Thinking effort for thinking models: low | high | max."},
        },
        "required": ["checks"],
    },
}


def _reply(out: TextIO, msg_id: Any, result: dict | None = None, error: dict | None = None) -> None:
    msg = {"jsonrpc": "2.0", "id": msg_id}
    msg.update({"error": error} if error else {"result": result})
    out.write(json.dumps(msg) + "\n")
    out.flush()


def handle(msg: dict) -> dict | None:
    """One JSON-RPC message -> its result (None for notifications). Raises ValueError for unknown methods."""
    method, params = msg.get("method"), msg.get("params") or {}
    if method == "initialize":
        return {"protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "hunch", "version": __version__}}
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": [JUDGE_TOOL]}
    if method == "tools/call":
        if params.get("name") != "judge":
            raise ValueError(f"unknown tool {params.get('name')!r}")
        args = params.get("arguments") or {}
        try:
            out = judge(args.get("context", ""), args.get("checks"), images=args.get("images"),
                        model=args.get("model"), effort=args.get("effort"))
        except HunchError as e:
            # a tool error the agent can read and act on, not a protocol error
            return {"isError": True, "content": [{"type": "text", "text": f"{e.code}: {e.message}"}]}
        return {"content": [{"type": "text", "text": json.dumps(out)}], "structuredContent": out}
    raise ValueError(f"method not found: {method}")


def serve(inp: TextIO = sys.stdin, out: TextIO = sys.stdout) -> int:
    for line in inp:
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _reply(out, None, error={"code": -32700, "message": "parse error"})
            continue
        if "id" not in msg:        # notification (e.g. notifications/initialized): no reply
            continue
        try:
            _reply(out, msg["id"], result=handle(msg))
        except ValueError as e:
            _reply(out, msg["id"], error={"code": -32601, "message": str(e)})
        except Exception as e:  # noqa: BLE001 - never let one bad call kill the agent's tool server
            _reply(out, msg["id"], error={"code": -32603, "message": f"{type(e).__name__}: {e}"})
    return 0
