"""python -m hunch            serve (HUNCH_HOST / HUNCH_PORT, default 127.0.0.1:8791)
python -m hunch judge JSON one judgment with the caller's LLM, no server: JSON (or - for stdin) is
                           {"context": ..., "checks": {...}, "images"?: [...], "model"?: ..., "effort"?: ...}
python -m hunch mcp        stdio MCP server exposing `judge` (the agent starts it; no separate server)
python -m hunch selftest   quick pre-flight check against the configured backend (exit 0 = pass)
python -m hunch qualify    full qualification of a model as a Hunch backend (exit 0 = qualified)
"""
import ipaddress
import os
import sys


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def serve() -> int:
    import uvicorn

    from .config import load_settings
    from .server import create_app

    host = os.environ.get("HUNCH_HOST", "127.0.0.1")
    settings = load_settings()
    if not settings.models:
        # zero-config: the LLM the agent already uses (see hunch.config.discover_settings)
        from .config import DiscoveryError, discover_settings
        try:
            settings = discover_settings()
        except DiscoveryError as e:
            print(f"no models configured and none discovered: {e}\n"
                  "create hunch.toml (see hunch.toml.example) or set HUNCH_BACKEND_URL", file=sys.stderr)
            return 2
        spec = settings.resolve(None)
        print(f"hunch: using {spec.backend_model} at {settings.backend_url} (mode auto)", file=sys.stderr)
    if not _is_loopback(host) and not settings.api_keys and os.environ.get("HUNCH_ALLOW_NOAUTH") != "1":
        print(f"refusing to listen on {host} without HUNCH_API_KEYS "
              "(set keys, or HUNCH_ALLOW_NOAUTH=1 if a proxy in front enforces auth)", file=sys.stderr)
        return 2
    uvicorn.run(create_app(settings), host=host, port=int(os.environ.get("HUNCH_PORT", "8791")),
                log_level=os.environ.get("HUNCH_LOG", "info"))
    return 0


def judge_cli(args: list[str]) -> int:
    import json

    from .client import judge
    from .engine import HunchError

    raw = sys.stdin.read() if not args or args[0] == "-" else args[0]
    try:
        req = json.loads(raw)
        out = judge(req.get("context", ""), req.get("checks"), images=req.get("images"),
                    model=req.get("model"), effort=req.get("effort"))
    except json.JSONDecodeError as e:
        print(json.dumps({"error": {"code": "invalid_request", "message": f"not JSON: {e}"}}))
        return 2
    except HunchError as e:
        print(json.dumps({"error": {"code": e.code, "message": e.message}}))
        return 1
    print(json.dumps(out))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd, rest = (argv[0], argv[1:]) if argv else ("serve", [])
    if cmd == "selftest":
        from .selftest import main as run
        return run(rest)
    if cmd == "qualify":
        from .qualify import main as run
        return run(rest)
    if cmd == "judge":
        return judge_cli(rest)
    if cmd == "mcp":
        from .mcp import serve as run
        return run()
    if cmd in ("serve",):
        return serve()
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
