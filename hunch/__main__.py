"""python -m hunch            serve (HUNCH_HOST / HUNCH_PORT, default 127.0.0.1:8791)
python -m hunch selftest   pre-flight check against the configured backend (exit 0 = pass)
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
        print("no models configured: create hunch.toml (see hunch.toml.example) or set HUNCH_BACKEND_MODEL", file=sys.stderr)
        return 2
    if not _is_loopback(host) and not settings.api_keys and os.environ.get("HUNCH_ALLOW_NOAUTH") != "1":
        print(f"refusing to listen on {host} without HUNCH_API_KEYS "
              "(set keys, or HUNCH_ALLOW_NOAUTH=1 if a proxy in front enforces auth)", file=sys.stderr)
        return 2
    uvicorn.run(create_app(settings), host=host, port=int(os.environ.get("HUNCH_PORT", "8791")),
                log_level=os.environ.get("HUNCH_LOG", "info"))
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["selftest"]:
        from .selftest import main
        sys.exit(main(sys.argv[2:]))
    sys.exit(serve())
