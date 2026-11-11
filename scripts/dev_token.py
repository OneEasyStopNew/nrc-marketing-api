"""Local development tokens, for testing the API before Cognito is available.

Usage:
    uv run python -m scripts.dev_token init                  # once: key pair in .dev-keys/
    uv run python -m scripts.dev_token issue --sub dev-praneeth [--hours 12]

`init` writes a private key and a public JWKS file. Point DEV_JWKS_FILE at the JWKS file and
restart the API. `issue` prints an access token that the API then accepts. The token carries the
issuer, client ID and scope from the API's own settings, so every other check still applies.

The private key never leaves this machine. Never set DEV_JWKS_FILE in staging or production.
"""

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Final

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from app.core.config import get_settings

KEY_DIR: Final = Path(__file__).resolve().parents[1] / ".dev-keys"
PRIVATE_KEY: Final = KEY_DIR / "private.pem"
JWKS: Final = KEY_DIR / "jwks.json"
KID: Final = "dev-local-1"
MAX_HOURS: Final = 24
_SUB: Final = re.compile(r"[A-Za-z0-9-]{1,128}")


def _write_atomic(path: Path, content: bytes, mode: int) -> None:
    # Owner-only permissions from creation (where the OS supports them), then an atomic rename.
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.chmod(tmp, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def init(*, force: bool) -> Path:
    """Create the key pair. Returns the JWKS path to set as DEV_JWKS_FILE."""
    if PRIVATE_KEY.exists() and not force:
        raise SystemExit(f"{PRIVATE_KEY} already exists; pass --force to replace it.")
    KEY_DIR.mkdir(mode=0o700, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_jwk = RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
    jwks = {"keys": [{**public_jwk, "kid": KID, "alg": "RS256", "use": "sig"}]}
    _write_atomic(PRIVATE_KEY, pem, 0o600)
    _write_atomic(JWKS, json.dumps(jwks, indent=2).encode("utf-8"), 0o644)
    return JWKS


def issue(sub: str, *, hours: int) -> str:
    """Sign an access token for `sub` that matches the API's issuer, client and scope."""
    if not _SUB.fullmatch(sub):
        raise SystemExit(f"invalid --sub {sub!r}: use letters, digits and hyphens")
    if not 1 <= hours <= MAX_HOURS:
        raise SystemExit(f"--hours must be between 1 and {MAX_HOURS}")
    try:
        private_key = serialization.load_pem_private_key(PRIVATE_KEY.read_bytes(), password=None)
    except FileNotFoundError:
        raise SystemExit("No dev key yet: run `python -m scripts.dev_token init` first.") from None
    settings = get_settings()
    now = int(time.time())
    claims = {
        "sub": sub,
        "username": sub,
        "iss": settings.cognito_issuer,
        "client_id": settings.cognito_client_ids[0],
        "token_use": "access",
        "scope": settings.required_scope,
        "iat": now,
        "exp": now + hours * 3600,
    }
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": KID})  # type: ignore[arg-type]  # RSA key, checked by cryptography


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    commands = parser.add_subparsers(dest="command", required=True)
    init_cmd = commands.add_parser("init", help="create the local key pair")
    init_cmd.add_argument("--force", action="store_true", help="replace an existing key")
    issue_cmd = commands.add_parser("issue", help="print an access token")
    issue_cmd.add_argument("--sub", required=True, help="user ID; must match a seeded member")
    issue_cmd.add_argument("--hours", type=int, default=12, help=f"lifetime, 1-{MAX_HOURS}")
    args = parser.parse_args()

    if args.command == "init":
        path = init(force=args.force)
        print(f"Created {path}\nAdd to .env:  DEV_JWKS_FILE={path}\nThen restart the API.")
    else:
        print(issue(args.sub, hours=args.hours))


if __name__ == "__main__":
    main()
