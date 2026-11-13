"""Local dev tokens: accepted through the normal verification path, nothing weaker."""

from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import ValidationError

from app.auth.tokens import LocalJwks, verify_access_token
from app.core.config import Settings, get_settings
from app.core.errors import AuthenticationError
from scripts import dev_token


@pytest.fixture
def key_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / ".dev-keys"
    monkeypatch.setattr(dev_token, "KEY_DIR", folder)
    monkeypatch.setattr(dev_token, "PRIVATE_KEY", folder / "private.pem")
    monkeypatch.setattr(dev_token, "JWKS", folder / "jwks.json")


async def _verify(token: str, keys: LocalJwks) -> str:
    settings = get_settings()
    principal = await verify_access_token(
        token,
        keys,
        issuer=settings.cognito_issuer,
        client_ids=frozenset(settings.cognito_client_ids),
        required_scope=settings.required_scope,
    )
    return principal.sub


@pytest.mark.usefixtures("key_dir")
async def test_issued_token_passes_normal_verification() -> None:
    keys = LocalJwks.from_file(dev_token.init(force=False))

    sub = await _verify(dev_token.issue("dev-praneeth", hours=1), keys)

    assert sub == "dev-praneeth"


@pytest.mark.usefixtures("key_dir")
async def test_token_signed_by_another_key_is_rejected() -> None:
    keys = LocalJwks.from_file(dev_token.init(force=False))
    claims = jwt.decode(dev_token.issue("dev-x", hours=1), options={"verify_signature": False})
    forged = jwt.encode(
        claims,
        rsa.generate_private_key(public_exponent=65537, key_size=2048),
        algorithm="RS256",
        headers={"kid": dev_token.KID},
    )

    with pytest.raises(AuthenticationError, match="invalid or expired"):
        await _verify(forged, keys)


@pytest.mark.usefixtures("key_dir")
def test_init_refuses_to_overwrite_without_force() -> None:
    dev_token.init(force=False)

    with pytest.raises(SystemExit, match="already exists"):
        dev_token.init(force=False)


@pytest.mark.usefixtures("key_dir")
@pytest.mark.parametrize(
    ("sub", "hours", "message"),
    [("bad sub", 1, "invalid --sub"), ("dev-x", 0, "--hours"), ("dev-x", 25, "--hours")],
)
def test_issue_rejects_bad_input(sub: str, hours: int, message: str) -> None:
    dev_token.init(force=False)

    with pytest.raises(SystemExit, match=message):
        dev_token.issue(sub, hours=hours)


@pytest.mark.usefixtures("key_dir")
def test_init_keeps_the_existing_key_when_refusing() -> None:
    dev_token.init(force=False)
    original = dev_token.PRIVATE_KEY.read_bytes()

    with pytest.raises(SystemExit, match="already exists"):
        dev_token.init(force=False)

    assert dev_token.PRIVATE_KEY.read_bytes() == original
    assert not list(dev_token.KEY_DIR.glob("*.tmp"))


@pytest.mark.usefixtures("key_dir")
def test_init_with_force_replaces_the_key() -> None:
    dev_token.init(force=False)
    original = dev_token.PRIVATE_KEY.read_bytes()

    dev_token.init(force=True)

    assert dev_token.PRIVATE_KEY.read_bytes() != original


@pytest.mark.parametrize("app_env", ["staging", "production"])
@pytest.mark.parametrize("shortcut", [{"demo_stubs": True}, {"dev_jwks_file": Path("jwks.json")}])
def test_shortcuts_are_refused_outside_local_and_demo(
    app_env: str, shortcut: dict[str, object]
) -> None:
    with pytest.raises(ValidationError, match=f"APP_ENV='{app_env}'"):
        Settings.model_validate({"app_env": app_env, **shortcut})


@pytest.mark.parametrize("app_env", ["local", "demo"])
def test_shortcuts_are_allowed_locally(app_env: str) -> None:
    settings = Settings.model_validate(
        {"app_env": app_env, "demo_stubs": True, "dev_jwks_file": "jwks.json"}
    )

    assert settings.demo_stubs is True


def test_blank_dev_jwks_file_means_cognito() -> None:
    assert Settings.model_validate({"dev_jwks_file": "  "}).dev_jwks_file is None
