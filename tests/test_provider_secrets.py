# Higgsfield key from AWS Secrets Manager, with botocore's Stubber (no AWS calls).

import json
import logging
from collections.abc import Iterator

import boto3
import pytest
from botocore import UNSIGNED
from botocore.config import Config
from botocore.stub import Stubber
from mypy_boto3_secretsmanager import SecretsManagerClient
from pydantic import SecretStr

from app.core.config import Settings, get_settings
from app.services.providers.higgsfield import (
    ProviderSecretError,
    higgsfield_credentials,
    with_higgsfield_key,
)
from app.workers.connector import WorkerConfigError, load_provider_secrets

SECRET_ID = "nrc/stg/higgsfield"  # noqa: S105 (a secret name, not a value)
KEY = "kid-STUBBED:ksecret-STUBBED-VALUE"


@pytest.fixture
def client() -> SecretsManagerClient:
    # Unsigned, so no credentials are looked up; the Stubber answers every call.
    return boto3.Session().client(
        "secretsmanager", region_name="us-east-1", config=Config(signature_version=UNSIGNED)
    )


@pytest.fixture
def stubber(client: SecretsManagerClient) -> Iterator[Stubber]:
    with Stubber(client) as stub:
        yield stub
        stub.assert_no_pending_responses()


def _settings(**overrides: object) -> Settings:
    # An empty value, as an empty HIGGSFIELD_API_KEY= line in .env gives, counts as not set.
    values: dict[str, object] = {
        "higgsfield_secret_id": SECRET_ID,
        "higgsfield_api_key": SecretStr(""),
    }
    values.update(overrides)
    return get_settings().model_copy(update=values)


def _answer(stubber: Stubber, secret: object) -> None:
    stubber.add_response(
        "get_secret_value",
        {"Name": SECRET_ID, "SecretString": json.dumps(secret)},
        {"SecretId": SECRET_ID},
    )


def test_key_is_read_from_the_secret(
    client: SecretsManagerClient, stubber: Stubber, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    _answer(stubber, {"HIGGSFIELD_API_KEY": KEY, "other": "x"})

    loaded = with_higgsfield_key(_settings(), client)

    key_id, key_secret = higgsfield_credentials(loaded)
    assert (key_id.get_secret_value(), key_secret.get_secret_value()) == (
        "kid-STUBBED",
        "ksecret-STUBBED-VALUE",
    )
    assert "STUBBED" not in caplog.text
    assert "STUBBED" not in repr(loaded)


def test_legacy_entry_name_is_no_longer_read(
    client: SecretsManagerClient, stubber: Stubber
) -> None:
    _answer(stubber, {"HF_KEY": KEY})

    with pytest.raises(ProviderSecretError, match="no HIGGSFIELD_API_KEY entry"):
        with_higgsfield_key(_settings(), client)


def test_secret_is_not_read_when_the_key_is_set(
    client: SecretsManagerClient, stubber: Stubber
) -> None:
    settings = _settings(higgsfield_api_key=SecretStr("local:key"))

    assert with_higgsfield_key(settings, client) is settings


def test_secret_is_not_read_without_a_secret_id(client: SecretsManagerClient) -> None:
    settings = _settings(higgsfield_secret_id=None)

    assert load_provider_secrets(settings) is settings


@pytest.mark.parametrize(
    "secret",
    [
        {"OTHER": KEY},
        {"HIGGSFIELD_API_KEY": ""},
        {"HIGGSFIELD_API_KEY": 123},
        ["not", "an", "object"],
    ],
    ids=["missing", "empty", "not-a-string", "not-an-object"],
)
def test_unusable_secret_is_refused_without_echoing_it(
    client: SecretsManagerClient, stubber: Stubber, secret: object
) -> None:
    _answer(stubber, secret)

    with pytest.raises(ProviderSecretError) as refused:
        with_higgsfield_key(_settings(), client)

    assert SECRET_ID in str(refused.value)
    assert "STUBBED" not in str(refused.value)


def test_secret_that_is_not_json_is_refused(client: SecretsManagerClient, stubber: Stubber) -> None:
    stubber.add_response("get_secret_value", {"SecretString": f"{KEY} not json"})

    with pytest.raises(ProviderSecretError, match="not a JSON object") as refused:
        with_higgsfield_key(_settings(), client)

    assert refused.value.__cause__ is None
    assert "STUBBED" not in str(refused.value)


def test_aws_error_is_refused_without_details(
    client: SecretsManagerClient, stubber: Stubber
) -> None:
    stubber.add_client_error(
        "get_secret_value", "AccessDeniedException", "role arn:aws:iam::123:role/x denied"
    )

    with pytest.raises(ProviderSecretError) as refused:
        with_higgsfield_key(_settings(), client)

    assert str(refused.value) == f"Could not read secret {SECRET_ID}."
    assert refused.value.__cause__ is None


def test_worker_refuses_to_start_when_the_secret_is_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreadable(settings: Settings) -> Settings:
        raise ProviderSecretError(f"Could not read secret {SECRET_ID}.")

    monkeypatch.setattr("app.workers.connector.with_higgsfield_key", unreadable)

    with pytest.raises(WorkerConfigError, match=SECRET_ID):
        load_provider_secrets(_settings())
