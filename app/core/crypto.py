# Encryption of provider tokens at rest (AES-256-GCM). Only the API process holds the key.

import base64
import binascii
import os
import re
from typing import Final

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import SecretStr

_KEY_ID: Final = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_NONCE_BYTES: Final = 12
_FORMAT: Final = "v1"


class TokenKeyError(ValueError):
    pass


class TokenDecryptError(Exception):
    # The stored value was not written with this key for this record. Never carries the value.
    pass


class TokenCipher:
    """Encrypts a token bound to the record it belongs to.

    The associated data names the record (e.g. tenant and account id), so a ciphertext copied
    onto another row or tenant does not decrypt. Stored as "v1:<key-id>:<base64 nonce+ct>".
    """

    def __init__(self, key_spec: SecretStr) -> None:
        key_id, _, encoded = key_spec.get_secret_value().partition(":")
        if not _KEY_ID.fullmatch(key_id) or not encoded:
            raise TokenKeyError("TOKEN_ENCRYPTION_KEY must look like '<key-id>:<base64 key>'")
        try:
            key = base64.b64decode(encoded, validate=True)
        except binascii.Error as exc:
            raise TokenKeyError("TOKEN_ENCRYPTION_KEY is not valid base64") from exc
        if len(key) != 32:
            raise TokenKeyError("TOKEN_ENCRYPTION_KEY must decode to 32 bytes")
        self._key_id = key_id
        self._aead = AESGCM(key)

    def encrypt(self, plaintext: str, *, context: str) -> str:
        nonce = os.urandom(_NONCE_BYTES)
        sealed = self._aead.encrypt(nonce, plaintext.encode(), context.encode())
        return f"{_FORMAT}:{self._key_id}:{base64.b64encode(nonce + sealed).decode()}"

    def decrypt(self, stored: str, *, context: str) -> str:
        fmt, key_id, encoded = [*stored.split(":", 2), "", ""][:3]
        if fmt != _FORMAT or key_id != self._key_id:
            raise TokenDecryptError("token was encrypted with another key or format")
        try:
            raw = base64.b64decode(encoded, validate=True)
            nonce, sealed = raw[:_NONCE_BYTES], raw[_NONCE_BYTES:]
            return self._aead.decrypt(nonce, sealed, context.encode()).decode()
        except (binascii.Error, InvalidTag, ValueError) as exc:
            raise TokenDecryptError("token does not decrypt for this record") from exc
