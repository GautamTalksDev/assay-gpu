"""Ed25519 key files and signatures. Local files only. No network."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from assay.report.canonical import canonical_dumps
from assay.report.constants import SIGNATURE_ALG

_HEX_PRIV = 64
_HEX_PUB = 64
_HEX_SIG = 128


@dataclass(frozen=True, slots=True)
class KeyPair:
    private_key_hex: str
    public_key_hex: str


def _hex_of(data: bytes, length: int) -> str:
    text = data.hex()
    if len(text) != length:
        msg = f"expected {length} hex chars, got {len(text)}"
        raise ValueError(msg)
    return text


def _bytes_hex(text: str, length: int) -> bytes:
    raw = text.strip().lower()
    if len(raw) != length:
        msg = f"expected {length} hex chars, got {len(raw)}"
        raise ValueError(msg)
    try:
        return bytes.fromhex(raw)
    except ValueError as exc:
        msg = "value is not hexadecimal"
        raise ValueError(msg) from exc


def generate_keypair() -> KeyPair:
    private = Ed25519PrivateKey.generate()
    private_bytes = private.private_bytes(
        Encoding.Raw, PrivateFormat.Raw, NoEncryption()
    )
    public_bytes = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return KeyPair(
        private_key_hex=_hex_of(private_bytes, _HEX_PRIV),
        public_key_hex=_hex_of(public_bytes, _HEX_PUB),
    )


def write_keypair(path: Path, pair: KeyPair) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "crv": "Ed25519",
        "private_key_hex": pair.private_key_hex,
        "public_key_hex": pair.public_key_hex,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o600)


def load_keypair(path: Path) -> KeyPair:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("crv") != "Ed25519":
        msg = "key file crv must be Ed25519"
        raise ValueError(msg)
    private_hex = str(payload["private_key_hex"]).lower()
    public_hex = str(payload["public_key_hex"]).lower()
    derived = (
        Ed25519PrivateKey.from_private_bytes(_bytes_hex(private_hex, _HEX_PRIV))
        .public_key()
        .public_bytes(Encoding.Raw, PublicFormat.Raw)
    )
    if derived != _bytes_hex(public_hex, _HEX_PUB):
        msg = "public_key_hex does not match private_key_hex"
        raise ValueError(msg)
    return KeyPair(private_key_hex=private_hex, public_key_hex=public_hex)


def load_public_key_hex(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("{"):
        payload = json.loads(text)
        hex_text = str(payload["public_key_hex"]).lower()
    else:
        hex_text = text.split()[0].lower()
    _bytes_hex(hex_text, _HEX_PUB)
    return hex_text


def sign_bytes(message: bytes, pair: KeyPair) -> str:
    private = Ed25519PrivateKey.from_private_bytes(
        _bytes_hex(pair.private_key_hex, _HEX_PRIV)
    )
    return _hex_of(private.sign(message), _HEX_SIG)


def verify_bytes(message: bytes, signature_hex: str, public_key_hex: str) -> bool:
    public = Ed25519PublicKey.from_public_bytes(_bytes_hex(public_key_hex, _HEX_PUB))
    try:
        public.verify(_bytes_hex(signature_hex, _HEX_SIG), message)
    except (InvalidSignature, ValueError):
        return False
    return True


def signature_envelope(
    body: dict[str, object], pair: KeyPair, *, mode: str
) -> dict[str, str]:
    message = canonical_dumps(body).encode("ascii")
    return {
        "alg": SIGNATURE_ALG,
        "mode": mode,
        "public_key_hex": pair.public_key_hex,
        "signature_hex": sign_bytes(message, pair),
    }
