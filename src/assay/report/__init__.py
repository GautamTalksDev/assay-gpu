"""Attestation generation and offline verification. Local files only."""

from assay.report.attestation import sign_document, write_attestation
from assay.report.constants import SPEC_ID
from assay.report.verify import VerifyOutcome, verify_path

__all__ = [
    "SPEC_ID",
    "VerifyOutcome",
    "sign_document",
    "verify_path",
    "write_attestation",
]
