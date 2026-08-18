"""attestation-v1 constants. Disclaimer text is part of the signed body."""

SPEC_ID = "attestation-v1"
SIGNATURE_ALG = "Ed25519"
MODE_SELF_SIGNED = "self-signed"
MODE_ASSAY_SIGNED = "assay-signed"

SELF_SIGNED_DISCLAIMER = (
    "SELF-SIGNED. A provider grading its own hardware is not independent "
    "evidence. This signature proves the report bytes were not altered after "
    "signing. It does not prove an independent party observed the run."
)

ASSAY_SIGNED_NOTE = (
    "ASSAY-SIGNED. This mode is reserved for reports issued by an Assay "
    "project-observed run. v1 does not include an issuance service. A "
    "verifier must pin the Assay project public key with --pubkey; the "
    "embedded key alone does not establish independence."
)
