#!/usr/bin/env python3
"""Write the planned F1 fault-record header and sign it (our run).

The header fills the fields of the chapter's planned F1 record (Adversarial
Verification, table 1 and section 15.4.2): identity fields, the oracle digest,
and outcome fields left "unavailable" because no trial has run. The oracle is
hashed with SHA-256 and the header signed with a fixed Ed25519 demonstration
key derived from a public string (not a test-cell key). Needs the `cryptography`
package. Writes data/f1_record.json and data/f1_record.txt.

Run from this folder: python3 make_record.py
"""

import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ONSET = 0.082
RESTRICTED_BOUND = 0.6296253647824702
V_RESTRICTED = 1.0
PAYLOAD = 50.0
MU_INSPECTED = 0.12
BUS_NOMINAL = 47.7

HERE = Path(__file__).resolve().parent

oracle = {
    "reference_event": "last_valid_renewal",
    "onset_ms_max": round(ONSET * 1e3),
    "travel_mm_max": round(RESTRICTED_BOUND * 1e3, 1),
    "pass_requires": ["preconditions_verified", "witnesses_complete",
                      "onset_margin>=0", "travel_margin>=0"],
}
oracle_bytes = json.dumps(oracle, sort_keys=True, separators=(",", ":")).encode()

header = {
    "trial_id": "F1-restricted-001",
    "claim": "rack-end stop within clear distance after proposer stall",
    "envelope": {"speed_mps": V_RESTRICTED, "payload_kg": round(PAYLOAD),
                 "floor_mu": MU_INSPECTED, "drive_bus_V": BUS_NOMINAL},
    "injection": {"mode": "kill chunk-policy task", "phase": "1 tick after renewal",
                  "timer_isr": "left running"},
    "oracle_sha256": hashlib.sha256(oracle_bytes).hexdigest(),
    "outcome": {"trials": "unavailable", "max_onset_ms": "unavailable",
                "max_travel_mm": "unavailable"},
    "verdict": None,
}
header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
# A fixed demonstration key, so reruns reproduce the slide's excerpt.
key = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(b"mlsysbook-l15-demo-key").digest())
sig = key.sign(header_bytes)
record = {"header": header, "oracle": oracle, "signature_ed25519": sig.hex()}
(HERE / "data" / "f1_record.json").write_text(json.dumps(record, indent=1) + "\n")

short = [
    '"trial_id": "F1-restricted-001",',
    f'"envelope": {{"speed_mps": {V_RESTRICTED}, "payload_kg": {round(PAYLOAD)}, "floor_mu": {MU_INSPECTED}}},',
    '"injection": {"mode": "kill chunk-policy task", "timer_isr": "left running"},',
    f'"oracle_sha256": "{header["oracle_sha256"][:24]}...",',
    '"outcome": {"trials": "unavailable", "max_onset_ms": "unavailable",',
    '            "max_travel_mm": "unavailable"},',
    '"verdict": null',
    f'signature_ed25519: {sig.hex()[:40]}...',
]
(HERE / "data" / "f1_record.txt").write_text("\n".join(short) + "\n")
print(json.dumps(oracle, indent=1))
print("\n".join(short))
