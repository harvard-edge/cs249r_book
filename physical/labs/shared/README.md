# Shared kit assets

Reusable MPU/MCU contracts and helpers for the Physical AI Kit.

| Path | Purpose |
| --- | --- |
| `mpu/` | Linux-side libraries: logging, clock sync helpers, intent emitters |
| `mcu/` | Real-time-side libraries: watchdogs, limit checks, safe-state routines |
| `contracts/` | Stable message schemas (observation, intent, permit/refuse, health) |
| `instrumentation/` | Timestamp correlation, evidence-record helpers, energy/latency hooks |

Keep schemas versioned. A lab may extend a contract; it must not silently break
earlier chapter checkpoints.
