# MLPerf EDU Security Review

Last reviewed on July 18, 2026. This review covers the execution paths that
handle generated code, upstream pickle files, and historical runtimes. It
supports a controlled source-checkout preview. It does not approve an
unattended or multi-tenant service.

## Generated Code

HumanEval+ candidates run only inside the EvalPlus Docker sandbox. The runner
uses a digest-pinned Python base image and a byte-pinned EvalPlus source tree.
The container has no network, a read-only root filesystem, no Linux
capabilities, no-new-privileges, a host-user mapping, an init process, and
explicit CPU, memory, process, core-file, and open-file limits. Its temporary
filesystem is `noexec`, `nosuid`, and `nodev`. The dataset mount is read-only.
The writable workspace contains only the generated samples and evaluator
results, and the runner rejects any change to the copied sample input.

The run records the local image ID and performs a canonical-solution evaluator
self-check before scoring a candidate. The sandbox is appropriate for a
supervised single-user preview. It still trusts the local Docker daemon and a
locally built image. A production service would need an independently built,
scanned, signed, and digest-addressed evaluator image plus a separately
administered isolation boundary. Generated code must never execute directly on
the host.

## Upstream Pickle Files

EDM requires two upstream Python pickle files, the model checkpoint and the
official Inception detector. Their URLs, sizes, and SHA-256 values are fixed in
the asset catalog. The runner now hashes each already-open file immediately
before deserializing that same file descriptor. A mismatch fails before
`pickle.load` executes. It also checks the required checkpoint key and rejects
payloads that do not contain a loadable PyTorch module.

This control makes the preview accept only the known upstream bytes. It does
not make pickle a safe format for arbitrary input because deserialization can
execute code. The runner must never accept a user-selected pickle or a
user-selected digest. Production publication should convert the reviewed
objects to a non-executable format in an isolated build pipeline, or perform
deserialization in a separately sandboxed conversion job and publish the safe
derived artifact.

## Historical Runtimes

The MiniGo handoff requires an immutable `image@sha256` reference, a reviewed
professional-game input decision, pinned source files, a working GPU runtime,
and a complete environment preflight. The DLRM handoff requires the official
checkpoint digest, the licensed full dataset, and a prepared Python runtime.
Both paths fail closed when those requirements are absent.

Neither external runtime has been executed or security-qualified on its
required system during this local review. The DLRM Python process is a trusted
native research environment rather than a sandbox. The MiniGo image is
immutable, but its final least-privilege container policy still needs to be
tested against the historical TensorFlow and CUDA stack. Production approval
therefore requires the following work.

- Build both runtimes from reviewed source and publish immutable signed images
  with software bills of materials and vulnerability scan results.
- Run as a non-root user with no network, no capabilities,
  no-new-privileges, a read-only root, bounded resources, and explicit writable
  mounts wherever the historical workloads permit those controls.
- Separate licensed datasets, checkpoints, run outputs, and container build
  credentials, and document their retention and deletion policy.
- Exercise the hardened images on the required DLRM and MiniGo systems before
  calling either path production-ready.

## Decision

The current controls are sufficient for a supervised classroom or researcher
preview on a trusted single-user machine. Production remains blocked on safe
EDM artifact conversion, independently built and signed runtime images, legacy
environment qualification, vulnerability response, and authenticated release
provenance.
