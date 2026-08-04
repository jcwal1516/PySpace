# PySpace security best-practices review

Reviewed: 2026-08-03

Scope: Python source, CLI, result serialization, release scripts, dependency
declarations, and planned GitHub automation for PySpace 0.1.0.

## Executive summary

No confirmed critical or high-severity vulnerability was found. PySpace is a
local scientific library without a listening service, authentication surface,
or credential store. The most important code boundary—loading analysis
results—uses a versioned JSON/CSV/non-object-NPZ directory format and explicitly
rejects pickle. External commands use fixed executable/argument arrays without
a shell, and imports do not create directories, probe hardware, configure
logging, or mutate random/plotting state.

The private disclosure target is fixed to the public PySpace repository and is
enabled as part of repository publication. No open code-security finding
remains from this review.

## Findings

1. **SEC-001 — Informational — Resolved: private disclosure target.**
   `SECURITY.md` links directly to the repository's private advisory form.
   Private vulnerability reporting is enabled during repository publication;
   public issues are not the disclosure channel.

2. **SEC-002 — Informational — Resolved: unsafe object deserialization.**
   `src/pyspace/serialization.py` stores validated JSON metadata, CSV tables,
   and arrays loaded with `allow_pickle=False`; object arrays, unknown schema
   versions, missing members, traversal outside the bundle, existing output
   destinations, out-of-bundle CSV or NPZ symlinks, and CLI pickle paths are
   rejected. Regression tests cover the boundary. Dataclass tags are treated as
   inert metadata and never imported.

3. **SEC-003 — Informational — Resolved: command and upstream-source trust.**
   `src/pyspace/parity.py` and test tooling call subprocesses with argument
   lists and no shell. Live parity accepts only the pinned SPACE commit and an
   empty Git status. Network cloning is an explicit parity operation, not an
   import-time action.

4. **SEC-004 — Informational — Mitigated: dependency and build supply chain.**
   Runtime and optional dependencies use bounded minimum declarations; optional
   copyleft community backends are separated and noticed. CI runs `pip-audit`,
   Gitleaks, CodeQL, public-tree/data checks, distribution inspection, and a
   clean-wheel smoke test. Workflow actions are pinned to full commit hashes.
   `DEPENDENCY_INVENTORY.json` captures the exact local candidate environment;
   release operators must review changes on every build.

5. **SEC-005 — Informational — Accepted local-use risk: resource exhaustion.**
   Large valid microscopy images and tables can legitimately consume substantial
   memory and CPU, so the library does not impose an arbitrary input-size cap.
   The source-tree gate limits committed artifacts, resource inspection is
   opt-in, and callers processing untrusted files should enforce quotas before
   invoking PySpace. This becomes a higher-severity issue if the library is ever
   exposed directly through a multi-user network service.

## Verification expectations

The release gate is `make audit` plus `make build`; exact executed results are
recorded separately in `RELEASE_VERIFICATION.md`. GitHub-only scanners are
reported separately after they run in the hosted repository.
