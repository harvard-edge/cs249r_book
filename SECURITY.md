# Security Policy

Thank you for helping keep MLSysBook and its users safe.

This repository contains a textbook plus several actively-developed software
sub-projects. This document explains what we consider in-scope, how to report a
vulnerability privately, and what response you can expect.

## Reporting a vulnerability

> [!IMPORTANT]
> **Do not open a public GitHub issue for a security vulnerability.**
> Public reports give attackers a head start on users who haven't patched yet.

Use one of these private channels:

1. **Preferred — GitHub Private Vulnerability Report:**
   <https://github.com/harvard-edge/cs249r_book/security/advisories/new>
   (requires a free GitHub account; gives us a structured triage thread)
2. **Email:** `vj@eecs.harvard.edu` and `nkhoshnevis@g.harvard.edu`.
   Please include "MLSysBook security" in the subject line.

Please include:

- A description of the issue and the impact you believe it has
- Steps to reproduce, or a proof-of-concept
- The affected sub-project and version / commit SHA
- Whether the issue is already public anywhere

## Response expectations

We are an academic open-source project, not a product team with a 24/7 rotation.
That said, we take security reports seriously:

| Stage | Target |
|---|---|
| Initial acknowledgement | within **5 business days** |
| Triage and severity assessment | within **10 business days** |
| Fix or mitigation plan | scoped to severity; critical issues prioritized |
| Public disclosure | coordinated with reporter; default 90-day embargo |

If you do not hear back within 5 business days, please escalate by re-sending
the email and CC'ing both addresses above.

## In-scope assets

These are the components where a vulnerability report makes sense:

| Component | Type | Why it's in scope |
|---|---|---|
| **`interviews/staffml/`** | Public Next.js web app | Serves StaffML to users; auth, data integrity, XSS, IDOR, etc. |
| **`interviews/staffml/worker/`, `staffml-vault-worker/`** | Cloudflare Workers (public API) | Internet-exposed API endpoints |
| **`interviews/vault-cli/`** | Installable Python CLI | Code-execution risk for vault authors |
| **`tinytorch/`** | Installable Python package | Distributed via pip; supply-chain and code-execution risk |
| **`mlsysim/`** | Installable Python package | Same as above |
| **`mlperf-edu/`** | Installable Python package | Same as above |
| **`book/vscode-ext/`, `tinytorch/vscode-ext/`, `labs/vscode-ext/`, `kits/vscode-ext/`, `mlsysim/vscode-ext/`** | VSCode extensions | Run inside the user's editor; code-execution risk |
| **`labs/`** | WASM-based browser labs | Same-origin / sandbox-escape risk |
| **GitHub Actions workflows in `.github/workflows/`** | CI configuration | Token leakage, supply-chain injection |
| **`site/newsletter/`** | Newsletter pipeline | Subscriber data handling |

## Out of scope

These are **not** security issues — please use a regular bug report instead:

- Typos, factual errors, or pedagogical complaints in the textbook
- Broken links in chapters, slides, or the docs site
- Build failures of the Quarto book (`quarto render`)
- Stylistic issues with figures, tables, or callouts
- Findings against third-party services we link to (e.g. OpenReview, Google
  Scholar) — report those upstream
- Vulnerabilities in dependencies that are already public and have a fix
  available — please open a regular PR bumping the version
- "Best practice" hardening recommendations without a concrete attack scenario
  (we welcome these, but as PRs/discussions, not security advisories)

## Safe-harbor for good-faith research

We will not pursue legal action against researchers who:

- Act in good faith to identify and report vulnerabilities
- Avoid privacy violations, data destruction, and service degradation
- Give us reasonable time to remediate before public disclosure
- Do not exploit a vulnerability beyond what is necessary to demonstrate it

## Credit

Reporters who follow this policy will be credited in the published security
advisory and in release notes, unless they request anonymity.

---

*Last updated: 2026.*
