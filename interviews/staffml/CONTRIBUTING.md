# Contributing to StaffML

Thank you for your interest in contributing. StaffML is open source under
[AGPL v3](LICENSE) and welcomes bug reports, feature requests, and code
contributions.

## Before your first code contribution

We require contributors to sign our [Contributor License Agreement (CLA)](CLA.md)
before we can merge any code change. **Please read the CLA before opening a pull
request.** Documentation typo fixes and tiny non-substantive changes are exempt;
when in doubt, ask in the PR.

The CLA preserves StaffML's ability to offer commercial licenses to organizations
that cannot accept AGPL's network-deployment terms. It does not transfer your
copyright — you continue to own your code and may use it for any other purpose.
The full rationale is at the bottom of the [CLA file](CLA.md).

## What we need

- **Bug reports** — open a [GitHub issue](https://github.com/harvard-edge/cs249r_book/issues)
  describing what you expected, what happened, and how to reproduce.
- **Feature requests** — open a discussion first so we can talk through fit
  before you spend time on implementation.
- **Pull requests** — keep them small and focused. One concern per PR. Tests and
  documentation appreciated where applicable.

## How to sign the CLA

Once a CLA bot is wired up to this repository, opening your first pull request
will trigger an automatic comment asking you to sign by replying:

```
I have read the CLA Document and I hereby sign the CLA
```

Until that bot is active, please add the following line to your pull request
description:

> I have read the StaffML CLA at [interviews/staffml/CLA.md](CLA.md) and agree
> to its terms for this and any future contributions to StaffML.

Your GitHub username, agreement date, and the SHA of the CLA file at the time
of signing will be recorded.

## Maintainer setup notes

To activate automated CLA collection, add the following workflow file at
`.github/workflows/staffml-cla.yml` (paths-filtered so it only runs on StaffML
PRs):

```yaml
name: StaffML CLA Assistant

on:
  issue_comment:
    types: [created]
  pull_request_target:
    types: [opened, closed, synchronize]
    paths:
      - 'interviews/staffml/**'

permissions:
  actions: write
  contents: write
  pull-requests: write
  statuses: write

jobs:
  cla-check:
    runs-on: ubuntu-latest
    steps:
      - name: CLA Assistant
        if: (github.event.comment.body == 'I have read the CLA Document and I hereby sign the CLA') || github.event_name == 'pull_request_target'
        uses: contributor-assistant/github-action@v2.6.1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          PERSONAL_ACCESS_TOKEN: ${{ secrets.CLA_SIGNATURE_PAT }}
        with:
          path-to-signatures: 'interviews/staffml/.cla-signatures.json'
          path-to-document: 'https://github.com/harvard-edge/cs249r_book/blob/dev/interviews/staffml/CLA.md'
          branch: 'main'
          allowlist: 'dependabot[bot],github-actions[bot]'
```

Setup steps:
1. Create a fine-grained Personal Access Token with `repo:contents:write` scope.
2. Add it as a repository secret named `CLA_SIGNATURE_PAT`.
3. Commit the workflow file above.
4. The bot will create `interviews/staffml/.cla-signatures.json` on first signature.

The workflow uses [`contributor-assistant/github-action`](https://github.com/contributor-assistant/github-action),
which is self-hosted (signatures stored in this repo, no third-party service).
