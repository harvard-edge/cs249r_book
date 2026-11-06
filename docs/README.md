# Repository Docs

Documentation that applies to the repository as a whole. Each project keeps its own docs next to its code (for example `binder/docs/` for the book toolchain).

| File | Covers |
|---|---|
| [`REPO_LAYOUT.md`](REPO_LAYOUT.md) | Where every area of the repository lives and why |
| [`CI-VARIABLES.md`](CI-VARIABLES.md) | Repository variables the GitHub workflows read |
| [`VERSIONING.md`](VERSIONING.md) | How book and project releases are versioned |

## Cloudflare service naming

Use lowercase kebab-case for deployed resource names. Start with the owning
product and name the service's responsibility: `staffml-interviewer`,
`staffml-vault`, `staffml-analytics`, `mlsysbook-stats`, and `socratiq-api`.
Display names may retain product capitalization, such as SocratiQ.

| Resource | Convention | SocratiQ example |
| --- | --- | --- |
| Production Worker | `<product>-<role>` | `socratiq-api` |
| Nonproduction Worker | `<product>-<role>-<environment>` | `socratiq-api-staging`, `socratiq-api-local` |
| Storage resource, when needed | `<product>-<role>-<purpose>[-<environment>]` | `socratiq-api-quota-staging` |
| Public API base | `/api/<service>` on the product's published origin | `https://mlsysbook.ai/api/socratiq` |
| Runtime bindings and secrets | Uppercase snake case describing their purpose | `AI_RATE_LIMITER`, `GROQ_API_KEY` |
| New validation workflow | `<product>-validate[-<target>].yml` | `socratiq-validate.yml` |
| New release workflow | `<product>-publish-<target>.yml` | `socratiq-publish-live.yml` |

Production names have no environment suffix; use `staging` and `local` for
nonproduction resources. Binding names are local to a Worker and may repeat,
but underlying resources and credentials should be separately provisioned for
each service and environment. Numeric rate-limit namespace identifiers must be
allocated uniquely within the account; a descriptive binding name does not
isolate counters by itself.

The API base is a stable public contract, independent of the model or hosting
provider. Do not put provider names, model names, release numbers, or personal
names in service identifiers. Keep service source and Wrangler configuration
inside the owning project's directory. Keep deployment credentials in secret
stores, never in source or resource names.

These conventions apply to new resources. Existing deployed names and URLs
remain supported until an explicit migration; do not rename them solely for
cosmetic consistency. Examples above do not imply resources are provisioned.
