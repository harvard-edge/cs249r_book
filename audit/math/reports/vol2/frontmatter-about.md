# Math Audit Report: `book/quarto/contents/vol2/frontmatter/about.qmd`

## Checked scope

Audited `book/quarto/contents/vol2/frontmatter/about.qmd` for mathematical statements, equations, numeric calculations, unit conversions, complexity claims, and prose-equation consistency using direct reasoning only. Source `.qmd` files were not modified.

Quantitative and consistency-bearing content checked includes:

- Line 11: AlexNet training on two GPUs for five to six days, and GPT-4 training estimate of 25,000 GPUs for roughly three months.
- Line 13: 10,000-accelerator cluster, 2 percent annual failure rate, and a node failure roughly every two hours.
- Lines 17-21: five named fleet-scale "laws," including inference lifetime operational expenditure exceeding training capital expenditure by 100--1000$\times$.
- Line 29: illustrative scaling example with 512 added nodes, 4$\times$ aggregate compute, 1.8$\times$ AllReduce latency, and 0.87 net scaling efficiency.
- Lines 35-53 and 63: "four parts" / Part I--IV structure and the four Fleet Stack layers.
- Lines 57-60 and 67-89: LaTeX stack macros with numeric intensity parameters.
- Lines 113-115, 133-135, 167-174: resource counts, course/program references, 10,000-node scale prose, one-semester / half-semester module structure, and four Part introductions.

## Findings

### 1. Cluster failure interval does not match the stated annual failure rate

- **Line 13:** `in a 10,000-accelerator cluster with a 2 percent annual failure rate, a node fails roughly every two hours`

With a 2 percent annual failure rate per accelerator and 10,000 accelerators, the expected cluster-wide failure count is:

```text
10,000 accelerators * 0.02 failures/accelerator-year = 200 failures/year
```

There are 8,760 hours in a non-leap year, so the expected interval is:

```text
8,760 hours/year / 200 failures/year = 43.8 hours/failure
```

That is roughly one failure every 44 hours, not every two hours. To get a failure every two hours from 10,000 accelerators would require about 4,380 failures/year, or a per-accelerator annual failure rate of about 43.8 percent.

**Suggested correction:** Change the interval to `roughly every 44 hours` if the 2 percent annual failure rate and 10,000-accelerator count are intended, or revise the failure rate/device count if `roughly every two hours` is the intended scale.

## Notes

- Line 20's `100--1000$\times$` comparison is dimensionally acceptable because lifetime inference OPEX and training CAPEX are both cost quantities before forming the ratio.
- Line 29's scaling example is not independently derivable from the stated values alone: the 0.87 net scaling efficiency depends on unstated baseline step-time and communication-fraction assumptions. It is acceptable as an illustrative measured result, but not checkable as a standalone calculation.
- Lines 35-53 and 63 are internally consistent: the text introduces four parts and four Fleet Stack layers, and the subsequent prose names exactly four of each.
- Lines 57-60 and 67-89 use stack intensity parameters in the expected 0-100 range. The prose descriptions match the dominant macro values: Infrastructure is highest for Compute Infrastructure, Distribution is highest for Distributed Training, and Governance is highest for Responsible AI.
- No formal equations, algorithmic complexity claims, or unit conversions appear in this file.
