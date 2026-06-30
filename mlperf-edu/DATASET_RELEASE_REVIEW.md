# MLPerf EDU Dataset Release Review

Last checked: 2026-06-28.

This document tracks dataset decisions needed before MLPerf EDU claims a public
score-bearing release. The suite is runnable today, but public-result status
requires clean source, license, redistribution, and attribution decisions.

## Current Public Warnings

| Dataset | Workloads | Current status | Why it is not closed |
|---|---|---|---|
| MovieLens-100K | `micro-dlrm-train` | `restricted-needs-approval` | GroupLens documents research-use conditions, citation requirements, no redistribution without separate permission, and noncommercial restrictions. |

## Closed Dataset Decisions

| Dataset | Workloads | Status | Decision |
|---|---|---|---|
| TinyShakespeare | `nanogpt-train` | `public-ok-fetch-only` | Replaced the prior unlicensed hosted corpus with a deterministic MLPerf EDU tiny excerpt generated from Project Gutenberg eBook 100. |
| Fashion-MNIST | `resnet18-train`, `mobilenetv2-train` | `public-ok-with-attribution` | Replaces CIFAR-100 as the default score-bearing public vision-training dataset because the upstream project carries an MIT license. |
| CIFAR-100 | none score-bearing by default | moved out of public default | CIFAR-backed experiments can remain useful for systems-only or optional variants, but public score-bearing vision rows now use Fashion-MNIST. |

## Source Evidence

- TinyShakespeare source recipe: <https://www.gutenberg.org/files/100/100-0.txt>
- Project Gutenberg license/terms: <https://www.gutenberg.org/policy/license.html>
- MovieLens-100K page: <https://grouplens.org/datasets/movielens/100k/>
- MovieLens-100K README/license text: <https://files.grouplens.org/datasets/movielens/ml-100k-README.txt>
- CIFAR-10/100 official page: <https://www.cs.toronto.edu/~kriz/cifar.html>
- Fashion-MNIST project and MIT license: <https://github.com/zalandoresearch/fashion-mnist>
- MNIST Keras loader page: <https://keras.io/api/datasets/mnist/>

## Decision Paths

| Dataset | Preferred release path | Fallback |
|---|---|---|
| MovieLens-100K | Keep score-bearing only with explicit permission or an MLCommons-approved fetch-only policy that does not redistribute data. | Move `micro-dlrm-train` to systems-only and add a clearly open recommender dataset for public scoring. |
| CIFAR-100 | Keep out of default score-bearing public rows unless explicit terms are resolved. | Use Fashion-MNIST for public vision training. |

## Review Questions

1. Can public MLPerf EDU score-bearing workloads use fetch-only datasets whose
   upstream terms restrict redistribution?
2. Do MLCommons reviewers accept Fashion-MNIST as the public score-bearing
   vision-training dataset for the first MLPerf EDU release?
3. For teaching releases, should restricted datasets remain available only under
   `pro`/systems-only paths with clear warnings?

## Implementation Checklist

| Status | Action |
|---|---|
| Done | Structured asset dossiers expose `public_release_status`, public policy, and next steps. |
| Done | `mlperf audit --policy public` can treat unresolved dataset release warnings as blockers. |
| Done | Reports surface dataset release status in JSON, HTML, and CSV. |
| Done | TinyShakespeare replaced by a deterministic Project Gutenberg source recipe. |
| Done | Public score-bearing vision training moved from CIFAR-100 to MIT-licensed Fashion-MNIST. |
| Open | Get MLCommons reviewer decision on MovieLens-100K fetch-only score-bearing use. |
