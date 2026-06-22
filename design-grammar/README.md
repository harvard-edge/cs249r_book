# ML Systems Design Grammar

<!-- EARLY-RELEASE-CALLOUT:START -->
> [!NOTE]
> **📌 Early release (2026)**
>
> This artifact shipped with the **2026** MLSysBook refresh. The catalog data, build scripts, and HTML outputs are **actively iterated** as the curriculum and tooling evolve.
>
> **Feedback** — [GitHub issues](https://github.com/harvard-edge/cs249r_book/issues) or pull requests.
<!-- EARLY-RELEASE-CALLOUT:END -->

This directory holds the **ML Systems Design Grammar**. The primitive catalog is
the parts catalog: `grammar.yml` is the source of truth for primitives, resources,
controls, and metrics. The rewrite playbook lives in `rewrite-rules.yml`, where rules
such as tiling, fusion, sharding, batching, caching, quantization, scheduling,
and virtualization are mapped to the constraints they relieve.

The teaching loop is:

```text
naive system + binding constraint -> rewrite rule -> feasible system
```

See `DESIGN_GRAMMAR.md` for the conceptual frame. Node build scripts generate
HTML, and validation scripts consume the catalog data. See `package.json` for
`build` and `validate` scripts.

---

## Contributors

Thanks to everyone who helps improve this artifact!

**Legend:** 🪲 Bug Hunter · 🧑‍💻 Code Contributor · ✍️ Doc Wizard · 🎨 Design Artist · 🧠 Idea Spark · 🔎 Code Reviewer · 🧪 Test Tinkerer · 🛠️ Tool Builder

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<p><em>No contributors listed yet—be the first!</em></p>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

**Recognize a contributor:** Comment on an issue or PR in this area:

```text
@all-contributors please add @username for code, doc, ideas, or bug in design-grammar
```

You can say **design-grammar**, **ml-systems-design-grammar**, or **primitive catalog** in the comment so the bot picks this project.
