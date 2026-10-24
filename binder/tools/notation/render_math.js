// Render book math to MathML with KaTeX, for binder's notation checks.
//
// Input (stdin, JSON):  {"macros": {"\\Name": "latex", ...},
//                        "items": [{"id": "...", "math": "...", "display": false}, ...]}
// Output (stdout, JSON): {"id": "<mathml>" or "ERROR: <message>", ...}
//
// Two math strings render the same for a reader exactly when their MathML is
// identical, so binder compares these strings instead of modelling TeX itself.
// Macros are passed per call, so the "before" side can use the registry as it
// was at an earlier commit and the "after" side the registry as it is now.

"use strict";
const katex = require("katex");

let raw = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { raw += chunk; });
process.stdin.on("end", () => {
  const req = JSON.parse(raw);
  const out = {};
  for (const item of req.items) {
    // KaTeX mutates the macros object when a document defines macros, so each
    // item gets a fresh copy.
    const macros = Object.assign({}, req.macros);
    try {
      const html = katex.renderToString(item.math, {
        output: "mathml",
        displayMode: Boolean(item.display),
        macros,
        throwOnError: true,
        strict: "ignore",
        trust: false,
      });
      // The <annotation> element carries the TeX source, which differs whenever
      // a macro replaces raw LaTeX. Only the rendered structure matters.
      out[item.id] = html.replace(/<annotation[^>]*>[\s\S]*?<\/annotation>/g, "");
    } catch (err) {
      out[item.id] = "ERROR: " + String(err.message || err);
    }
  }
  process.stdout.write(JSON.stringify(out));
});
