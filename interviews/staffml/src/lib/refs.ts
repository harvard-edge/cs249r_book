/**
 * URL classifier for "deep dive" / chapter / paper references.
 *
 * The corpus has 4,000+ deep_dive_url values pointing at heterogeneous
 * destinations: ~40% mlsysbook.ai, ~24% the harvard-edge GitHub mirror,
 * ~10% arxiv, and the rest distributed across PyTorch docs, NVIDIA blogs,
 * vendor pages, papers, and personal blogs. Until this code existed every
 * link was hard-labeled "Read on MLSysBook.ai" — wrong on ~60% of links.
 *
 * This module returns a structured `RefInfo` so the UI can label each
 * reference honestly with the right source name and a sensible verb.
 */

export type RefSource =
  | "book"          // mlsysbook.ai (the textbook)
  | "arxiv"         // arxiv.org papers
  | "github"        // GitHub repos / blobs / issues
  | "pytorch"       // pytorch.org docs/blog
  | "nvidia"        // developer.nvidia.com / docs.nvidia.com
  | "google"        // research.google / cloud.google / developers.google
  | "huggingface"   // huggingface.co
  | "vendor-docs"   // aws, microsoft, azure, intel, amd, etc.
  | "blog"          // personal/company engineering blogs
  | "paper"         // ACM, IEEE, USENIX, NeurIPS, etc.
  | "wiki"          // wikipedia, wikichip
  | "external";     // catch-all

export interface RefInfo {
  /** Short, user-facing label, e.g. "Read on MLSysBook.ai" or "arXiv paper". */
  label: string;
  /** Stable identifier for routing, icons, and analytics. */
  source: RefSource;
  /** Whether this points at the textbook itself (vs. an external reference). */
  isBook: boolean;
  /** Hostname extracted from the URL, useful for fallback display. */
  host: string;
  /** True when the destination is currently known-broken or known-flaky.
   *  Used to show a "may be unavailable" indicator next to the link.
   *  Remove the flag once mlsysbook.ai chapter routes are back online. */
  mayBeUnavailable: boolean;
}

/** Internal: pull a hostname out of a URL without throwing on garbage. */
function safeHost(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

/**
 * Classify a deep-dive URL into a `RefInfo`. Pure function, no I/O.
 *
 * The dispatch goes from most specific to least specific. New hostnames
 * just need a new branch — there is no remote lookup, no cache, nothing
 * to invalidate.
 */
export function classifyRef(url: string | undefined | null, titleHint?: string): RefInfo {
  if (!url) {
    return { label: "Reference", source: "external", isBook: false, host: "", mayBeUnavailable: false };
  }
  const host = safeHost(url);

  // The textbook itself.
  // NOTE: as of 2026-04, mlsysbook.ai chapter routes return 404 even though
  // the homepage links to them. Mark as "may be unavailable" until the
  // deployment is fixed. Remove the flag once verified working.
  if (host === "mlsysbook.ai" || host.endsWith(".mlsysbook.ai")) {
    return { label: "Read on MLSysBook.ai", source: "book", isBook: true, host, mayBeUnavailable: true };
  }
  // The GitHub-hosted dev mirror of the same book — fully retired, all 404.
  if (host === "harvard-edge.github.io") {
    return { label: "Read on MLSysBook.ai", source: "book", isBook: true, host, mayBeUnavailable: true };
  }

  // Papers
  if (host === "arxiv.org") {
    return { label: "Read paper on arXiv", source: "arxiv", isBook: false, host, mayBeUnavailable: false };
  }
  if (
    host === "dl.acm.org" ||
    host === "ieeexplore.ieee.org" ||
    host === "www.usenix.org" ||
    host === "papers.nips.cc" ||
    host === "proceedings.neurips.cc" ||
    host === "proceedings.mlr.press" ||
    host === "aclanthology.org" ||
    host === "jmlr.csail.mit.edu" ||
    host === "www.jmlr.org" ||
    host === "eprint.iacr.org"
  ) {
    return { label: "Research paper", source: "paper", isBook: false, host, mayBeUnavailable: false };
  }

  // Frameworks and accelerator vendors
  if (host === "pytorch.org" || host.endsWith(".pytorch.org")) {
    return { label: "PyTorch docs", source: "pytorch", isBook: false, host, mayBeUnavailable: false };
  }
  if (host === "developer.nvidia.com" || host === "docs.nvidia.com" ||
      host === "blogs.nvidia.com" || host === "www.nvidia.com" ||
      host === "images.nvidia.com" || host === "enterprise-support.nvidia.com") {
    return { label: "NVIDIA documentation", source: "nvidia", isBook: false, host, mayBeUnavailable: false };
  }
  if (host === "research.google" || host === "research.google.com" ||
      host === "cloud.google.com" || host === "developers.google.com" ||
      host === "sre.google" || host === "openxla.org") {
    return { label: "Google reference", source: "google", isBook: false, host, mayBeUnavailable: false };
  }
  if (host === "huggingface.co") {
    return { label: "Hugging Face", source: "huggingface", isBook: false, host, mayBeUnavailable: false };
  }

  // GitHub (repos, blobs, raw, gists, pages)
  if (host === "github.com" || host === "raw.githubusercontent.com" ||
      host === "gist.github.com") {
    return { label: "GitHub", source: "github", isBook: false, host, mayBeUnavailable: false };
  }

  // Vendor docs
  if (host === "aws.amazon.com" || host === "docs.aws.amazon.com" ||
      host.endsWith(".microsoft.com") || host === "learn.microsoft.com" ||
      host.endsWith(".intel.com") || host === "intel.github.io" ||
      host === "www.tsmc.com" || host === "www.broadcom.com" ||
      host === "www.cisco.com" || host === "www.arista.com" ||
      host === "www.databricks.com") {
    return { label: "Vendor documentation", source: "vendor-docs", isBook: false, host, mayBeUnavailable: false };
  }

  // Wikis
  if (host === "en.wikipedia.org" || host === "en.wikichip.org" ||
      host === "fuse.wikichip.org") {
    return { label: "Wikipedia", source: "wiki", isBook: false, host, mayBeUnavailable: false };
  }

  // Engineering blogs (a partial list of the ones that show up in the corpus)
  const blogHosts = new Set([
    "engineering.fb.com", "ai.meta.com", "research.facebook.com",
    "netflixtechblog.com", "eng.lyft.com", "engineering.linkedin.com",
    "blog.vllm.ai", "vllm.ai", "docs.vllm.ai", "vllm.readthedocs.io",
    "kipp.ly", "horace.io", "huyenchip.com", "lilianweng.github.io",
    "colah.github.io", "finbarr.ca", "eugeneyan.com", "blog.eleuther.ai",
    "bair.berkeley.edu", "blog.apnic.net", "queue.acm.org", "cacm.acm.org",
    "spectrum.ieee.org", "a16z.com", "openai.com", "machinelearning.apple.com",
    "petewarden.com", "martinfowler.com", "martin.kleppmann.com",
    "horace.io", "andrew.gibiansky.com", "mwburke.github.io",
    "colin-scott.github.io", "ethernetalliance.org",
  ]);
  if (blogHosts.has(host)) {
    return { label: "Engineering blog", source: "blog", isBook: false, host, mayBeUnavailable: false };
  }

  // Catch-all: name it after the host so the user at least knows where it goes.
  return {
    label: `Read on ${host || "external site"}`,
    source: "external",
    isBook: false,
    host,
    mayBeUnavailable: false,
  };
}

/**
 * Compose a "Learn more" line that tells the user honestly where the link
 * goes. Used by both Practice and the Vault topic detail.
 *
 * Returns a tuple of (prefix, host-or-source-name, suffix) so the caller
 * can style the host span differently.
 */
export function refLearnMore(url: string | undefined | null, title?: string): {
  info: RefInfo;
  text: string;
} {
  const info = classifyRef(url);
  const text = title ? `${info.label} — ${title}` : info.label;
  return { info, text };
}
