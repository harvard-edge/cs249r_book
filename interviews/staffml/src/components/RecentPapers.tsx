"use client";

interface Paper {
  title: string;
  authors: string;
  url: string;
}

const papers: Paper[] = [
  {
    title: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate",
    authors: "Zandieh et al., 2025",
    url: "https://arxiv.org/abs/2504.19874",
  },
  // add 3-5 more here
];

export default function RecentPapers({ className }: { className?: string }) {
  return (
    <section
      aria-labelledby="recent-papers-heading"
      className={`p-5 rounded-xl border border-accentBlue/30 bg-accentBlue/5 ${className ?? ""}`}
    >
      <h2 id="recent-papers-heading" className="text-sm font-semibold text-textPrimary mb-3">
        Recent papers
      </h2>
      <ul className="space-y-3">
        {papers.map((p) => (
          <li key={p.url} className="border-b border-borderSubtle last:border-0 pb-3 last:pb-0">
            <a href={p.url} target="_blank" rel="noopener noreferrer" className="text-[13px] font-medium text-textPrimary hover:text-accentBlue transition-colors">
              {p.title}
            </a>
            <p className="text-[12px] text-textSecondary mt-0.5">{p.authors}</p>
          </li>
        ))}
      </ul>
    </section>
  );
}