import path from 'path';

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  // Emit `<page>/index.html` instead of `<page>.html` so GitHub Pages serves
  // trailing-slash URLs (e.g. /staffml/practice/) correctly. Without this the
  // dev preview returns 404 for any deep link with a trailing slash, which
  // falls through to the surrounding Quarto book's 404 template.
  trailingSlash: true,
  images: { unoptimized: true },
  poweredByHeader: false,
  // When deployed to a subdirectory (e.g. /interviews/), set NEXT_PUBLIC_BASE_PATH=/interviews
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || '',
  // GitHub Actions sets GITHUB_REPOSITORY on every build; forwarding it keeps
  // the app's GitHub links correct through a repository rename (lib/env.ts).
  env: {
    NEXT_PUBLIC_GITHUB_REPOSITORY:
      process.env.NEXT_PUBLIC_GITHUB_REPOSITORY || process.env.GITHUB_REPOSITORY || '',
  },
  outputFileTracingRoot: path.join(process.cwd(), '../../'),
};

export default nextConfig;
