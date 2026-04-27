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
  outputFileTracingRoot: path.join(process.cwd(), '../../'),
  eslint: {
    // Warning: This allows production builds to successfully complete even if
    // your project has ESLint errors.
    ignoreDuringBuilds: true,
  },
};

export default nextConfig;
