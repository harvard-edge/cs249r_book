import path from 'path';

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
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
