/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  images: { unoptimized: true },
  poweredByHeader: false,
  // When deployed to a subdirectory (e.g. /staffml/), set NEXT_PUBLIC_BASE_PATH=/staffml
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || '',
};

export default nextConfig;
