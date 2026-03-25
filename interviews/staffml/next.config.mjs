/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  images: { unoptimized: true },
  // Disable x-powered-by header
  poweredByHeader: false,
};

export default nextConfig;
