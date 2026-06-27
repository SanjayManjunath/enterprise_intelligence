import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // 1. Forces Next to generate a micro-bundle, slashing your Docker RAM/Disk footprint
  output: "standalone",

  async rewrites() {
    return [
      {
        source: '/api/:path*',
        // 2. Explicitly mapped to the Docker Compose service name
        destination: 'http://enterprise-backend:8000/:path*' 
      }
    ]
  }
};

export default nextConfig;
