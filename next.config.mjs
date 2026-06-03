/** @type {import('next').NextConfig} */
const nextConfig = {
  // Emit a self-contained server bundle (.next/standalone) for a lean Docker image.
  output: "standalone",
  // better-sqlite3 is a native module — keep it out of the bundler and load it at runtime.
  // NOTE: production builds use the webpack builder (`next build --webpack`) — Next 16's
  // default Turbopack builder can't collect page data through this native external package yet.
  serverExternalPackages: ["better-sqlite3"],
};

export default nextConfig;
