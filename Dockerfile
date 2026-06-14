# syntax=docker/dockerfile:1
# Prima web app — Next.js (App Router) + LangGraph agent fleet.
# Multi-stage: install/compile deps → build standalone → minimal runtime.
# Node 22 has reliable better-sqlite3 prebuilt binaries (native module).

FROM node:22-slim AS base
ENV NEXT_TELEMETRY_DISABLED=1

# ---- deps: install node_modules (build tools present so better-sqlite3 can compile if no prebuild) ----
FROM base AS deps
WORKDIR /app
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 make g++ ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY package.json package-lock.json ./
RUN npm ci

# ---- build: compile the Next.js standalone server ----
FROM base AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# ---- runner: just the standalone output + native module + static assets ----
FROM base AS runner
WORKDIR /app
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0
# Standalone bundle already contains the traced node_modules (incl. better-sqlite3 .node)
COPY --from=build /app/public ./public
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
# Bundled AIOps KPI sample for the read-only dataset viewer (read at runtime by /api/kpi).
COPY --from=build /app/data/aiops-sample.json ./data/aiops-sample.json
# DB + CSV cache live on the mounted volume (PRIMA_DB_PATH=/data/prima.db); ensure it exists.
RUN mkdir -p /data
EXPOSE 3000
CMD ["node", "server.js"]
