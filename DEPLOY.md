# Deploying Prima to Fly.io

Two Fly apps in the same org:

| App         | What it is                          | Source dir   | Reached at                     |
| ----------- | ----------------------------------- | ------------ | ------------------------------ |
| `prima-web` | Next.js + LangGraph agent fleet     | repo root    | public HTTPS URL               |
| `prima-ml`  | FastAPI + PyTorch (VAE + Chronos)   | `ml-service/`| `prima-ml.internal:8000` (private) |

The web app talks to the ML app over Fly's private 6PN network, so the ML service needs
**no public traffic**. If `prima-ml` is down, the web app automatically falls back to the
statistical anomaly detector — it never blocks.

## 0. One-time setup

```bash
# Install the CLI and log in
brew install flyctl          # or: curl -L https://fly.io/install.sh | sh
fly auth login
```

## 1. Deploy the ML ensemble (`prima-ml`)

```bash
cd ml-service
fly apps create prima-ml
fly deploy                   # builds the CPU-torch image (~a few minutes the first time)
```

> The first `/detect` and `/forecast` call downloads the Chronos-Bolt weights into the
> container (ephemeral). If cold starts feel slow, give it a HuggingFace cache volume:
> `fly volumes create hf_cache --size 2 --region yyz`, then add a `[[mounts]]` block to
> `ml-service/fly.toml` mapping it to `/app/.hfcache`. If you see OOM kills, bump
> `memory` to `4096mb` in `ml-service/fly.toml` and `fly deploy` again.

## 2. Deploy the web app (`prima-web`)

```bash
cd ..                                    # back to repo root
fly apps create prima-web

# Persistent disk for the SQLite warehouse (must exist before the first deploy)
fly volumes create prima_data --size 1 --region yyz

# Required secret — the agents call a live Claude model on every request
fly secrets set ANTHROPIC_API_KEY=sk-ant-...

fly deploy
fly open                                 # opens the dashboard
```

That's it. On the **first request ever**, the app downloads the UCI Online Retail dataset
and builds `prima.db` on the volume (tens of seconds). Every request after that — and every
restart — reuses the persisted DB, so it's fast.

## Notes & knobs

- **Region:** keep both apps in the same `primary_region` (default `yyz` (Toronto)) so the private
  hop is local. Change it in both `fly.toml`s and the `fly volumes create --region` flags.
- **Scale-to-zero:** both apps use `min_machines_running = 0` (cheapest). The web volume
  keeps the DB warm, so only the very first request is slow. Set the web app's value to `1`
  if you want zero cold starts for a live demo.
- **Internal address:** the web app reads `ML_SERVICE_URL=http://prima-ml.internal:8000`
  (set in `fly.toml`). If you rename the ML app, update that value.
- **Changing the model:** edit `PRIMA_MODEL` in `fly.toml` (e.g. `claude-opus-4-8`) and redeploy.
- **Logs / status:** `fly logs -a prima-web`, `fly status -a prima-web` (same for `prima-ml`).
- **Web only (skip ML):** you can deploy just step 2 — leave `prima-ml` undeployed and the
  graph runs with the statistical detector. The VAE/Chronos columns simply stay dark.

## Local Docker smoke test (optional)

```bash
# Web app
docker build -t prima-web .
docker run --rm -p 3000:3000 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e PRIMA_DB_PATH=/data/prima.db \
  -v prima_data:/data \
  prima-web                              # http://localhost:3000

# ML service
docker build -t prima-ml ./ml-service
docker run --rm -p 8000:8000 prima-ml    # http://localhost:8000/health
```

To wire them together locally, run the ML container and pass
`-e ML_SERVICE_URL=http://host.docker.internal:8000` to the web container.

## Makefile shortcuts

`make help` lists everything. Common ones:

```bash
make check          # full CI gate locally: lint + typecheck + build
make fly-setup      # one-time: create both apps + the web volume
make secrets KEY=sk-ant-...   # set ANTHROPIC_API_KEY on the web app
make deploy         # deploy ML then web
make deploy-web     # web only
make logs           # tail web logs
```

## CI auto-deploy (GitHub Actions)

`.github/workflows/deploy.yml` runs on every push/PR to `main`:

1. **verify** — `npm ci` → lint → `tsc --noEmit` → production build (the gate; runs on PRs too).
2. On a push to `main`: **deploy-web** always; **deploy-ml** only when `ml-service/**` changed
   (the PyTorch image is slow to build, so we skip it otherwise).

One-time setup to enable it:

```bash
# Create a Fly deploy token and add it to the repo as a secret named FLY_API_TOKEN
fly tokens create deploy -x 999999h
gh secret set FLY_API_TOKEN          # paste the token when prompted
```

> Builds run on Node 22 with `--remote-only`, so the GitHub runner needs no Docker —
> Fly's remote builder builds the images. The apps and the web volume must already
> exist (`make fly-setup`) before the first automated deploy.

## Next.js 16 note

Production builds use the **webpack builder** (`next build --webpack`). Next 16's default
Turbopack builder can't yet collect page data through the native `better-sqlite3` external
package; webpack handles it cleanly and still emits the `standalone` output the Docker image
needs. Dev (`npm run dev`) uses Turbopack as normal.
