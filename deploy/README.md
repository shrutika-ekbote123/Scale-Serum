# Deploying to a server with GitHub Actions

Push to `main` → GitHub Actions verifies the build, SSHes in as `root`, enters the
app directory, pulls that exact commit, installs any missing dependencies,
`pm2 restart`s the process and health-checks it. **If the health check fails it
automatically rolls back** to the previous commit and the workflow fails red.

| File | What it is |
|------|------------|
| [`../.github/workflows/deploy.yml`](../.github/workflows/deploy.yml) | The GitHub Actions workflow. |
| [`deploy.sh`](deploy.sh) | Runs on the server (piped in over SSH). Pull → install → pm2 restart → health-check → rollback. |
| [`pm2.config.json`](pm2.config.json) | Fallback pm2 process definition, used only if pm2 has no process registered and there is no `ecosystem.config.js`. |
| [`nginx.conf.example`](nginx.conf.example) | Reverse proxy + TLS front, with the 60s timeout Script Lab needs. |

`brand-brain.service` is the systemd alternative to pm2. Use **one or the other** —
if both are enabled they fight over the same port.

---

## 1. The server as it is today

The live box is already set up. Everything in the workflow defaults to this layout,
so no repository variables are needed:

| | |
|---|---|
| App directory | `/root/Marketing_tool` |
| Virtualenv | `venv/` (not `.venv/` — `deploy.sh` detects whichever exists) |
| pm2 process | `marketing-tool`, running under **root's** pm2 daemon |
| pm2 config | `ecosystem.config.js` in the app directory — untracked, so `git reset --hard` never touches it |

Verify any of it with:

```bash
pm2 list                        # marketing-tool must be here, status "online"
pm2 describe marketing-tool     # exec cwd, the uvicorn args, and the real port
```

If the port there is not 3001, set the `HEALTH_URL` repository variable to match,
or every deploy will "fail" the health check and roll back a perfectly good commit.

### pm2 is per-user

Each Linux user gets its own pm2 daemon with its own process list. The workflow
logs in as `root`, so `marketing-tool` must be in **root's** `pm2 list` — which it
is. If it were ever started by another user, `pm2 restart` would fail with *process
not found* and `deploy.sh` would start a second copy that dies on the busy port.

### If `pm2: command not found` during a deploy

A non-interactive `ssh root@IP "command"` does not read `~/.bashrc`, so an
nvm-installed pm2 is invisible even though it works when you log in by hand.
`deploy.sh` sources `~/.nvm/nvm.sh` itself to cover this. If pm2 lives somewhere
else, a system-wide `npm install -g pm2` lands in `/usr/bin` and is always on `PATH`.

### Rebuilding from scratch

```bash
apt update && apt install -y python3 python3-venv python3-pip git curl nginx
curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt install -y nodejs
npm install -g pm2

git clone https://github.com/Rafe-wdc/Marketing_tool.git /root/Marketing_tool
cd /root/Marketing_tool
cp .env.example .env
nano .env                 # GEMINI_API_KEY, API_KEY, ALLOWED_ORIGINS, MONGODB_URI
chmod 600 .env

python3 -m venv venv
venv/bin/pip install -r requirements.txt

pm2 start deploy/pm2.config.json
pm2 save                  # remember this process list
pm2 startup               # prints a command — run it, so pm2 comes back on reboot

curl http://127.0.0.1:3001/health          # -> {"ok":true,"model":"gemini-2.5-flash"}
```

> If the repo is private, clone over SSH with a deploy key instead of HTTPS, or the
> workflow's `git fetch` has no credentials and fails.

### Nginx + TLS

```bash
cp deploy/nginx.conf.example /etc/nginx/sites-available/marketing-tool
nano /etc/nginx/sites-available/marketing-tool      # set your server_name
ln -s /etc/nginx/sites-available/marketing-tool /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
apt install -y certbot python3-certbot-nginx
certbot --nginx -d api.yourdomain.com
```

---

## 2. SSH access for GitHub Actions

The workflow logs in as **root with the Vultr password**, using `sshpass`. Two
things to check on the server before wiring it up:

```bash
# Password logins must be enabled for root (the Vultr default, but confirm):
sudo grep -E '^(PermitRootLogin|PasswordAuthentication)' /etc/ssh/sshd_config
#   PermitRootLogin yes
#   PasswordAuthentication yes

# Grab the host key fingerprint for SSH_KNOWN_HOSTS:
ssh-keyscan -p 22 <server-ip>
```

Verify the login works before pushing:

```bash
# Note the quoted form — this is exactly how the workflow runs, so it also proves
# pm2 is on PATH non-interactively (see §1 if it is not).
ssh root@<server-ip> 'pm2 restart marketing-tool && echo ok'
```

> A root password in CI is weaker than a key: it is reusable, it grants full
> control of the box, and anything that can read the runner's environment gets it.
> If you want to tighten this later, switch back to a dedicated `deploy` user with
> an SSH key — `deploy.sh` still supports it (it uses `sudo` whenever it is not
> running as root).

---

## 3. GitHub secrets and variables

**Settings → Secrets and variables → Actions**

Secrets:

| Secret | Value |
|--------|-------|
| `SSH_HOST` | server IP or hostname |
| `SSH_PASSWORD` | the root password |
| `SSH_KNOWN_HOSTS` | output of `ssh-keyscan -p 22 <server-ip>` |

`SSH_KNOWN_HOSTS` is technically optional, but with password auth it matters more
than usual: without it the workflow sends the root password to whatever host
answers on that IP, with no way to detect a man-in-the-middle. Set it. Re-run
`ssh-keyscan` if you ever rebuild the server.

Variables (only if your setup differs from the defaults):

| Variable | Default |
|----------|---------|
| `SSH_USER` | `root` |
| `SSH_PORT` | `22` |
| `APP_DIR` | `/root/Marketing_tool` |
| `PM2_NAME` | `marketing-tool` — must match a name in `pm2 list` |
| `HEALTH_URL` | `http://127.0.0.1:3001/health` — must match the port pm2 actually runs uvicorn on |

**The defaults already match the live server**, so you should not need to set any
of these. Confirm with `pm2 describe marketing-tool` — the `script args` line shows
the real port, and `exec cwd` the real directory.

The deploy job uses a `production` environment. GitHub creates it on the first run;
add required reviewers under **Settings → Environments → production** if you want a
manual approval gate before each deploy.

---

## 4. Running a deploy

- **Automatic** — push to `main`. Doc-only, `frontend/` and `.gitignore` changes are
  skipped, since they do not affect the running service.
- **Manual** — **Actions → Deploy to server → Run workflow**. Deploys the HEAD of
  the branch you pick.

Deploys are serialised per branch (`concurrency`), so two pushes in a row queue
rather than fighting over the server.

---

## 5. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Missing repository secret(s)` | Secrets not set, or set on the wrong repo/environment. |
| `Permission denied, please try again` | Wrong `SSH_PASSWORD`, or `PasswordAuthentication`/`PermitRootLogin` is `no` in `/etc/ssh/sshd_config`. |
| `No ED25519 host key is known for ...` / `Host key verification failed` | `SSH_KNOWN_HOSTS` has no entry matching `SSH_HOST`. The **Pin the host key** step now catches this first and prints why. Usually only one `ssh-keyscan` line was pasted, or the scan used a hostname while `SSH_HOST` is an IP — the two strings are compared literally and must match exactly. Also re-run `ssh-keyscan` if the server was rebuilt; the keys change. |
| `detected dubious ownership` | `git config --global --add safe.directory /root/Marketing_tool` as the deploying user. `deploy.sh` does this itself, so it usually means it ran on a different path. |
| `pm2 is not on PATH` | pm2 installed under nvm, or not installed for `root`. See §1 above. |
| `pm2 process not found — starting it` on every deploy | pm2 was started as a different user, or `PM2_NAME` does not match a name in `pm2 list`. Check as root. |
| `APP_DIR ... does not exist` | The `APP_DIR` variable points somewhere the app is not. The live path is `/root/Marketing_tool`. |
| `could not read Username for 'https://github.com'` | The server's checkout is a private repo cloned over HTTPS, so `git fetch` has no credentials. Fix on the server: `git remote set-url origin git@github.com:Rafe-wdc/Marketing_tool.git` with a deploy key, or embed a PAT in the HTTPS URL. |
| `HEALTH CHECK FAILED` then rollback | The new commit does not boot. The workflow log prints the last 60 pm2 log lines; usually a missing env var — most often `GEMINI_API_KEY`, which makes `app.py` raise at startup. |
| pm2 shows `errored` / restart count climbing | `pm2 logs marketing-tool --lines 50`. A wrong `interpreter` path in `pm2.config.json` (pm2 defaulting to node) shows up as an instant syntax error. |
| `$APP_DIR/.env is missing` | Config was never created on the server, or `git reset --hard` ran in the wrong directory. `.env` is gitignored, so deploys never overwrite it. |
| Deploy succeeds but the API 502s | Nginx is pointing at the wrong port, or the app binds `127.0.0.1` while Nginx proxies a different address. |

Useful on the server:

```bash
pm2 list                                       # is it online, how many restarts
pm2 logs marketing-tool --lines 50                # recent output
pm2 logs marketing-tool                           # live logs
pm2 describe marketing-tool                    # the exact command, cwd and port
cd /root/Marketing_tool && git log -1 --oneline   # which commit is live
```

**Manual deploy / rollback** — same script the workflow pipes in:

```bash
cd /root/Marketing_tool
bash deploy/deploy.sh                          # deploy the tip of main
DEPLOY_SHA=<good-sha> bash deploy/deploy.sh    # roll back to a known-good commit
```

Running it by hand is also the fastest way to debug a red workflow — it is the
exact same script, and you see the error immediately without the SSH layer.

---

## Notes

- The server checkout is reset with `git reset --hard` on every deploy — never edit
  files directly on the server, they will be wiped. `.env` and `.venv/` are
  gitignored and survive.
- `.env` is never touched by the workflow. Changing config is a server-side edit
  plus `pm2 restart marketing-tool --update-env` — env is read once at startup.
- `pm2 save` runs after every deploy, so the saved process list stays current. Run
  `pm2 startup` once at setup or pm2 will not come back after a reboot.
- `deploy.sh` is piped in from the checkout, so the server always runs the deploy
  script from the commit being deployed. Changes to it take effect immediately.

## Sales Call Analyzer

The analyzer needs `DEEPGRAM_API_KEY` in `$APP_DIR/.env` on the server.
`ecosystem.config.js` only forwards `GEMINI_API_KEY` through pm2 - everything
else is read from `.env` by `load_dotenv()`, so add it there.

It is optional: with no key the app boots normally, supplied transcripts still
analyse, and audio requests report `transcription_not_configured`. Check it with
`curl -s localhost:3001/health | jq .sales_call_analyzer` - the response says
`configured` or `not_configured` and never contains the key itself.

The analyzer also needs `MONGODB_URI` (it stores analyses in
`sales_call_analyses`); without it the endpoints return 503 like the Brand Brain
store. See `SALES_CALL_ANALYZER.md` for the API contract.
