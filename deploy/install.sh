#!/usr/bin/env bash
# FitScan Server Install Script
# Ubuntu 22.04 LTS — installs FitScan as a standalone service on port 8000
# Usage: sudo bash deploy/install.sh   (run from fitscan/ or scp to server)
# Run once on a fresh server alongside your StitchHub install.

set -euo pipefail

REPO_URL="https://github.com/yourorg/fitscan.git"  # FitScan repo (its own git repo)
APP_USER="fitscan"
APP_DIR="/home/${APP_USER}/app"         # clones fitscan repo here
FITSCAN_DIR="${APP_DIR}"                # fitscan IS the repo root
PYTHON_VERSION="3.11"
PORT="8000"

GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[FITSCAN]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
die()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

[ "$(id -u)" -eq 0 ] || die "Run as root: sudo bash fitscan/deploy/install.sh"

# ─── 1. System packages ───────────────────────────────────────────
log "Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
  python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
  postgresql-15 postgresql-server-dev-15 \
  redis-server git build-essential curl \
  libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6

# ─── 2. pgvector ──────────────────────────────────────────────────
log "Installing pgvector..."
cd /tmp
git clone --depth 1 https://github.com/pgvector/pgvector.git pgvector_build 2>/dev/null || true
cd pgvector_build 2>/dev/null && make PG_CONFIG=/usr/bin/pg_config && \
  make install PG_CONFIG=/usr/bin/pg_config || true
cd / && rm -rf /tmp/pgvector_build

# ─── 3. App user ──────────────────────────────────────────────────
log "Creating fitscan system user..."
id "${APP_USER}" &>/dev/null || useradd -m -s /bin/bash "${APP_USER}"
echo "${APP_USER} ALL=(ALL) NOPASSWD: /bin/systemctl restart fitscan, /bin/systemctl status fitscan" \
  > /etc/sudoers.d/fitscan
chmod 0440 /etc/sudoers.d/fitscan

# ─── 4. Clone FitScan repo ───────────────────────────────────────
log "Cloning FitScan repo..."
if [ -d "${APP_DIR}/.git" ]; then
  sudo -u "${APP_USER}" git -C "${APP_DIR}" pull origin main
else
  sudo -u "${APP_USER}" git clone "${REPO_URL}" "${APP_DIR}"
fi

# ─── 5. Python venv in fitscan/ ───────────────────────────────────
log "Setting up Python venv at ${FITSCAN_DIR}/venv..."
# chumpy's setup.py does `import pip`, which fails under PEP 517 build
# isolation. The same fix lives in .github/workflows/fitscan-ci.yml —
# install numpy + chumpy outside isolation BEFORE the editable install,
# so the latter sees chumpy as already-satisfied and doesn't rebuild it.
sudo -u "${APP_USER}" bash -c "
  cd ${FITSCAN_DIR}
  python${PYTHON_VERSION} -m venv venv
  venv/bin/pip install --upgrade pip setuptools wheel --quiet
  venv/bin/pip install 'numpy>=1.26,<2' --quiet
  venv/bin/pip install --no-build-isolation 'chumpy>=0.70' --quiet
  venv/bin/pip install -e . psycopg2-binary pgvector --quiet
"

# ─── 6. Environment file ──────────────────────────────────────────
if [ ! -f "${FITSCAN_DIR}/.env" ]; then
  sudo -u "${APP_USER}" cp "${FITSCAN_DIR}/.env.example" "${FITSCAN_DIR}/.env"
  warn "Created ${FITSCAN_DIR}/.env — you MUST set ANTHROPIC_API_KEY"
fi
sudo -u "${APP_USER}" sed -i "s|APP_ENV=.*|APP_ENV=production|" "${FITSCAN_DIR}/.env"

# ─── 7. PostgreSQL ────────────────────────────────────────────────
log "Setting up PostgreSQL for FitScan..."
systemctl enable postgresql && systemctl start postgresql
DB_PASSWORD=$(openssl rand -hex 16)
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='fitscan'" | grep -q 1 || \
  sudo -u postgres psql -c "CREATE USER fitscan WITH PASSWORD '${DB_PASSWORD}';"
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='fitscan_prod'" | grep -q 1 || \
  sudo -u postgres createdb fitscan_prod -O fitscan
sudo -u postgres psql -d fitscan_prod -c "CREATE EXTENSION IF NOT EXISTS vector;" || true
DB_URL="postgresql://fitscan:${DB_PASSWORD}@localhost/fitscan_prod"
sudo -u "${APP_USER}" sed -i "s|DATABASE_URL=.*|DATABASE_URL=${DB_URL}|" "${FITSCAN_DIR}/.env"
log "PostgreSQL configured. DATABASE_URL updated."

# ─── 8. Alembic migrations ────────────────────────────────────────
log "Running database migrations..."
sudo -u "${APP_USER}" bash -c "
  cd ${FITSCAN_DIR}
  set -a && source .env && set +a
  venv/bin/alembic upgrade head
"

# ─── 9. systemd service ───────────────────────────────────────────
log "Creating systemd service: fitscan.service..."
cat > /etc/systemd/system/fitscan.service << EOF
[Unit]
Description=FitScan Body Measurement API
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${FITSCAN_DIR}
EnvironmentFile=${FITSCAN_DIR}/.env
ExecStart=${FITSCAN_DIR}/venv/bin/uvicorn fitscan.api.main:app \\
    --host 127.0.0.1 \\
    --port ${PORT} \\
    --workers 4 \\
    --log-level info
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable fitscan
systemctl start fitscan
sleep 3
systemctl is-active --quiet fitscan && log "FitScan service running on :${PORT}" || \
  die "Service failed. Check: sudo journalctl -u fitscan -n 50"

# ─── 10. Summary ──────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   FitScan installed successfully!        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo "  Running at:   http://127.0.0.1:${PORT}  (internal only)"
echo "  API docs:     http://127.0.0.1:${PORT}/docs"
echo "  Service:      sudo systemctl status fitscan"
echo "  Logs:         sudo journalctl -u fitscan -f"
echo "  App path:     ${FITSCAN_DIR}"
echo ""
echo -e "${YELLOW}Required next steps:${NC}"
echo "  1. Edit ${FITSCAN_DIR}/.env — set ANTHROPIC_API_KEY=sk-ant-..."
echo "  2. Download SMPL model from https://smpl.is.tue.mpg.de/"
echo "     Place SMPL_NEUTRAL.pkl at ${FITSCAN_DIR}/data/models/smpl/"
echo "  3. sudo -u fitscan bash -c 'cd ${FITSCAN_DIR} && venv/bin/python scripts/download_models.py'"
echo "  4. sudo systemctl restart fitscan"
echo "  5. curl http://127.0.0.1:${PORT}/api/v1/health"
echo ""
echo -e "${YELLOW}Add to your StitchHub Nginx config:${NC}"
echo "  # FitScan is internal — StitchHub calls it at 127.0.0.1:${PORT}"
echo "  # No public exposure needed unless you want /docs accessible"
echo ""
