#!/usr/bin/env bash

set -Eeuo pipefail

AUTODL_HOST="${AUTODL_HOST:-}"
AUTODL_PORT="${AUTODL_PORT:-22}"
AUTODL_USER="${AUTODL_USER:-root}"
AUTODL_KEY="${AUTODL_KEY:-${HOME}/.ssh/id_ed25519}"
AUTODL_REMOTE_DIR="${AUTODL_REMOTE_DIR:-/root/autodl-tmp/vlmintune}"
AUTODL_PYTHON="${AUTODL_PYTHON:-/root/autodl-tmp/venvs/vlmintune/bin/python}"

usage() {
  cat <<'EOF'
Usage: scripts/autodl_sync_test.sh [--dry-run | --sync-only]

Synchronize the current vlmintune working tree to AutoDL over SSH and run the
remote unit-test suite. Connection settings can be overridden with:

  AUTODL_HOST        required
  AUTODL_PORT        default: 22
  AUTODL_USER
  AUTODL_KEY
  AUTODL_REMOTE_DIR
  AUTODL_PYTHON

Options:
  --dry-run    Show the rsync changes without transferring files or testing.
  --sync-only  Transfer files without running the remote test suite.
EOF
}

mode="sync-test"
case "${1:-}" in
  "") ;;
  --dry-run) mode="dry-run" ;;
  --sync-only) mode="sync-only" ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ -z "$AUTODL_HOST" ]]; then
  echo "AUTODL_HOST is required." >&2
  exit 2
fi
if [[ ! "$AUTODL_HOST" =~ ^[A-Za-z0-9.-]+$ ]]; then
  echo "Unsafe AUTODL_HOST: $AUTODL_HOST" >&2
  exit 2
fi
if [[ ! "$AUTODL_PORT" =~ ^[0-9]+$ ]]; then
  echo "AUTODL_PORT must be numeric: $AUTODL_PORT" >&2
  exit 2
fi
if [[ ! "$AUTODL_USER" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Unsafe AUTODL_USER: $AUTODL_USER" >&2
  exit 2
fi
if [[ "$AUTODL_REMOTE_DIR" != /root/autodl-tmp/* ]] \
  || [[ "$AUTODL_REMOTE_DIR" == "/root/autodl-tmp/" ]] \
  || [[ "$AUTODL_REMOTE_DIR" == *".."* ]] \
  || [[ "$AUTODL_REMOTE_DIR" =~ [^A-Za-z0-9_./-] ]]; then
  echo "AUTODL_REMOTE_DIR must be a safe child of /root/autodl-tmp" >&2
  exit 2
fi
if [[ ! "$AUTODL_PYTHON" =~ ^/[A-Za-z0-9_./-]+$ ]] \
  || [[ "$AUTODL_PYTHON" == *".."* ]]; then
  echo "AUTODL_PYTHON must be a safe absolute path" >&2
  exit 2
fi
if [[ ! -f "$AUTODL_KEY" ]]; then
  echo "SSH private key not found: $AUTODL_KEY" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_target="${AUTODL_USER}@${AUTODL_HOST}"
ssh_args=(
  -i "$AUTODL_KEY"
  -p "$AUTODL_PORT"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o ConnectTimeout=15
)

printf -v rsync_rsh 'ssh -i %q -p %q -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15' \
  "$AUTODL_KEY" "$AUTODL_PORT"

rsync_args=(
  -acz
  --delete-delay
  --human-readable
  --stats
  --exclude=/.git/
  --exclude=/.venv/
  --exclude=/.pytest_cache/
  --exclude=/.pycache_validation/
  --exclude=/.mypy_cache/
  --exclude=/.ruff_cache/
  --exclude=/.agents/
  --exclude=/.codex/
  --exclude=__pycache__/
  --exclude=.DS_Store
  --exclude='*.pyc'
  --exclude=/.hf_token
  --exclude=/.env
  --exclude=/.env.local
  --exclude=/.coverage
  --exclude=/htmlcov/
  --exclude=/output/
  --exclude=/outputs/
  --exclude=/experiments/
  --exclude=/experiment_results/
  --exclude=/eval_outputs/
  --exclude=/run_logs/
  --exclude='*.safetensors'
  --exclude='*.pt'
  --exclude='*.pth'
  --exclude='*.tgz'
  --exclude='*.tar'
  --exclude='*.zip'
  --exclude='*.pem'
  --exclude='*.key'
  -e "$rsync_rsh"
)

echo "Checking AutoDL connection: $remote_target"
ssh "${ssh_args[@]}" "$remote_target" \
  "test -d /root/autodl-tmp && mkdir -p '$AUTODL_REMOTE_DIR' && command -v rsync >/dev/null"

if [[ "$mode" == "dry-run" ]]; then
  echo "Dry-run sync: $repo_root -> $remote_target:$AUTODL_REMOTE_DIR"
  rsync "${rsync_args[@]}" --dry-run --itemize-changes \
    "$repo_root/" "$remote_target:$AUTODL_REMOTE_DIR/"
  echo "Dry run complete; no files were transferred."
  exit 0
fi

sync_stable=0
for sync_attempt in 1 2 3; do
  echo "Synchronizing attempt $sync_attempt/3: $repo_root -> $remote_target:$AUTODL_REMOTE_DIR"
  rsync "${rsync_args[@]}" "$repo_root/" "$remote_target:$AUTODL_REMOTE_DIR/"

  verification_output="$(
    rsync "${rsync_args[@]}" --dry-run --itemize-changes \
      "$repo_root/" "$remote_target:$AUTODL_REMOTE_DIR/"
  )"
  if grep -Eq '^(<|>|c|h|\*)' <<<"$verification_output"; then
    echo "The local worktree changed during sync; retrying with the latest files." >&2
    printf '%s\n' "$verification_output" >&2
    continue
  fi

  sync_stable=1
  break
done

if [[ "$sync_stable" -ne 1 ]]; then
  echo "Could not obtain a stable worktree snapshot after 3 attempts." >&2
  exit 3
fi

if [[ "$mode" == "sync-only" ]]; then
  echo "Sync complete; tests were skipped."
  exit 0
fi

echo "Running remote tests with $AUTODL_PYTHON"
ssh "${ssh_args[@]}" "$remote_target" \
  "set -eu -o pipefail; cd '$AUTODL_REMOTE_DIR'; PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='$AUTODL_REMOTE_DIR/src' '$AUTODL_PYTHON' -m pytest -q -p no:cacheprovider tests"

echo "AutoDL sync and test completed successfully."
