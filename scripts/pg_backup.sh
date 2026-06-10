#!/usr/bin/env bash
#
# Nightly PostgreSQL backup for the Am I A Robot production database.
#
# Dumps the production database out of the postgres_production container, gzips
# it into a timestamped file, and prunes old backups. Connections inside the
# container are trusted (local socket), so no password is needed here.
#
# Install on the droplet via cron (see DEPLOYMENT.md), e.g.:
#   30 3 * * * /home/deploy/Projects/Am_I_A_Robot/scripts/pg_backup.sh >> /var/log/amiarobot-backup.log 2>&1
set -euo pipefail

CONTAINER="postgres_production"
DB_USER="robot_user"
DB_NAME="am_i_a_robot_prod"
BACKUP_DIR="${BACKUP_DIR:-/home/deploy/backups}"
RETAIN=14  # number of daily backups to keep

mkdir -p "$BACKUP_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
outfile="$BACKUP_DIR/amiarobot_${timestamp}.sql.gz"

# Dump and compress. If the dump fails, the partial file is removed and we exit
# non-zero so cron mail / the log shows the failure.
if ! docker exec "$CONTAINER" pg_dump -U "$DB_USER" "$DB_NAME" | gzip > "$outfile"; then
    echo "$(date -Is) ERROR: pg_dump failed; removing partial $outfile" >&2
    rm -f "$outfile"
    exit 1
fi

size="$(stat -c%s "$outfile")"
if [ "$size" -lt 100 ]; then
    echo "$(date -Is) ERROR: backup suspiciously small (${size} bytes): $outfile" >&2
    exit 1
fi

# Prune all but the most recent $RETAIN backups.
ls -1t "$BACKUP_DIR"/amiarobot_*.sql.gz 2>/dev/null | tail -n +$((RETAIN + 1)) | xargs -r rm -f

echo "$(date -Is) OK: wrote $outfile (${size} bytes); kept latest $RETAIN"
