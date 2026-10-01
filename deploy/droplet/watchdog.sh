#!/usr/bin/env bash
# Restart wildecho-api after 3 consecutive failed local health checks. Docker's
# restart policy only covers crashes; this covers a hung process. Run from cron:
#   */2 * * * * root /opt/wildecho-api/watchdog.sh
set -u
STATE=/var/tmp/wildecho-watchdog.failures
if curl -fs --max-time 15 http://127.0.0.1:7860/v1/health | grep -q '"model_loaded":true'; then
  echo 0 > "$STATE"
  exit 0
fi
failures=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
echo "$failures" > "$STATE"
if [ "$failures" -ge 3 ]; then
  logger -t wildecho-watchdog "health failed $failures times; restarting container"
  docker unpause wildecho-api >/dev/null 2>&1
  docker restart wildecho-api >/dev/null
  echo 0 > "$STATE"
fi
