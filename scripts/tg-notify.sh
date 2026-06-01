#!/usr/bin/env bash
# Minimal Telegram notifier for autonomous-loop phase reports.
# Reads TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID from .env at repo root.
# Usage: tg-notify.sh "<message>"  OR  echo "msg" | tg-notify.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"
[[ -f "$ENV_FILE" ]] || { echo "[tg-notify] no .env at $ENV_FILE" >&2; exit 0; }

# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

: "${TELEGRAM_BOT_TOKEN:?missing TELEGRAM_BOT_TOKEN}"
: "${TELEGRAM_CHAT_ID:?missing TELEGRAM_CHAT_ID}"

if [[ $# -gt 0 ]]; then
  MSG="$*"
else
  MSG="$(cat)"
fi

# Telegram MarkdownV2 escape minimal set (just enough; full escape brittle)
# Use plain text mode for safety.
curl -sS -X POST \
  "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
  -d chat_id="${TELEGRAM_CHAT_ID}" \
  -d text="${MSG}" \
  -d disable_web_page_preview=true \
  > /dev/null && echo "[tg-notify] sent" || echo "[tg-notify] failed" >&2
