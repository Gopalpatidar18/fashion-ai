#!/bin/sh
set -eu

PYTHON=""
for candidate in \
  "$HOME/.local/bin/python3.12" \
  "$HOME/.local/bin/python3.11" \
  "$HOME/.local/bin/python3.10" \
  python3.12 python3.11 python3.10 python3
 do
  if [ -n "$candidate" ] && ( [ -x "$candidate" ] || command -v "$candidate" >/dev/null 2>&1 ); then
    if "$candidate" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)' 2>/dev/null; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  printf '%s\n' 'Python 3.10, 3.11, or 3.12 is required. Install Python and run this script again.' >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  "$PYTHON" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if [ ! -f fashion_model.keras ]; then
  .venv/bin/python train_model.py
fi

exec .venv/bin/python app.py