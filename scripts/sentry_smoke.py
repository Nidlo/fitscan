"""Sentry smoke test — fitscan side.

One-shot probe to verify SENTRY_DSN is wired correctly. Mirrors the
backend's ``php artisan sentry:smoke-test`` so an operator can verify
both services emit events with the same single-command UX.

Usage:

    python scripts/sentry_smoke.py
    python scripts/sentry_smoke.py --message "custom probe"

Behaviour:
- DSN unset → exits 0 with a friendly skip message (so CI / dev installs
  that haven't provisioned a Sentry project yet stay green).
- DSN set → fires a captureMessage tagged ``source=cli`` /
  ``command=sentry-smoke``, prints the event id, and reminds the
  operator to look in the Sentry UI within ~30s.

Exit codes:
- 0 — skipped (no DSN) or event sent.
- 1 — sentry-sdk not installed or capture failed.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fire a Sentry test event so an operator can verify "
        "the DSN is wired correctly.",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="Override the test message body.",
    )
    args = parser.parse_args()

    dsn = (os.getenv("SENTRY_DSN") or "").strip()
    if not dsn:
        print("SENTRY_DSN is not set — skipping. Set it in fitscan/.env to enable Sentry.")
        print("Once set, re-run: python scripts/sentry_smoke.py")
        return 0

    try:
        import sentry_sdk
    except ImportError:
        print("sentry-sdk is not installed. Run `make setup` to install dependencies.", file=sys.stderr)
        return 1

    sentry_sdk.init(
        dsn=dsn,
        environment=os.getenv("APP_ENV", "development"),
        traces_sample_rate=0.0,
        send_default_pii=False,
    )

    message = args.message or (
        "fitscan sentry-smoke — " + datetime.now(timezone.utc).isoformat()
    )

    with sentry_sdk.push_scope() as scope:
        scope.set_tag("source", "cli")
        scope.set_tag("command", "sentry-smoke")
        scope.set_tag("service", "fitscan")
        scope.set_tag("environment", os.getenv("APP_ENV", "development"))
        event_id = sentry_sdk.capture_message(message)

    sentry_sdk.flush(timeout=5.0)

    if not event_id:
        print("capture_message returned no event id. Sentry SDK may not be initialised.", file=sys.stderr)
        return 1

    print(f"Sentry test event sent. Event ID: {event_id}")
    print("Open https://sentry.io/issues/?query=is:unresolved and look for:")
    print(f'  "{message}"')
    print("If it appears within ~30s, Sentry is wired correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
