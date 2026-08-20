from __future__ import annotations

import os
import sys
import threading
import time

from .runs import load_run_spec, update_status
from .service import run_agent


def _heartbeat(run_id: str, stop: threading.Event) -> None:
    while not stop.wait(5):
        update_status(run_id, heartbeat_at=time.time(), pid=os.getpid())


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if not args:
        print("usage: python -m liverkg_cli.worker RUN_ID [--resume]", file=sys.stderr)
        return 2
    run_id = args[0]
    resume = "--resume" in args[1:]
    stop = threading.Event()
    thread = threading.Thread(target=_heartbeat, args=(run_id, stop), daemon=True)
    thread.start()
    try:
        spec = load_run_spec(run_id)
        return run_agent(spec, resume=resume)
    except KeyboardInterrupt:
        update_status(run_id, state="stopped", exit_code=130)
        return 130
    except Exception as exc:
        update_status(run_id, state="failed", error=str(exc), exit_code=1)
        print(f"[LiverKG worker] {exc}", file=sys.stderr)
        return 1
    finally:
        stop.set()


if __name__ == "__main__":
    raise SystemExit(main())
