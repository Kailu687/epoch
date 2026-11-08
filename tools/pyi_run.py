"""Run PyInstaller with a patient retry around exe post-processing.

On machines with real-time antivirus, a freshly written exe can stay locked for several seconds, and PyInstaller's built-in
retry (about 4 s in total) gives up. This wrapper retries for up to two minutes. Usage: python tools/pyi_run.py <pyinstaller args>
"""
import sys
import time

import PyInstaller.building.api as api
from PyInstaller.__main__ import run


def _patient_retry(func, *args, max_attempts=20):
    last = None
    for _ in range(60):
        try:
            return func(*args)
        except Exception as e:  # PermissionError / pywin32 error while the file is locked
            last = e
            time.sleep(2)
    raise RuntimeError(f"Execution of {func.__name__!r} failed after 2 minutes of retries") from last


api.EXE._retry_operation = staticmethod(_patient_retry)

if __name__ == "__main__":
    run(sys.argv[1:])
