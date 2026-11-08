"""Build a Windows one-folder app with PyInstaller, smoke-test it, zip it, and print (not publish) its checksum.

Usage: python tools/build_release.py
Outputs: build/Epoch/ (frozen app), build/Epoch-<version>-win-x64.zip, and build/build_report.json.
This script NEVER edits releases/releases.json: adding a release entry is a deliberate, separate step (see HANDOVER).
"""
import hashlib
import os
import json
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
OUT = Path(os.environ["EPOCH_BUILD_DIR"]) if os.environ.get("EPOCH_BUILD_DIR") else ROOT / "build"  # a scratch dir avoids antivirus locks on fresh exes


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    version = re.search(r'__version__ = "([^"]+)"', (APP / "engine_trainer" / "__init__.py").read_text()).group(1)
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "pyi_run.py"), "--noconfirm", "--distpath", str(OUT), "--workpath", str(OUT / "work"),
                    str(APP / "engine_trainer.spec")], cwd=APP, check=True)
    exe = OUT / "Epoch" / "EpochCLI.exe"
    gui_exe = OUT / "Epoch" / "Epoch.exe"
    report = {"version": version, "built_on": platform.platform(), "python": platform.python_version(),
              "exe_exists": exe.is_file() and gui_exe.is_file()}
    # smoke test the FROZEN binary (includes spawning the training worker process)
    r = subprocess.run([str(exe), "--selftest"], capture_output=True, text=True, timeout=600)
    report["selftest_returncode"] = r.returncode
    report["selftest_output"] = (r.stdout + r.stderr).strip()[-500:]
    if r.returncode != 0:
        (OUT / "build_report.json").write_text(json.dumps(report, indent=2))
        sys.exit("frozen selftest FAILED:\n" + report["selftest_output"])
    # GUI launcher must stay alive for a few seconds (no crash on start), then is killed
    import time
    g = subprocess.Popen([str(gui_exe)])
    time.sleep(10)
    report["gui_alive_after_10s"] = g.poll() is None
    g.kill()
    if not report["gui_alive_after_10s"]:
        (OUT / "build_report.json").write_text(json.dumps(report, indent=2))
        sys.exit("frozen GUI exited during start-up")
    zpath = OUT / f"Epoch-{version}-win-x64.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((OUT / "Epoch").rglob("*")):
            if f.is_file():
                z.write(f, Path("Epoch") / f.relative_to(OUT / "Epoch"))
    with zipfile.ZipFile(zpath) as z:
        report["zip_test"] = z.testzip() is None
        report["zip_entries"] = len(z.namelist())
        report["zip_has_exe"] = all(f"Epoch/{n}.exe" in z.namelist() for n in ("Epoch", "EpochCLI"))
    report.update(zip=str(zpath), size=zpath.stat().st_size, sha256=sha(zpath))
    shutil.rmtree(OUT / "work", ignore_errors=True)
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
