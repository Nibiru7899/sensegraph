"""Install the bounded structural-data dependencies, keeping the baseline lock intact."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sensegraph.acquisition import Downloader
from sensegraph.io import read_json


def main():
    os.environ["PIP_CACHE_DIR"] = str(ROOT / ".cache/pip")
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    downloader = Downloader(ROOT)
    downloader.allowance("milestone2-dependency-resolution", 100_000_000)
    report = ROOT / "experiments/milestone2/dependencies.json"
    lock = ROOT / "requirements-structural.lock"
    locked = ["-r", str(lock)] if lock.exists() else []
    subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--only-binary=:all:", "--report", str(report), *locked, ".[structural]"], cwd=ROOT, check=True)
    for item in read_json(report)["install"]:
        info = item["download_info"]
        if "dir_info" in info:
            continue
        url = info["url"]
        downloader.fetch(url, ".cache/wheels/" + url.rsplit("/", 1)[-1], info["archive_info"]["hashes"]["sha256"])
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(ROOT / ".cache/wheels"), "--no-build-isolation", *locked, "-e", ".[structural]"], cwd=ROOT, check=True)
    locked = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all", "--exclude-editable"], text=True)
    (ROOT / "requirements-structural.lock").write_text(locked, encoding="utf-8")


if __name__ == "__main__":
    main()
