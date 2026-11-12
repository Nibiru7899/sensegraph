"""Run with an existing Python. All installed runtimes and wheels stay local."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sensegraph.acquisition import Downloader


def run(args):
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def main():
    os.environ["PIP_CACHE_DIR"] = str(ROOT / ".cache/pip")
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    downloader = Downloader(ROOT)
    # Includes metadata fetched by pip and earlier source-inspection requests.
    downloader.allowance("metadata-and-pip-resolution", 100_000_000)
    runtime = ROOT / ".runtime/python311"
    interpreter = runtime / "python.exe"
    if not interpreter.exists():
        installer = downloader.fetch("https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe", ".runtime/python-3.11.9-amd64.exe")
        run([installer, "/quiet", "InstallAllUsers=0", f"TargetDir={runtime}", "PrependPath=0", "AssociateFiles=0", "Shortcuts=0", "Include_launcher=0", "Include_test=0"])
    venv = ROOT / ".venv/Scripts/python.exe"
    if not venv.exists():
        run([interpreter, "-m", "venv", ROOT / ".venv"])
    wheels = ROOT / ".cache/wheels"
    # Fetch official Windows wheels directly, avoiding a second torch resolution/download.
    for package, version in [("torch", "2.8.0"), ("torchvision", "0.23.0")]:
        index = downloader.fetch(f"https://download.pytorch.org/whl/cu128/{package}/", f"data/metadata/{package}-index.html")
        links = re.findall(r'href="([^\"]+)"', index.read_text())
        link = next(x for x in links if f"{package}-{version}%2Bcu128-cp311-cp311-win_amd64.whl" in x)
        url, _, fragment = link.partition("#sha256=")
        filename = urllib.request.url2pathname(url.rsplit("/", 1)[1])
        downloader.fetch(url, f".cache/wheels/{filename}", fragment or None)
    run([venv, "-m", "pip", "install", "--no-deps", "--no-index", "--find-links", wheels, "torch==2.8.0", "torchvision==0.23.0"])
    report = ROOT / "experiments/dependency_resolution.json"
    lock = ROOT / "requirements.lock"
    locked = ["-r", lock] if lock.exists() else []
    run([venv, "-m", "pip", "install", "--dry-run", "--only-binary=:all:", "--report", report, *locked, ".", "wheel", "setuptools>=68"])
    resolution = json.loads(report.read_text(encoding="utf-8"))
    for item in resolution["install"]:
        info = item["download_info"]
        if info.get("dir_info") is not None:
            continue
        url = info["url"]
        digest = info.get("archive_info", {}).get("hashes", {}).get("sha256")
        downloader.fetch(url, f".cache/wheels/{url.rsplit('/', 1)[1]}", digest)
    # Build backend is needed for the editable local project, not a network build environment.
    run([venv, "-m", "pip", "install", "--no-index", "--find-links", wheels, "wheel", "setuptools>=68"])
    run([venv, "-m", "pip", "install", "--no-index", "--find-links", wheels, "--no-build-isolation", *locked, "-e", "."])
    frozen = subprocess.check_output([str(venv), "-m", "pip", "freeze", "--all", "--exclude-editable"], text=True)
    (ROOT / "requirements.lock").write_text(frozen, encoding="utf-8")
    print(f"Environment ready. Accounted transfers: {downloader.ledger['transferred_bytes']} bytes")


if __name__ == "__main__":
    main()
