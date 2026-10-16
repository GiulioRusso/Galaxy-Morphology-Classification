"""Download Galaxy10_DECals.h5 (2.7 GB) from the Zenodo record linked by astroNN and verify its MD5.

    python scripts/download_data.py [--out data]
"""
import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5"
MD5 = "c6b7b4db82b3a5d63d6a7e3e5249b51c"  # from https://zenodo.org/api/records/10845026


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    out = Path(p.parse_args().out) / "Galaxy10_DECals.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    for attempt in range(20):  # Zenodo connections sometimes hang: time out and resume from the partial file
        if out.exists():
            break
        done = tmp.stat().st_size if tmp.exists() else 0
        print(f"downloading {URL} -> {out} (from byte {done}, attempt {attempt + 1})", flush=True)
        req = urllib.request.Request(URL, headers={"Range": f"bytes={done}-"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab" if r.status == 206 else "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.rename(out)
        except (OSError, TimeoutError) as e:
            print(f"interrupted: {e}", flush=True)
    got = md5(out)
    if got != MD5:
        raise SystemExit(f"MD5 mismatch for {out}: {got} != {MD5}. Delete it and re-run.")
    print(f"ok: {out} md5={got}")


if __name__ == "__main__":
    main()
