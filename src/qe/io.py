"""Lokasi berkas dan penulisan hasil."""

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
REPORTS = os.path.join(ROOT, "reports")
CONFIGS = os.path.join(ROOT, "configs")


def bootstrap_path():
    """Tambahkan src/ ke sys.path supaya skrip bisa dijalankan langsung."""
    src = os.path.join(ROOT, "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def write_json(relpath: str, obj) -> str:
    path = os.path.join(RESULTS, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")
    return path


def read_json(relpath: str):
    with open(os.path.join(RESULTS, relpath), encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(relpath: str, rows, fieldnames=None) -> str:
    import csv

    path = os.path.join(RESULTS, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = list(rows)
    fieldnames = fieldnames or (list(rows[0]) if rows else [])
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_report(name: str, text: str) -> str:
    path = os.path.join(REPORTS, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    return path


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace(os.sep, "/")
