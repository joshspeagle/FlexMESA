"""Fill Git LFS pointer files in an upstream MESA checkout from local copies.

A checkout made with ``GIT_LFS_SKIP_SMUDGE=1`` holds small pointer files in
place of the real data. Each pointer records the SHA-256 and size of the
object. This script looks for a file with the same relative path under each
source directory and copies it into place only when its size and SHA-256
match the pointer, so every filled file is verified.

Usage:
    python fill_lfs.py <mesa checkout> <source dir> [<source dir> ...]
"""

import hashlib
import os
import shutil
import sys

POINTER_PREFIX = b"version https://git-lfs"


def sha256_of(path: str) -> str:
    """Return the hex SHA-256 digest of the file at ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_pointers(root: str) -> dict[str, tuple[str, int]]:
    """Map each pointer file (path relative to ``root``) to (oid, size)."""
    pointers: dict[str, tuple[str, int]] = {}
    for dirpath, _, names in os.walk(root):
        if ".git" in dirpath.split(os.sep):
            continue
        for name in names:
            path = os.path.join(dirpath, name)
            try:
                if os.path.getsize(path) > 300:
                    continue
                with open(path, "rb") as fh:
                    text = fh.read()
            except OSError:
                continue
            if text.startswith(POINTER_PREFIX):
                fields = dict(
                    line.split(" ", 1)
                    for line in text.decode().strip().split("\n")
                )
                oid = fields["oid"].split(":")[1]
                rel = os.path.relpath(path, root)
                pointers[rel] = (oid, int(fields["size"]))
    return pointers


def main() -> None:
    root, sources = sys.argv[1], sys.argv[2:]
    pointers = find_pointers(root)
    print("pointers", len(pointers))
    filled = 0
    missing = []
    for rel, (oid, size) in sorted(pointers.items()):
        for source in sources:
            cand = os.path.join(source, rel)
            if (
                os.path.isfile(cand)
                and os.path.getsize(cand) == size
                and sha256_of(cand) == oid
            ):
                shutil.copyfile(cand, os.path.join(root, rel))
                filled += 1
                break
        else:
            missing.append((rel, oid, size))
    print("filled", filled, "missing", len(missing))
    for item in missing:
        print("MISSING", *item)


if __name__ == "__main__":
    main()
