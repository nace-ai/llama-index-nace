"""Fail when the `_drex/` copies in the packages are not byte-identical."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tree(directory: Path) -> dict[str, bytes]:
    """Map each file's path inside `directory` to its bytes, skipping caches."""
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def main(root: Path = ROOT) -> int:
    """Compare every `_drex/` tree with the first; print differences and return the exit code."""
    copies = sorted(root.glob("packages/*/llama_index/*/nace/_drex"))
    if len(copies) < 2:
        print(f"{len(copies)} _drex copy found; nothing to compare.")
        return 0
    reference = copies[0]
    expected = tree(reference)
    differing: list[str] = []
    for other in copies[1:]:
        actual = tree(other)
        for name in sorted(expected.keys() | actual.keys()):
            if expected.get(name) != actual.get(name):
                differing.append(
                    f"{name} ({reference.parent.parent.parent.parent.name} vs "
                    f"{other.parent.parent.parent.parent.name})"
                )
    if differing:
        print("_drex copies differ:")
        for line in differing:
            print(f"  {line}")
        return 1
    print(f"{len(copies)} _drex copies are identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
