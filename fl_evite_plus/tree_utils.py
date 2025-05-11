import json
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=1)
def load_tree(path: str) -> dict[int, list[int]]:
    """Return parent→children mapping (values as int)."""
    with open(Path(path), "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {int(p): [int(c) for c in kids] for p, kids in raw.items()}

def all_nodes(tree: dict[int, list[int]]) -> set[int]:
    parents = set(tree.keys())
    children = {c for kids in tree.values() for c in kids}
    return parents | children
