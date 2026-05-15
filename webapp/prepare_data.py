"""
Prepare static menu data for the CatFlow web app from the CatFlow dataset.

Sources (extracted from data.tar.gz under webapp/_newdata/data/):
  * dataset_per_adsorbate/<symbol>.lmdb   -> adsorbate menu (67 adsorbates),
    each gives the adsorbate species + canonical reference geometry.
  * dataset/val_id/dataset.lmdb           -> slab primitive-cell composition
    menu for structure prediction (real factorized primitive cells).

Produces webapp/data/{adsorbates.json, compositions.json}.

Run once before starting the server:
    python prepare_data.py
"""
from __future__ import annotations

import json
import pickle
import warnings
from collections import Counter
from functools import reduce
from math import gcd
from pathlib import Path

import lmdb
import numpy as np
from ase.data import chemical_symbols

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

DATASET_ROOT = HERE / "_newdata" / "data"
PER_ADS_DIR = DATASET_ROOT / "dataset_per_adsorbate"
COMPOSITION_LMDB = DATASET_ROOT / "dataset" / "val_id" / "dataset.lmdb"

# Elements with a gas-phase reference energy (adsorption energy well defined).
EVALUABLE_ELEMENTS = {"H", "C", "N", "O"}


def subscript(formula: str) -> str:
    return formula.translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def first_entry(lmdb_path: Path) -> dict | None:
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    try:
        with env.begin() as txn:
            for key, val in txn.cursor():
                if key.decode("ascii") == "length":
                    continue
                return pickle.loads(val)
    finally:
        env.close()
    return None


def formula_from_numbers(numbers: list[int]) -> tuple[str, str]:
    """(raw formula, reduced formula) from a list of atomic numbers."""
    cnt = Counter(chemical_symbols[z] for z in numbers)
    raw = "".join(f"{el}{cnt[el]}" for el in sorted(cnt))
    g = reduce(gcd, cnt.values())
    red = "".join(
        f"{el}{cnt[el] // g if cnt[el] // g > 1 else ''}" for el in sorted(cnt))
    return raw, red


def build_adsorbates() -> list[dict]:
    """One adsorbate per dataset_per_adsorbate/<symbol>.lmdb file."""
    out = []
    for path in sorted(PER_ADS_DIR.glob("*.lmdb")):
        symbol = path.name[:-len(".lmdb")]          # e.g. "*CO", "*N*NH"
        entry = first_entry(path)
        if entry is None:
            continue
        numbers = [int(z) for z in np.asarray(entry["ads_atomic_numbers"])]
        # canonical reference geometry used to condition the flow model
        positions = np.asarray(entry["ref_ads_pos"], dtype=float).reshape(-1, 3)
        if len(numbers) != len(positions) or not numbers:
            continue
        raw, _ = formula_from_numbers(numbers)
        symbols = [chemical_symbols[z] for z in numbers]
        out.append({
            "id": symbol,                            # stable id = the symbol
            "name": symbol,
            "formula": raw,
            "display": subscript(raw),
            "n_atoms": len(numbers),
            "numbers": numbers,
            "positions": [[float(x) for x in p] for p in positions],
            "evaluable": set(symbols).issubset(EVALUABLE_ELEMENTS),
        })
    out.sort(key=lambda a: (a["n_atoms"], a["name"]))
    return out


def build_compositions(max_keep: int = 300) -> list[dict]:
    """Unique slab compositions from the val_id dataset.

    Deduplicated by reduced formula (so 'Pt' appears once, not Pt1/Pt2/Pt4);
    for each, a real primitive cell of moderate size is kept as the conditioning
    composition.
    """
    env = lmdb.open(str(COMPOSITION_LMDB), subdir=False, readonly=True,
                    lock=False, readahead=False, meminit=False)
    # reduced formula -> list of candidate primitive cells (atomic-number lists)
    groups: dict[str, list[list[int]]] = {}
    scanned = 0
    with env.begin() as txn:
        for key, val in txn.cursor():
            if key.decode("ascii") == "length":
                continue
            scanned += 1
            try:
                e = pickle.loads(val)
                numbers = [int(z) for z in
                           e["primitive_slab"].get_atomic_numbers()]
                if not (1 <= len(numbers) <= 48):
                    continue
                _, red = formula_from_numbers(numbers)
                groups.setdefault(red, []).append(sorted(numbers))
            except Exception:
                continue
    env.close()

    out = []
    for red, cells in groups.items():
        # representative: smallest cell with >= 4 atoms if available, else
        # the smallest cell overall (keeps conditioning in-distribution).
        cells.sort(key=len)
        rep = next((c for c in cells if len(c) >= 4), cells[0])
        cnt = Counter(chemical_symbols[z] for z in rep)
        raw = "".join(f"{el}{cnt[el]}" for el in sorted(cnt))
        out.append({
            "id": raw,
            "formula": red,
            "raw": raw,
            "display": subscript(raw),
            "n_atoms": len(rep),
            "numbers": rep,
            "elements": sorted(cnt.keys()),
        })
    # elemental metals first, then alloys; alphabetical within each group
    out.sort(key=lambda c: (len(c["elements"]), c["formula"]))
    print(f"  scanned {scanned} val_id entries -> {len(out)} unique reduced "
          f"compositions (keeping {min(len(out), max_keep)})")
    return out[:max_keep]


def main():
    print("[1/2] Building adsorbate menu from dataset_per_adsorbate/ ...")
    adsorbates = build_adsorbates()
    (DATA_DIR / "adsorbates.json").write_text(json.dumps(adsorbates, indent=1))
    print(f"  wrote {len(adsorbates)} adsorbates")

    print("[2/2] Building slab composition menu from val_id dataset ...")
    compositions = build_compositions()
    (DATA_DIR / "compositions.json").write_text(
        json.dumps(compositions, indent=1))
    print(f"  wrote {len(compositions)} compositions")
    print("Done.")


if __name__ == "__main__":
    main()
