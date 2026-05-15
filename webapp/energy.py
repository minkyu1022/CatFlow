"""
Adsorption-energy evaluation with the UMA (uma-s-1p1) machine-learned potential.

Mirrors CatFlow's scripts/relax_energy/E_ads_eval_batch.py:
  * fix the bulk atoms (tag 0), relax surface + adsorbate with LBFGS,
  * E_ads = E_system - E_slab - E_adsorbate(gas-phase reference).
"""
from __future__ import annotations

import contextlib
import io
import threading
import warnings

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import LBFGS

warnings.filterwarnings("ignore")

# Gas-phase reference energies (eV) — computed with UMA, from CatFlow.
OC20_GAS_PHASE_ENERGIES = {
    "H": -3.48483361833793,
    "O": -7.185616160375758,
    "C": -7.232295041080779,
    "N": -8.09079187764214,
}

FMAX = 0.05
MAX_STEPS = 300

# Adsorbates for which we draw an illustrative activity volcano.
# name -> (descriptor label, optimal binding energy eV, curve width eV)
VOLCANO_SPECS = {
    "*OH":  ("ΔE(*OH)  — ORR/OER descriptor", 0.10, 0.9),
    "*O":   ("ΔE(*O)  — oxygen evolution descriptor", 1.60, 1.1),
    "*OOH": ("ΔE(*OOH)  — ORR descriptor", 3.20, 1.0),
    "*H":   ("ΔE(*H)  — hydrogen evolution descriptor", 0.00, 0.6),
    "*CO":  ("ΔE(*CO)  — CO2 reduction descriptor", -0.60, 0.9),
    "*N":   ("ΔE(*N)  — nitrogen reduction descriptor", -0.20, 1.0),
}


def adsorbate_reference_energy(adsorbate: Atoms) -> float:
    total = 0.0
    for sym in adsorbate.get_chemical_symbols():
        if sym not in OC20_GAS_PHASE_ENERGIES:
            raise ValueError(f"no gas-phase reference for element '{sym}'")
        total += OC20_GAS_PHASE_ENERGIES[sym]
    return total


class EnergyEvaluator:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self._lock = threading.Lock()
        self.predict_unit = None

    def load(self):
        import torch
        from fairchem.core import pretrained_mlip
        # UMA's loader only accepts the bare strings "cpu" / "cuda".
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[energy] loading UMA potential (uma-s-1p1) on {dev} ...",
              flush=True)
        self.predict_unit = pretrained_mlip.get_predict_unit(
            "uma-s-1p1", device=dev)
        print("[energy] UMA ready.", flush=True)

    def _calc(self):
        from fairchem.core import FAIRChemCalculator
        return FAIRChemCalculator(self.predict_unit, task_name="oc20")

    def evaluate(self, system: Atoms, progress=None) -> dict:
        """Relax `system` (tags: 0 bulk / 1 surface / 2 adsorbate) and return
        adsorption-energy results."""
        def say(m):
            if progress:
                progress(m)

        with self._lock:
            tags = np.asarray(system.get_tags())
            if not np.any(tags == 2):
                raise ValueError("structure has no adsorbate atoms (tag 2)")

            system = system.copy()
            system.center()
            slab = system.copy()[tags != 2]
            adsorbate = system.copy()[tags == 2]

            sys_tags = np.asarray(system.get_tags())
            if np.any(sys_tags == 0):
                system.set_constraint(FixAtoms(
                    indices=[i for i, t in enumerate(sys_tags) if t == 0]))
            slab_tags = np.asarray(slab.get_tags())
            if np.any(slab_tags == 0):
                slab.set_constraint(FixAtoms(
                    indices=[i for i, t in enumerate(slab_tags) if t == 0]))

            calc = self._calc()
            system.calc = calc
            slab.calc = calc

            say("Computing initial energy…")
            e_sys_initial = float(system.get_potential_energy())

            say("Relaxing slab + adsorbate system…")
            opt = LBFGS(system, logfile=None)
            with contextlib.redirect_stdout(io.StringIO()):
                opt.run(fmax=FMAX, steps=MAX_STEPS)
            e_sys_relaxed = float(system.get_potential_energy())
            steps_sys = opt.get_number_of_steps()

            say("Relaxing clean slab…")
            opt_slab = LBFGS(slab, logfile=None)
            with contextlib.redirect_stdout(io.StringIO()):
                opt_slab.run(fmax=FMAX, steps=MAX_STEPS)
            e_slab_relaxed = float(slab.get_potential_energy())

            e_adsorbate = adsorbate_reference_energy(adsorbate)

            e_ads_initial = e_sys_initial - (e_slab_relaxed + e_adsorbate)
            e_ads_relaxed = e_sys_relaxed - (e_slab_relaxed + e_adsorbate)

            relaxed_system = system.copy()
            relaxed_system.calc = None

        from catflow_engine import structure_payload
        slab_part = relaxed_system[
            np.asarray(relaxed_system.get_tags()) != 2]
        return {
            "e_ads_initial": e_ads_initial,
            "e_ads_relaxed": e_ads_relaxed,
            "e_sys_initial": e_sys_initial,
            "e_sys_relaxed": e_sys_relaxed,
            "e_slab_relaxed": e_slab_relaxed,
            "e_adsorbate_ref": e_adsorbate,
            "relax_steps": int(steps_sys),
            "slab_formula": slab_part.get_chemical_formula(),
            "adsorbate_formula": adsorbate.get_chemical_formula(),
            "relaxed_structure": structure_payload(relaxed_system),
        }


def volcano_data(adsorbate_name: str, e_ads: float) -> dict | None:
    """Return illustrative activity-volcano data for supported adsorbates."""
    spec = VOLCANO_SPECS.get(adsorbate_name)
    if spec is None:
        return None
    label, optimum, width = spec

    xs = np.linspace(optimum - 3 * width, optimum + 3 * width, 81)
    # symmetric activity volcano (inverted V) about the optimum
    ys = (1.0 - np.abs(xs - optimum) / (3 * width)).clip(0.0, 1.0)
    point_activity = float(
        (1.0 - abs(e_ads - optimum) / (3 * width)))
    point_activity = max(0.0, min(1.0, point_activity))
    return {
        "descriptor": label,
        "optimum": optimum,
        "curve_x": [float(x) for x in xs],
        "curve_y": [float(y) for y in ys],
        "point_x": float(e_ads),
        "point_y": point_activity,
    }
