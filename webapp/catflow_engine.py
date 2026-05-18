"""
CatFlow inference engine for the web app.

Loads the de-novo-generation and structure-prediction checkpoints and exposes
a single `generate()` call that:
  * builds a model input batch from a chosen adsorbate (+ composition),
  * runs 50-step flow-matching sampling for 10 samples with full trajectory,
  * assembles every sample / trajectory frame into an ASE structure,
  * keeps only the structurally valid final samples and, for each, only the
    valid intermediate trajectory frames.
"""
from __future__ import annotations

import io
import sys
import threading
import uuid
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write as ase_write

# --- make the CatFlow package importable -----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# PyTorch 2.6+ : checkpoints were saved with full pickles.
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw["weights_only"] = False
    return _orig_load(*a, **kw)
torch.load = _patched_load

from src.module.effcat_module import EffCatModule              # noqa: E402
from src.data.lmdb_dataset import collate_fn_with_dynamic_padding  # noqa: E402
from scripts.sampling.assemble import assemble                 # noqa: E402
from src.models.loss.validation import (                       # noqa: E402
    compute_structural_validity_single,
)

CKPT_DIR = Path(__file__).resolve().parent / "ckpts"

N_SAMPLES = 10
SAMPLING_STEPS = 50


# ---------------------------------------------------------------------------
# structure -> PDB helpers
# ---------------------------------------------------------------------------
def atoms_to_pdb(atoms: Atoms) -> str:
    buf = io.StringIO()
    ase_write(buf, atoms, format="proteindatabank")
    return buf.getvalue()


def structure_payload(atoms: Atoms) -> dict:
    """JSON-serialisable description of one structure for the frontend."""
    tags = atoms.get_tags().tolist() if atoms.has("tags") else [0] * len(atoms)
    cell = atoms.get_cell()
    # 2x2x1 supercell, built in ASE so cell + coordinates stay in one frame.
    # Doing the tiling here (rather than shifting copies in JS by the cell
    # vectors) avoids the PDB CRYST1 round-trip bug: ase_write rotates atoms
    # into a standard cell orientation, so a JS shift by the original,
    # un-rotated cell vectors lands the replicas in the wrong frame.
    return {
        "pdb": atoms_to_pdb(atoms),
        "pdb_super": atoms_to_pdb(atoms.repeat((2, 2, 1))),
        "tags": tags,
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "cell": [[float(x) for x in v] for v in np.asarray(cell)],
    }


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
class CatFlowEngine:
    def __init__(self, device: str = "cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()        # models are not thread-safe
        self.dng_model = None
        self.sp_model = None
        # job store: job_id -> {sample_idx -> ASE Atoms (final)}
        self.jobs: dict[str, dict] = {}

    # -- model loading -------------------------------------------------------
    def _load(self, ckpt_path: Path) -> EffCatModule:
        model = EffCatModule.load_from_checkpoint(
            str(ckpt_path), map_location="cpu")
        model.eval()
        model.to(self.device)
        return model

    def load_models(self):
        dng_ckpt = self._find_ckpt("dng")
        print(f"[engine] loading de-novo model ({dng_ckpt.name}) "
              f"on {self.device} ...", flush=True)
        self.dng_model = self._load(dng_ckpt)
        sp_ckpt = self._find_ckpt("sp")
        print(f"[engine] loading structure-prediction model "
              f"({sp_ckpt.name}) ...", flush=True)
        self.sp_model = self._load(sp_ckpt)
        print("[engine] models ready.", flush=True)

    def _find_ckpt(self, subdir: str) -> Path:
        """Locate the .ckpt under ckpts/<subdir>/ (any nesting)."""
        cands = sorted((CKPT_DIR / subdir).rglob("*.ckpt"))
        if not cands:
            raise FileNotFoundError(
                f"No checkpoint found under {CKPT_DIR / subdir}/ — download and "
                f"extract the checkpoint tar there (see webapp/HANDOFF.md).")
        non_last = [c for c in cands if "last" not in c.name.lower()]
        return (non_last or cands)[0]

    # -- batch construction --------------------------------------------------
    def _raw_sample(self, prim_numbers, prim_positions, prim_cell,
                    ads_numbers, ads_positions) -> dict:
        """One LMDB-style record consumable by collate_fn_with_dynamic_padding."""
        primitive_slab = Atoms(
            numbers=prim_numbers, positions=prim_positions,
            cell=prim_cell, pbc=True)
        ads_pos = np.asarray(ads_positions, dtype=float).reshape(-1, 3)
        return {
            "primitive_slab": primitive_slab,
            "supercell_matrix": np.eye(3, dtype=float),
            "n_slab": 1,
            "n_vac": 1,
            "ads_atomic_numbers": np.asarray(ads_numbers, dtype=int),
            "ads_pos": ads_pos,
            "ref_ads_pos": ads_pos,
            "bind_ads_atom_symbol": 0,
            "ref_energy": 0.0,
        }

    def _to_device(self, batch: dict) -> dict:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    # -- public API ----------------------------------------------------------
    @torch.no_grad()
    def generate(self, mode: str, adsorbate: dict,
                 composition: dict | None = None,
                 progress=None) -> dict:
        """
        mode        : "denovo" or "structure"
        adsorbate   : entry from adsorbates.json
        composition : entry from compositions.json (required for "structure")
        progress    : optional callable(str) for status updates
        """
        def say(msg):
            if progress:
                progress(msg)

        with self._lock:
            if mode == "denovo":
                model = self.dng_model
                # primitive slab is regenerated by the model; a 1-atom dummy
                # is enough to define the batch.
                raw = self._raw_sample(
                    prim_numbers=[1], prim_positions=[[0.0, 0.0, 0.0]],
                    prim_cell=np.eye(3) * 5.0,
                    ads_numbers=adsorbate["numbers"],
                    ads_positions=adsorbate["positions"])
            elif mode == "structure":
                if composition is None:
                    raise ValueError("composition required for structure mode")
                model = self.sp_model
                nums = composition["numbers"]
                # arbitrary positions: coordinates are generated by the model,
                # only the element identities condition the prediction.
                pos = np.random.RandomState(0).rand(len(nums), 3) * 8.0
                raw = self._raw_sample(
                    prim_numbers=nums, prim_positions=pos,
                    prim_cell=np.eye(3) * 10.0,
                    ads_numbers=adsorbate["numbers"],
                    ads_positions=adsorbate["positions"])
            else:
                raise ValueError(f"unknown mode: {mode}")

            # Both modes (de-novo dng=True and structure-prediction dng=False)
            # use the same scheme: build the batch as N_SAMPLES identical rows
            # (batch_size=N_SAMPLES, multiplicity=1). This yields N_SAMPLES
            # independent samples and avoids the multiplicity>1 code path,
            # which has a shape bug in the dng=False encoder. For dng=True the
            # per-row atom count is still histogram-sampled independently, so
            # the 10 samples differ even though the input rows are identical.
            batch = collate_fn_with_dynamic_padding([raw] * N_SAMPLES)
            batch = self._to_device(batch)

            say(f"Running flow matching ({SAMPLING_STEPS} steps, "
                f"{N_SAMPLES} samples)…")
            out = model(
                batch,
                num_sampling_steps=SAMPLING_STEPS,
                center_during_sampling=False,
                multiplicity_flow_sample=1,
                return_trajectory=True,
            )

            say("Assembling and filtering structures…")
            result = self._postprocess(mode, model, batch, out)
        return result

    # -- assembly + validity -------------------------------------------------
    def _postprocess(self, mode, model, batch, out) -> dict:
        def npy(x):
            return x.detach().cpu().numpy() if torch.is_tensor(x) else x

        prim_coords = npy(out["sampled_prim_slab_coords"])      # (M, N, 3)
        ads_coords = npy(out["sampled_ads_coords"])             # (M, A, 3)
        lattices = npy(out["sampled_lattice"])                  # (M, 6)
        sc_mats = npy(out["sampled_supercell_matrix"])          # (M, 3, 3)
        scales = npy(out["sampled_scaling_factor"])             # (M,)
        n_samples = prim_coords.shape[0]

        ads_types = npy(batch["ref_ads_element"])[0]            # (A,)
        ads_mask = npy(batch["ads_atom_pad_mask"])              # may be (1|M, A)

        is_dng = bool(getattr(model, "dng", False))
        if is_dng:
            prim_types_all = npy(out["sampled_prim_slab_element"])   # (M, N)
        else:
            prim_types_all = npy(batch["ref_prim_slab_element"])     # (M, N)
        prim_mask_all = npy(batch["prim_slab_atom_pad_mask"])        # (M, N)

        # trajectories: (steps+1, M, ...)
        traj_prim = npy(out["prim_slab_coord_trajectory"])
        traj_ads = npy(out["ads_coord_trajectory"])
        traj_lat = npy(out["lattice_trajectory"])
        traj_sc = npy(out["supercell_matrix_trajectory"])
        traj_scale = npy(out["scaling_factor_trajectory"])
        traj_elem = npy(out["prim_slab_element_trajectory"]) if (
            is_dng and "prim_slab_element_trajectory" in out) else None
        n_steps = traj_prim.shape[0]

        def get_prim(idx):
            return prim_types_all[idx], prim_mask_all[idx]

        def get_ads_mask(idx):
            return ads_mask[idx] if ads_mask.shape[0] > idx else ads_mask[0]

        def check_valid(sys_prim, sys_ads, lat, sc, scale,
                        ptypes, atypes, pmask, amask):
            task = (
                sys_prim[None], sys_ads[None], lat[None],
                sc.reshape(3, 3)[None], np.array([scale]),
                ptypes, atypes, pmask, amask,
            )
            res = compute_structural_validity_single(task)
            return bool(res[0])

        samples = []
        for i in range(n_samples):
            ptypes, pmask = get_prim(i)
            amask = get_ads_mask(i)
            try:
                final_sys, _ = assemble(
                    generated_prim_slab_coords=prim_coords[i],
                    generated_ads_coords=ads_coords[i],
                    generated_lattice=lattices[i],
                    generated_supercell_matrix=sc_mats[i].reshape(3, 3),
                    generated_scaling_factor=float(scales[i]),
                    prim_slab_atom_types=ptypes,
                    ads_atom_types=ads_types,
                    prim_slab_atom_mask=pmask,
                    ads_atom_mask=amask,
                )
            except Exception:
                continue

            valid = check_valid(prim_coords[i], ads_coords[i], lattices[i],
                                sc_mats[i], float(scales[i]),
                                ptypes, ads_types, pmask, amask)
            if not valid:
                continue

            # valid trajectory frames for this sample
            frames = []
            for s in range(n_steps):
                step_ptypes = (traj_elem[s, i] if traj_elem is not None
                               else ptypes)
                try:
                    fsys, _ = assemble(
                        generated_prim_slab_coords=traj_prim[s, i],
                        generated_ads_coords=traj_ads[s, i],
                        generated_lattice=traj_lat[s, i],
                        generated_supercell_matrix=traj_sc[s, i].reshape(3, 3),
                        generated_scaling_factor=float(traj_scale[s, i]),
                        prim_slab_atom_types=step_ptypes,
                        ads_atom_types=ads_types,
                        prim_slab_atom_mask=pmask,
                        ads_atom_mask=amask,
                    )
                except Exception:
                    continue
                # always keep the first / last; keep interior frames only if
                # structurally valid (per the user spec).
                keep = s in (0, n_steps - 1)
                if not keep:
                    keep = check_valid(
                        traj_prim[s, i], traj_ads[s, i], traj_lat[s, i],
                        traj_sc[s, i], float(traj_scale[s, i]),
                        step_ptypes, ads_types, pmask, amask)
                if keep:
                    frames.append((s, fsys))

            samples.append({
                "atoms": final_sys,
                "final": structure_payload(final_sys),
                "trajectory": [
                    {"step": s, **structure_payload(a)} for s, a in frames],
                "n_traj_total": n_steps,
            })

        job_id = uuid.uuid4().hex[:12]
        self.jobs[job_id] = {i: s["atoms"] for i, s in enumerate(samples)}
        # keep the job store bounded (beta service)
        while len(self.jobs) > 60:
            self.jobs.pop(next(iter(self.jobs)))

        return {
            "job_id": job_id,
            "mode": mode,
            "n_generated": n_samples,
            "n_valid": len(samples),
            "samples": [
                {k: v for k, v in s.items() if k != "atoms"}
                for s in samples
            ],
        }

    def get_structure(self, job_id: str, sample_idx: int) -> Atoms:
        if job_id not in self.jobs or sample_idx not in self.jobs[job_id]:
            raise KeyError("structure not found (job expired?)")
        return self.jobs[job_id][sample_idx].copy()
