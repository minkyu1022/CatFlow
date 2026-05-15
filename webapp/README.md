# CatFlow Web Demo (beta)

A mobile-first web UI that serves the CatFlow flow-matching models together
with the UMA potential for on-the-fly catalyst generation and energy
evaluation.

## What it does

**De novo generation** — pick an adsorbate; the model generates the whole
slab–adsorbate system (composition + structure).

**Structure prediction** — pick an adsorbate *and* a slab composition; the
model predicts the structure for that fixed composition. The composition can
be chosen from the dataset menu **or specified freely** (e.g. `Pt3Ni`).

The adsorbate menu (67 species) and the dataset composition menu are extracted
from the CatFlow dataset itself — `dataset_per_adsorbate/<symbol>.lmdb` and the
`val_id` factorized LMDB — so the conditioning is fully in-distribution.

For every request the app:

1. runs 50-step flow-matching sampling for **10 samples** (with full
   trajectory),
2. assembles each sample / trajectory frame into an ASE structure,
3. keeps only the **structurally valid** final samples and, per sample, only
   the **valid intermediate trajectory frames**,
4. shows each valid sample in two viewers — final structure and generation
   trajectory (frame slider + play),
5. on **E eval**, relaxes the structure with UMA (`uma-s-1p1`), reports the
   chemical formula (slab + adsorbate) and the adsorption energy
   `ΔE_ads = E_sys − E_slab − E_adsorbate`,
6. draws an illustrative activity **volcano** for selected descriptors
   (`*OH, *O, *OOH, *H, *CO, *N`).

## Layout

```
webapp/
├── server.py          FastAPI backend (endpoints + static serving)
├── catflow_engine.py  model loading, batch building, sampling, assembly
├── energy.py          UMA relaxation + adsorption energy + volcano
├── prepare_data.py    builds the adsorbate / composition menus
├── run.sh             entry point
├── data/              adsorbates.json, compositions.json (generated)
├── ckpts/             dng/ and sp/ checkpoints (downloaded)
├── _newdata/          CatFlow dataset (data.tar.gz, used by prepare_data.py)
└── static/            index.html, app.js, style.css  (mobile-first UI)
```

The interface uses a dark "scientific-instrument" theme — framed molecular
specimens, a catalytic-teal accent and an amber energy accent — and is built
for phones.

## Run

```bash
cd CatFlow/webapp
./run.sh                 # serves on 0.0.0.0:8000
```

Environment variables: `PORT` (default 8000), `CATFLOW_DEVICE`
(default `cuda:0`).

Open `http://<host>:8000`. The UI is designed for phones; on a remote
machine use an SSH tunnel (`ssh -L 8000:localhost:8000 <host>`).

## Notes

* Models load once at startup (~20 s): the de-novo checkpoint, the
  structure-prediction checkpoint, and UMA. Generation is ~5 s and an energy
  evaluation ~5–30 s on an H200.
* Samples are produced as a batch of 10 independent rows
  (`multiplicity = 1`); this avoids a latent `multiplicity > 1` shape bug in
  the structure-prediction encoder.
* Runs in the `adsorbgen` micromamba env
  (`/home/irteam/micromamba/envs/adsorbgen`).
