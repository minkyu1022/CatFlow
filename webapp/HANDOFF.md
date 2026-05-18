# CatFlow Web App — Agent Handoff

This document lets an agent on a **fresh GPU server** pick up the CatFlow web
app in one read. Everything under `webapp/` is committed to the repo; the large
binaries (checkpoints, dataset) are **not** and must be re-downloaded.

---

## 1. What this is

A mobile-first beta web service for the **CatFlow** flow-matching catalyst
model. Two tasks:

- **De novo generation** — user picks an adsorbate; the model generates the
  whole slab + adsorbate system.
- **Structure prediction** — user picks an adsorbate **and** a slab
  composition (from a dataset menu *or* a free-typed formula like `Pt3Ni`);
  the model predicts the structure for that fixed composition.

For each request: 50-step flow-matching sampling → 10 samples (+ trajectory) →
assemble → keep only structurally valid samples and valid trajectory frames →
show two NGL viewers per sample (final structure + trajectory). An **E eval**
button relaxes a structure with the UMA potential (`uma-s-1p1`) and reports the
chemical formula and adsorption energy `ΔE_ads = E_sys − E_slab − E_adsorbate`,
plus an activity volcano for selected adsorbates.

**Status: fully working and tested** (de novo, structure prediction, custom
composition, energy eval, volcano) on the original server. The headless
end-to-end test `test_ui.py` passes with no console errors.

---

## 2. Repo layout (`webapp/`)

Committed (in git):

```
webapp/
├── server.py          FastAPI backend — endpoints + static serving
├── catflow_engine.py  model loading, batch building, 50-step sampling, assembly
├── energy.py          UMA relaxation + adsorption energy + volcano data
├── prepare_data.py    builds the adsorbate / composition menus from the dataset
├── run.sh             entry point
├── test_ui.py         headless-browser end-to-end test (Playwright)
├── README.md          user-facing readme
├── HANDOFF.md         this file
├── data/
│   ├── adsorbates.json    67-adsorbate menu  (generated; committed so the app
│   └── compositions.json  300-composition menu  runs without the dataset)
└── static/            index.html, app.js, style.css  (mobile-first dark UI)
```

NOT in git (re-download — see §4):

```
webapp/ckpts/      de-novo + structure-prediction checkpoints  (~7 GB)
webapp/_newdata/   CatFlow dataset, only needed to rebuild the menus  (~2 GB)
```

Only files under `webapp/` were added to the CatFlow repo — no existing
CatFlow source was modified.

---

## 3. Environment

Needs a Python env (3.10 / 3.11) with a CUDA build of PyTorch plus:

```
pytorch-lightning (lightning)  fairchem-core      ase        pymatgen
lmdb    omegaconf    hydra-core    einops    pebble    torch_geometric
smact   numpy   tqdm   fastapi   uvicorn   pydantic   slowapi
playwright            (only for running test_ui.py)
```

`fairchem-core` provides the UMA potential (`pretrained_mlip` /
`FAIRChemCalculator`). On the original server an existing env already had
torch + lightning + fairchem + ase + pymatgen + lmdb + fastapi + uvicorn, and
only these were added:

```
pip install smact einops pebble torch_geometric playwright
```

If starting clean: `CatFlow/environment.yml` lists most deps (note: the torch
lines in it are commented out — install a CUDA torch build separately, then
`pip install` the rest). The repo `README.md` documents the original
`conda env create -f environment.yml` flow.

Headless UI testing (`test_ui.py`) additionally needs system libs for Chromium
(`playwright install chromium`, plus nss/nspr/X libs/fonts). This is optional —
skip it if you only need the server. On the original server those libs were
installed via conda-forge (`nss nspr libxcb xorg-libx* libxkbcommon libdrm
libgbm alsa-lib expat dbus atk-1.0 at-spi2-atk at-spi2-core fontconfig
font-ttf-dejavu-sans-mono fonts-conda-ecosystem`) and `LD_LIBRARY_PATH` pointed
at the env's `lib/`.

---

## 4. Bring-up on a fresh server

```bash
# 1. clone
git clone https://github.com/minkyu1022/CatFlow.git
cd CatFlow/webapp

# 2. checkpoints  (~7 GB) — extract so a *.ckpt lands anywhere under
#    ckpts/dng/ and ckpts/sp/  (the engine globs for them)
mkdir -p ckpts/dng ckpts/sp
curl -L "https://www.dropbox.com/scl/fi/io7wcq5p53d9n19rmtr70/dng_catflow_ckpts.tar?rlkey=pqjeexn60zr2bunivydexyrt1&dl=1" -o /tmp/dng.tar
curl -L "https://www.dropbox.com/scl/fi/0t71a7ntqjjpakqs2qs62/sp_catflow_ckpts.tar?rlkey=5e3c7k2olvj8pmy3btw6twxui&dl=1"  -o /tmp/sp.tar
tar -xf /tmp/dng.tar -C ckpts/dng     # -> ckpts/dng/gen_430M_final_L1_relpos/dng_model.ckpt
tar -xf /tmp/sp.tar  -C ckpts/sp      # -> ckpts/sp/pred_430M_final_L1_relpos/sp_model.ckpt

# 3. (optional) dataset — ONLY needed to rebuild data/*.json; the committed
#    JSONs already work. Download if you want to regenerate the menus.
mkdir -p _newdata
curl -L "https://www.dropbox.com/scl/fi/d057y8ip31i5821qsjdxq/data.tar.gz?rlkey=bkvmiifx1omkz9rt6hy8z8tfk&dl=1" -o /tmp/data.tar.gz
tar -xzf /tmp/data.tar.gz -C _newdata   # -> _newdata/data/dataset_per_adsorbate/, _newdata/data/dataset/val_id/
# then:  python prepare_data.py          # rebuilds data/adsorbates.json + compositions.json

# 4. run
./run.sh                # serves on 0.0.0.0:8000  (env: PORT, CATFLOW_DEVICE)
```

`run.sh` hard-codes the original server's interpreter path — edit the `PY=`
line to the new env's python, or just run `python server.py` directly with the
env active. Models + UMA load at startup (~20 s); the log prints
`[server] ready.`

Verify: `curl localhost:8000/api/adsorbates` (67 items),
`curl localhost:8000/api/compositions` (300 items), and the headless test
`python test_ui.py` (server must be running).

---

## 5. How it works

**`catflow_engine.py`** — loads both checkpoints (`EffCatModule.load_from_checkpoint`,
`torch.load` patched for PyTorch ≥2.6). `generate(mode, adsorbate, composition)`
builds an LMDB-style record, runs it through `collate_fn_with_dynamic_padding`,
samples, assembles every sample/trajectory frame with `scripts/sampling/assemble`,
and filters with `compute_structural_validity_single`. Final ASE structures are
kept in an in-memory job store (`job_id` → structures) for the eval endpoint.

**Batch scheme (important):** both modes build the batch as `N_SAMPLES (=10)`
identical rows with `multiplicity_flow_sample=1` — NOT `multiplicity=10`. The
`dng=False` (structure-prediction) encoder has a shape bug on the
`multiplicity>1` path (`joint_mask` concat mismatch in
`src/models/layers.py`). The chosen scheme produces 10 independent samples and
avoids that path. For `dng=True` the per-row primitive-cell atom count is still
histogram-sampled independently, so the 10 samples differ.

**`energy.py`** — loads UMA once (`get_predict_unit("uma-s-1p1", device="cuda")`
— must be the bare string `"cuda"`, not `"cuda:0"`). `evaluate()` fixes tag-0
bulk atoms, LBFGS-relaxes the system and the clean slab, and returns
`ΔE_ads`. `volcano_data()` returns illustrative volcano curves for
`*OH *O *OOH *H *CO *N`.

**`server.py`** — FastAPI. Endpoints: `GET /api/adsorbates`,
`GET /api/compositions`, `POST /api/generate` (`mode`, `adsorbate_id`,
`composition_id` **or** `custom_composition` free-text formula),
`POST /api/eval` (`job_id`, `sample_idx`, `adsorbate_id`). `parse_composition()`
turns a free formula (`Pt3Ni`) into atomic numbers.

**Data menus** — `prepare_data.py` builds them from the CatFlow dataset:
adsorbates (67) from `_newdata/data/dataset_per_adsorbate/<symbol>.lmdb`;
compositions (300, deduped by reduced formula) from
`_newdata/data/dataset/val_id/dataset.lmdb`.

**`static/`** — dark "scientific-instrument" themed mobile UI; NGL.js viewers
(loaded from unpkg CDN) on a dark background; trajectory frame slider + play.

---

## 6. Gotchas

- UMA device string must be `"cuda"` / `"cpu"` exactly (not `"cuda:0"`).
- `dng=False` + `multiplicity>1` is broken upstream — see the batch scheme above.
- `torch.load` is monkey-patched (`weights_only=False`) for the old-format ckpts.
- The adsorbate `id` is the symbol string (e.g. `*CO`); the composition `id`
  is the raw formula string. `_newdata/` is only a build-time input.

---

## 7. Public deployment via GitHub Pages — DONE

The app is live at **`https://minkyu1022.github.io/CatFlow/app`** (permanent
URL — safe to print on a QR code; see `webapp/catflow_demo_qr.png`).

Hybrid setup: **UI on GitHub Pages, backend on this GPU server.**

- `docs/app/` — static frontend (copy of `static/` with relative asset paths).
  Served at `…/CatFlow/app`. The backend URL is NOT baked into `app.js`; it is
  read at startup from `docs/app/api_url.txt` with a cache-buster, so a changed
  tunnel URL reaches returning visitors despite Pages' ~10 min asset cache.
- `server.py` — CORS enabled for the `https://minkyu1022.github.io` origin
  (override with `CATFLOW_CORS_ORIGINS`).
- `tunnel.sh` — starts a cloudflared quick tunnel, writes the tunnel URL into
  `docs/app/api_url.txt`, and commits + pushes it (`--no-push` to skip).
- `serve.sh` — runs the backend and `tunnel.sh` together in one terminal.
- `docs/index.html` — has a "Demo" badge linking to `./app`.

**Operating it (e.g. for a poster session):**

```bash
cd webapp
PORT=8200 ./serve.sh        # backend + tunnel in one command; Ctrl-C stops both
```

Run it inside `tmux`/`screen` so it survives disconnects. (To watch the two
processes separately, run `PORT=8200 ./run.sh` and `./tunnel.sh` in two
windows instead.) The QR/link target never changes; only the backend tunnel
URL does, and `tunnel.sh` re-syncs it on every run. Pages picks up a pushed
URL in ~1 min. Port 8000 is used by another service on this host — hence 8200.

Caveat: the Pages page is only a shell — the GPU server **and** the cloudflared
tunnel must stay up for generation/eval to work. Quick-tunnel URLs are
ephemeral (change on every cloudflared restart); a stable URL needs a named
tunnel, which requires a Cloudflare-managed domain.

---

## 8. Asset links

| Asset | Dropbox |
|-------|---------|
| De-novo checkpoint   | `https://www.dropbox.com/scl/fi/io7wcq5p53d9n19rmtr70/dng_catflow_ckpts.tar?rlkey=pqjeexn60zr2bunivydexyrt1&dl=1` |
| Structure-pred. ckpt | `https://www.dropbox.com/scl/fi/0t71a7ntqjjpakqs2qs62/sp_catflow_ckpts.tar?rlkey=5e3c7k2olvj8pmy3btw6twxui&dl=1` |
| CatFlow dataset      | `https://www.dropbox.com/scl/fi/d057y8ip31i5821qsjdxq/data.tar.gz?rlkey=bkvmiifx1omkz9rt6hy8z8tfk&dl=1` |
