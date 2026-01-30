# CatFlow: Co-generation of Slab-Adsorbate Systems via Flow Matching

A flow matching framework for de novo generation and structure prediction of heterogeneous catalysts. CatFlow jointly generates slab structures and adsorbate coordinates within a unified objective, directly capturing surface-adsorbate interactions.

<p align="center">
  <img src="assets/concept.png" width="37%">
  <img src="assets/cogen_traj.png" width="53%">
</p>

## Key Features

- **Co-generation of slab-adsorbate systems**: First framework to jointly generate slab structures and adsorbate coordinates via flow matching.
- **Factorized representation**: Decomposes slab-adsorbate systems into primitive cells, transformation matrices, vacuum scaling factors, and adsorbates, reducing learnable variables by 9.2× on average.
- **Two task modes**: Supports both de novo generation (with discrete flow matching for atomic species) and structure prediction (fixed composition).
- **Adsorbate conditioning**: Generates catalyst structures conditioned on target adsorbate species.

## Installation

### Environment Setup

```bash
conda env create -f environment.yml
conda activate catflow
```

### Dependencies

The main dependencies include:
- PyTorch >= 2.8
- PyTorch Lightning
- PyMatGen
- ASE
- fairchem-core

## Data

CatFlow uses the Open Catalyst 2020 (OC20) IS2RES dataset. The data processing pipeline transforms raw OC20 structures into a factorized representation.

### Data Processing

The factorized representation consists of four components:

1. **Primitive cell** $(S_{\text{prim}})$: The repeating unit containing atomic species, coordinates, and lattice matrix
2. **Transformation matrix** $(M \in \mathbb{Z}^{3 \times 3})$: Specifies how to construct the slab from the primitive cell
3. **Vacuum scaling factor** $(k_{\text{vac}})$: Determines the vacuum region height
4. **Adsorbate**: Atomic species (condition) and Cartesian coordinates (learnable)

<!-- ## Model Architecture

CatFlow uses a transformer-based architecture with three main components:

| Component | Depth | Heads | Hidden Dim |
|-----------|-------|-------|------------|
| Atom Encoder | 8 | 12 | 768 |
| Token Transformer | 24 | 12 | 768 |
| Atom Decoder | 8 | 12 | 768 |

Total parameters: ~430M -->

## Training

### De Novo Generation

Train the model with discrete flow matching for atomic species:

```bash
bash bash_scripts/train_gen.sh
```

Key configurations:
- `model.flow_model_args.dng=true`: Enable discrete flow matching for de novo generation
- `model.training_args.flow_loss_type=x1_loss`: Use x1 prediction loss
- `model.training_args.loss_type=l1`: L1 loss for geometric variables

### Structure Prediction

Train the model with fixed atomic species:

```bash
bash bash_scripts/train_pred.sh
```

Key configurations:
- `model.flow_model_args.dng=false`: Disable discrete flow matching (species are given)

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `train.pl_trainer.devices` | Number of GPUs | 8 |
| `model.training_args.lr` | Learning rate | 1e-4 |
| `model.training_args.warmup_steps` | Warmup steps | 5000 |
| `data.batch_size.train` | Batch size per device | 128 |

## Sampling

### Generate Samples

```bash
bash bash_scripts/sample_all_subsets.sh
```

This script runs parallel generation across multiple GPUs. Key parameters:

```bash
CHECKPOINT_PATH="path/to/checkpoint.ckpt"
DATA_ROOT="path/to/dataset"
BASE_OUTPUT_DIR="path/to/outputs"
```

### Sampling Script

For single dataset sampling:

```bash
python scripts/sampling/save_samples.py \
    --checkpoint $CHECKPOINT_PATH \
    --val_lmdb_path $LMDB_PATH \
    --output_dir $OUTPUT_DIR \
    --num_samples 1 \
    --sampling_steps 50 \
    --batch_size 128
```

## Evaluation

### Adsorption Energy Evaluation

Evaluate the adsorption energy of generated structures:

```bash
bash bash_scripts/eval_E_ads.sh
```

This script:
1. Relaxes generated structures using pretrained GNN potentials (UMA)
2. Computes adsorption energies: $\Delta E_{\text{ads}} = E_{\text{sys}} - E_{\text{slab}} - E_{\text{ads}}$
3. Aggregates results across all adsorbate subsets

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Validity | Interatomic distances > 0.5 Å, cell volume > 0.1 Å³ |
| Uniqueness | Non-duplicate slab structures (via StructureMatcher) |
| Compositional diversity | Mean pairwise distance of compositional fingerprints |
| Match rate | Structural match with ground truth (structure prediction) |
| RMSD | Root mean square deviation from ground truth |
| $\Delta E_{\text{ads}}$ success rate | $\|\Delta E_{\text{ads}}^{\text{ref}} - \Delta E_{\text{ads}}^{\text{gen}}\| \leq 0.1$ eV |

## Project Structure

```
CatFlow/
├── configs/                    # Hydra configuration files
├── bash_scripts/              # Training and evaluation scripts
├── scripts/
│   ├── sampling/              # Sampling utilities
│   └── relax_energy/          # Energy evaluation scripts
├── src/
│   ├── data/                  # Data loading and processing
│   ├── models/
│   │   ├── flow.py           # Flow matching model
│   │   ├── layers.py         # Encoder, decoder, transformer layers
│   │   └── transformers.py   # DiT blocks and embedders
│   └── module/
│       └── effcat_module.py  # PyTorch Lightning module
├── environment.yml
└── primitive_atom_distribution.json
```

<!-- @inproceedings{catflow2026,
  title={CatFlow: Co-generation of Slab-Adsorbate Systems via Flow Matching},
  author={Anonymous},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
``` -->