from ase.visualize import view
from ase.atoms import Atoms
from ase.optimize import LBFGS
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from ase.build import molecule

def get_uma_calculator(model_name: str = "uma-s-1p1", device: str = "cuda") -> FAIRChemCalculator:
    """Get UMA calculator."""
    predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
    return FAIRChemCalculator(predictor, task_name="oc20")

calc = get_uma_calculator(device="cuda")

target_single_atom_list = ["C", "H", "N", "O"]

for target_single_atom in target_single_atom_list:
  
  if target_single_atom == "C":
    required_molecules = ["CO", "H2O", "H2"]
    coeffs = [1, -1, 1]
  elif target_single_atom == "H":
    required_molecules = ["H2"]
    coeffs   = [0.5]
  elif target_single_atom == "N":
    required_molecules = ["N2"]
    coeffs = [0.5]
  elif target_single_atom == "O":
    required_molecules = ["H2", "H2O"]
    coeffs = [-1, 1]

  formation_energy = 0.0

  for molecule_name, coeff in zip(required_molecules, coeffs):
    ref_molecule_obj = molecule(molecule_name)
    ref_molecule_obj.center(vacuum=15)
    ref_molecule_obj.calc = calc
    opt = LBFGS(ref_molecule_obj, logfile=None)
    opt.run(fmax=0.05)
    formation_energy += coeff * ref_molecule_obj.get_potential_energy()

  E_single_atom = formation_energy

  print("Single atom target:", target_single_atom)
  print("Single atom energy (UMA):", E_single_atom)
  print("=" * 80)