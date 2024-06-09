from functools import singledispatch
import subprocess
from typing import Any
from pathlib import Path
from typing import Sequence

import numpy as np

from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Superimposer import Superimposer
from numpy.typing import NDArray

from af_tools.output_types import AF2Model, AF2Prediction, AF3Prediction, AFPrediction


def load_structure(path: Path) -> Structure:
    if path.suffix == ".cif":
        parser = MMCIFParser()
    elif path.suffix == "pdb":
        parser = PDBParser()
    else:
        raise TypeError(f"Unknwon model file types:{path}")
    return parser.get_structure(path, path)


@singledispatch
def calculate_rmsd(target_model: Any,
                   ref_model: Path | Structure) -> float | NDArray:
    raise NotImplementedError(
        (f"Argument type {type(target_model)} for target_model_path is"
         "not implemented for calculate_rmsd function."))


@calculate_rmsd.register
def _(target_model: Path | Structure,
      ref_model: Path | Structure) -> float | NDArray:

    if isinstance(target_model, Path):
        target_model = load_structure(target_model)
    if isinstance(ref_model, Path):
        ref_model = load_structure(ref_model)

    ref_CAs: list[Atom] = []
    target_CAs: list[Atom] = []
    for ref_res, target_res in zip(ref_model.get_residues(),
                                   target_model.get_residues()):
        assert ref_res.resname == target_res.resname
        assert ref_res.id == target_res.id

        ref_CAs.append(ref_res["CA"])
        target_CAs.append(ref_res["CA"])

    sup = Superimposer()
    sup.set_atoms(ref_CAs, target_CAs)

    assert isinstance(sup.rms, float)
    return sup.rms


@calculate_rmsd.register
def _(target_model: list[Path] | list[Structure],
      ref_model: Path | Structure) -> float | NDArray:
    rmsds = np.full(len(target_model), np.nan, dtype=float)
    for i, target_model_p in enumerate(target_model):
        rmsds[i] = calculate_rmsd(target_model_p, ref_model)
    return rmsds


@singledispatch
def calculate_tm(target_model: Any, ref_model: Path) -> float | NDArray:
    raise NotImplementedError(
        (f"Argument type {type(target_model)} for target_model_path is"
         "not implemented for calculate_rmsd function."))


@calculate_tm.register
def _(target_model: Path, ref_model: Path) -> float | NDArray:
    cmd = [
        "USalign", "-outfmt", "2",
        str(target_model.absolute()),
        str(ref_model.absolute())
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return float(p.stdout.split("\n")[1].split()[3])


@calculate_tm.register
def _(target_model: list[Path], ref_model: Path) -> float | NDArray:
    tms = np.full(len(target_model), np.nan, dtype=float)
    for i, target_model_p in enumerate(target_model):
        tms[i] = calculate_tm(target_model_p, ref_model)
    return tms


def worker_af2output_get_pred(path: Path) -> Sequence[AF2Prediction]:
    from af_tools.data_types.af2output import AF2Output
    af2output = AF2Output(path=path, process_number=1)
    # print(len(af2output.predictions))
    return af2output.predictions


def worker_af3output_get_pred(path: Path) -> Sequence[AF3Prediction]:
    from af_tools.data_types.af3output import AF3Output
    af3output = AF3Output(path=path, process_number=1)
    return af3output.predictions
