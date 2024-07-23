from functools import singledispatch

from typing import Any
from pathlib import Path

from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom
from Bio.PDB.Superimposer import Superimposer

from numpy.typing import NDArray
import numpy as np
import pandas as pd

from af_tools.analysis.structure_tools import load_structure


@singledispatch
def calc_rmsd(target_model: Any, ref_model: str | Structure) -> float:
    raise NotImplementedError(
        (f"Argument type {type(target_model)} for target_model_path is"
         "not implemented for calc_rmsd function."))


@calc_rmsd.register
def _(target_model: str | Structure, ref_model: str | Structure) -> float:

    if isinstance(target_model, str):
        target_model = load_structure(target_model)
    if isinstance(ref_model, str):
        target_model = load_structure(ref_model)

    assert isinstance(target_model, Structure)
    assert isinstance(ref_model, Structure)

    ref_CAs: list[Atom] = []
    target_CAs: list[Atom] = []
    for ref_res, target_res in zip(ref_model.get_residues(),
                                   target_model.get_residues()):
        assert ref_res.resname == target_res.resname
        assert ref_res.id == target_res.id

        ref_CAs.append(ref_res["CA"])
        target_CAs.append(target_res["CA"])

    sup = Superimposer()
    sup.set_atoms(ref_CAs, target_CAs)

    assert isinstance(sup.rms, float)
    return sup.rms


@calc_rmsd.register
def _(target_model: pd.Series, ref_model: Path | Structure) -> float:
    return calc_rmsd(target_model["best_model_path"], ref_model)


def calc_pairwise_rmsds(models: list) -> NDArray:
    model_sturcts: list[Structure] = []
    for m in models:
        if isinstance(m, str):
            model_sturcts.append(load_structure(m))
        elif isinstance(m, Structure):
            model_sturcts.append(m)
        else:
            raise ValueError((f"The type of the model {m} is {type(m)}. Only",
                              "types str and Bio.PDB.Structure is supported."))

    len_models = len(models)
    rmsds = np.full((len_models, len_models), np.nan, dtype=float)

    for i in range(len_models):
        for j in range(len_models):
            if i < j:
                continue
            m1 = model_sturcts[i]
            m2 = model_sturcts[j]
            rmsds[i][j] = calc_rmsd(m1, m2)

    return rmsds
