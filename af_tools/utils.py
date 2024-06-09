from functools import singledispatch
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


def load_structure(path: str) -> Structure:
    if path.split(".")[-1] == "cif":
        parser = MMCIFParser()
    elif path.split(".")[-1] == "pdb":
        parser = PDBParser()
    else:
        raise Exception(f"Unknwon model file types:{path}")
    return parser.get_structure(path, path)


@singledispatch
def calculate_rmsd(target_model_path: Any,
                   ref_model_path: str) -> float | NDArray:
    raise NotImplementedError(
        (f"Argument type {type(target_model_path)} for target_model_path is"
         "not implemented for calculate_rmsd function."))


@calculate_rmsd.register
def _(target_model_path: str, ref_model_path: str) -> float | NDArray:
    ref_model = load_structure(ref_model_path)
    target_model = load_structure(target_model_path)

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
def _(target_model_path: list[str], ref_model_path: str) -> float | NDArray:
    rmsds = np.full(len(target_model_path), np.nan, dtype=float)
    for i, target_model_p in enumerate(target_model_path):
        rmsds[i] = calculate_rmsd(target_model_p, ref_model_path)
    return rmsds


def worker_calculate_rmsd2(ref_model: Structure | str, target_model_path: str,
                           index) -> tuple:

    target_model_structure = load_structure(target_model_path)

    if isinstance(ref_model, str):
        ref_model_structure = load_structure(ref_model)
    else:
        ref_model_structure = ref_model

    if ref_model_structure == target_model_structure:
        return (index, 0)

    m1_cas: list[Atom] = []
    m2_cas: list[Atom] = []
    for m1_res, m2_res in zip(ref_model_structure.get_residues(),
                              target_model_structure.get_residues()):
        assert m1_res.resname == m2_res.resname
        assert m1_res.id == m2_res.id

        m1_cas.append(m1_res["CA"])
        m2_cas.append(m2_res["CA"])

    sup = Superimposer()
    sup.set_atoms(m1_cas, m2_cas)

    assert isinstance(sup.rms, float)

    return (index, sup.rms)


def worker_wrapper_calculate_rmsd(args: tuple) -> tuple:
    ref_model_structure, target_model_path, index = args
    return worker_calculate_rmsd(ref_model=ref_model_structure,
                                 target_model_path=target_model_path,
                                 index=index)


def worker_af2output_get_pred(path: Path) -> Sequence[AF2Prediction]:
    from af_tools.data_types.af2output import AF2Output
    af2output = AF2Output(path=path, process_number=1)
    # print(len(af2output.predictions))
    return af2output.predictions


def worker_af3output_get_pred(path: Path) -> Sequence[AF3Prediction]:
    from af_tools.data_types.af3output import AF3Output
    af3output = AF3Output(path=path, process_number=1)
    return af3output.predictions


def worker_pred_ref_rmsd(pred: AFPrediction,
                         index: int) -> tuple[NDArray, int]:
    rmsds = np.full(len(pred.models), np.nan, dtype=float)
    ref_model = load_structure(str(pred.reference_path))

    for i, model in enumerate(pred.models):
        if isinstance(model, AF2Model):
            target_model_path = model.relaxed_pdb_path
        else:
            target_model_path = model.model_path

        rmsds[i] = worker_calculate_rmsd(
            ref_model=ref_model,
            target_model_path=str(target_model_path),
            index=i)

    return (rmsds, index)


def worker_wrapper_pred_ref_rmsd(args) -> tuple[NDArray, int]:
    pred, index = args
    return worker_pred_ref_rmsd(pred, index)
