from pathlib import Path
from typing import Sequence
from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Superimposer import Superimposer

from af_tools.output_types import AF2Prediction, AF3Prediction


def load_structure(path: str) -> Structure:
    if path.split(".")[-1] == "cif":
        parser = MMCIFParser()
    elif path.split(".")[-1] == "pdb":
        parser = PDBParser()
    else:
        raise Exception(f"Unknwon model file types:{path}")
    return parser.get_structure(path, path)


def worker_calculate_rmsd(ref_model: Structure | str, target_model_path: str,
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
