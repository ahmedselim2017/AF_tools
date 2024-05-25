from pathlib import Path
from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Superimposer import Superimposer


def load_structure(path: Path) -> Structure:
    if path.suffix == ".cif":
        parser = MMCIFParser()
    elif path.suffix == ".pdb":
        parser = PDBParser()
    else:
        raise Exception(f"Unknwon model file types:{str(path)}")
    return parser.get_structure(path.name, path)


def calculate_rmsd(ref_model_structure: Structure,
                   target_model_path: Path,
                   index: int = 0) -> tuple[int, float]:

    target_model_structure = load_structure(target_model_path)

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


def calculate_rmsd_wrapper(args: tuple) -> tuple[int, float]:
    ref_model_structure, target_model_path, index = args
    return calculate_rmsd(ref_model_structure=ref_model_structure,
                          target_model_path=target_model_path,
                          index=index)
