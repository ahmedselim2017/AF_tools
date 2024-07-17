from functools import singledispatch
import subprocess
from typing import Any
from pathlib import Path
import math

import numpy as np
from numpy.typing import NDArray

import pandas as pd

from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Superimposer import Superimposer
from Bio.PDB.NeighborSearch import NeighborSearch


def load_structure(path: Path | str) -> Structure:
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == ".cif":
        parser = MMCIFParser()
    elif path.suffix == ".pdb":
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
def _(target_model: Path | str | Structure,
      ref_model: Path | str | Structure) -> float | NDArray:

    if isinstance(ref_model, Path) or isinstance(ref_model, str):
        ref_model = load_structure(ref_model)
    if isinstance(target_model, Path) or isinstance(target_model, str):
        target_model = load_structure(target_model)

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


@calculate_rmsd.register
def _(target_model: list, ref_model: Path | Structure) -> float | NDArray:
    rmsds = np.full(len(target_model), np.nan, dtype=float)
    for i, target_model_p in enumerate(target_model):
        rmsds[i] = calculate_rmsd(target_model_p, ref_model)
    return rmsds


@calculate_rmsd.register
def _(target_model: pd.Series, ref_model: Path | Structure) -> float | NDArray:
    return calculate_rmsd(target_model["best_model_path"], ref_model)


@singledispatch
def calculate_tm(target_model: Any, ref_model: Path) -> float | NDArray:
    raise NotImplementedError(
        (f"Argument type {type(target_model)} for target_model_path is"
         "not implemented for calculate_tm function."))


@calculate_tm.register
def _(target_model: Path | str, ref_model: Path) -> float | NDArray:
    cmd = [
        "USalign", "-outfmt", "2", "-mm", "1", "-ter", "0",
        f"{str(target_model)}", f"{str(ref_model)}"
    ]
    print(' '.join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    return float(p.stdout.split("\n")[1].split()[3])


@calculate_tm.register
def _(target_model: list, ref_model: Path) -> float | NDArray:
    tms = np.full(len(target_model), np.nan, dtype=float)
    for i, target_model_p in enumerate(target_model):
        tms[i] = calculate_tm(target_model_p, ref_model)
    return tms


@calculate_tm.register
def _(target_model: pd.Series, ref_model: Path) -> float | NDArray:
    return calculate_tm(target_model["best_model_path"], ref_model)


def find_chain_interface(chainA: Chain,
                         chainB: Chain,
                         chainA_searcher: NeighborSearch | None = None,
                         chainB_searcher: NeighborSearch | None = None,
                         plddts: list[float] | None = None,
                         dist_cutoff: float = 4.5):

    if chainA_searcher is None:
        chainA_searcher = NeighborSearch(list(chainA.get_atoms()))
    if chainB_searcher is None:
        chainB_searcher = NeighborSearch(list(chainB.get_atoms()))

    A_int: set[Residue] = set()
    B_int: set[Residue] = set()

    int_plddts: list[float] = []
    for a_atom in chainA.get_atoms():
        for int_res in chainB_searcher.search(a_atom.get_coord(),
                                              dist_cutoff,
                                              level="R"):
            assert isinstance(int_res, Residue)
            B_int.add(int_res.id[1])
            if plddts is not None:
                for int_atom in int_res.get_atoms():
                    int_plddts.append(plddts[int_atom.serial_number - 1])

    for b_atom in chainB.get_atoms():
        for int_res in chainA_searcher.search(b_atom.get_coord(),
                                              dist_cutoff,
                                              level="R"):
            assert isinstance(int_res, Residue)
            A_int.add(int_res.id[1])

            if plddts is not None:
                for int_atom in int_res.get_atoms():
                    int_plddts.append(plddts[int_atom.serial_number - 1])

    if len(int_plddts) == 0:
        return 30, None, None
    return np.asarray(int_plddts).mean(), A_int, B_int


@singledispatch
def find_interface(data: Any, dist_cutoff: float = 4.5) -> tuple:
    raise NotImplementedError((f"Argument type {type(data)} for data is"
                               "not implemented for find_interface function."))


@find_interface.register
def _(data: list, dist_cutoff: float = 4.5) -> tuple:
    chain_ints = np.zeros((len(data), len(data)), dtype=object)

    searchers = np.zeros(len(data), dtype=object)

    for i, chain in enumerate(data):
        searchers[i] = NeighborSearch(list(chain.get_atoms()))

    for i, chainA in enumerate(data):
        for j, chainB in enumerate(data):
            if i <= j:
                continue
            _, chain_ints[i][j], chain_ints[j][i] = find_chain_interface(
                chainA=chainA,
                chainB=chainB,
                chainA_searcher=searchers[i],
                chainB_searcher=searchers[j],
                plddts=None,
                dist_cutoff=dist_cutoff)

    return None, chain_ints


@find_interface.register
def _(data: pd.Series, dist_cutoff: float = 4.5) -> tuple:

    model = load_structure(data["best_model_path"])  # type:ignore

    chains = list(model.get_chains())

    chain_ints = np.zeros((len(chains), len(chains)), dtype=object)
    mean_plddts: list[float] = []

    searchers = np.zeros(len(chains), dtype=object)

    for i, chain in enumerate(chains):
        searchers[i] = NeighborSearch(list(chain.get_atoms()))

    for i, chainA in enumerate(chains):
        for j, chainB in enumerate(chains):
            if i <= j:
                continue
            mean_plddt, chain_ints[i][j], chain_ints[j][
                i] = find_chain_interface(
                    chainA,
                    chainB,
                    searchers[i],
                    searchers[j],
                    data["plddt"],  # type:ignore
                    dist_cutoff)
            mean_plddts.append(mean_plddt)

    return np.asarray(mean_plddts).mean(), chain_ints


@singledispatch
def calculate_radiusofgyration(data: Any) -> float:
    raise NotImplementedError(
        (f"Argument type {type(data)} for data is"
         "not implemented for calculate_radiusofgyration function."))


@calculate_radiusofgyration.register
def _(data: Structure) -> float:
    cm = data.center_of_mass()

    tot_mass = 0.0
    A = 0.0
    for at in data.get_atoms():
        at_mass = at.mass
        A += at_mass * ((at.get_coord() - cm)**2).sum()
        tot_mass += at_mass
    return math.sqrt(A / tot_mass)


@calculate_radiusofgyration.register
def _(data: Path | str) -> float:
    structure = load_structure(data)
    return calculate_radiusofgyration(structure)


def _(data: pd.Series) -> float:
    structure = load_structure(data["best_model_path"])  # type: ignore
    return calculate_radiusofgyration(structure)
