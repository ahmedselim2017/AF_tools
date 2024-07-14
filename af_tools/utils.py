from functools import singledispatch
import subprocess
from typing import Any
from pathlib import Path

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


def load_structure(path: Path) -> Structure:
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
def _(target_model: Path | Structure,
      ref_model: Path | Structure) -> float | NDArray:

    if isinstance(ref_model, Path):
        ref_model = load_structure(ref_model)
    if isinstance(target_model, Path):
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
         "not implemented for calculate_rmsd function."))


@calculate_tm.register
def _(target_model: Path, ref_model: Path) -> float | NDArray:
    cmd = [
        "USalign", "-outfmt", "2", "-mm", "1", "-ter", "0",
        str(target_model.absolute()),
        str(ref_model.absolute())
    ]
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
                         chainA_searcher: NeighborSearch,
                         chainB_searcher: NeighborSearch,
                         plddts: list[float],
                         dist_cutoff: float = 4.0):
    A_int: set[Residue] = set()
    B_int: set[Residue] = set()
    int_plddts: list[float] = []
    for a_atom in chainA.get_atoms():
        for int_res in chainB_searcher.search(a_atom.get_coord(),
                                              dist_cutoff,
                                              level="R"):
            assert isinstance(int_res, Residue)
            B_int.add(int_res.id[1])
            for int_atom in int_res.get_atoms():
                int_plddts.append(plddts[int_atom.serial_number - 1])

    for b_atom in chainB.get_atoms():
        for int_res in chainA_searcher.search(b_atom.get_coord(),
                                              dist_cutoff,
                                              level="R"):
            assert isinstance(int_res, Residue)
            A_int.add(int_res.id[1])

            for int_atom in int_res.get_atoms():
                int_plddts.append(plddts[int_atom.serial_number - 1])

    if len(int_plddts) == 0:
        return 30, None, None
    return np.asarray(int_plddts).mean(), A_int, B_int


def find_interface(df: pd.Series, dist_cutoff: float = 4.0) -> tuple:

    model = load_structure(df["best_model_path"])  # type:ignore

    chains = list(model.get_chains())

    chain_ints = np.zeros((len(chains), len(chains)), dtype=object)
    # plddts = np.zeros((len(chains), len(chains)), dtype=float)
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
                    df["plddt"],  # type:ignore
                    dist_cutoff)
            mean_plddts.append(mean_plddt)

    return np.asarray(mean_plddts).mean(), chain_ints
