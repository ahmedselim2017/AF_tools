from functools import singledispatch
from typing import Any

from Bio.PDB.Structure import Structure
import numpy as np
import pandas as pd

from Bio.PDB.Chain import Chain
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Residue import Residue

from af_tools.analysis.structure_tools import load_structure


def find_chain_interface(chainA: Chain,
                         chainB: Chain,
                         chainA_searcher: NeighborSearch | None = None,
                         chainB_searcher: NeighborSearch | None = None,
                         dist_cutoff: float = 4.5,
                         plddts: list[float] | None = None,
                         def_plddt: int = 30):

    if chainA_searcher is None:
        chainA_searcher = NeighborSearch(list(chainA.get_atoms()))
    if chainB_searcher is None:
        chainB_searcher = NeighborSearch(list(chainB.get_atoms()))

    A_int: set[int] = set()
    B_int: set[int] = set()

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
        return def_plddt, None, None
    return np.asarray(int_plddts).mean(), A_int, B_int


@singledispatch
def find_interface(data: Any, dist_cutoff: float = 4.5, plddts=None) -> tuple:
    raise NotImplementedError((f"Argument type {type(data)} for data is"
                               "not implemented for find_interface function."))


@find_interface.register
def _(data: Structure,
      dist_cutoff: float = 4.5,
      plddts: list | None = None) -> tuple:

    chains = list(data.get_chains())
    len_chains = len(chains)

    chain_ints = np.zeros((len_chains, len_chains), dtype=object)
    mean_plddts = np.full(len_chains, np.nan, dtype=float)

    searchers = np.zeros(len(chains), dtype=object)

    for i, chain in enumerate(chains):
        searchers[i] = NeighborSearch(list(chain.get_atoms()))

    for i, chainA in enumerate(chains):
        for j, chainB in enumerate(chains):
            if i <= j:
                continue
            mean_plddts[i][j], chain_ints[i][j], chain_ints[j][
                i] = find_chain_interface(chainA=chainA,
                                          chainB=chainB,
                                          chainA_searcher=searchers[i],
                                          chainB_searcher=searchers[j],
                                          plddts=plddts,
                                          dist_cutoff=dist_cutoff)

    return mean_plddts, chain_ints


@find_interface.register
def _(data: str,
      dist_cutoff: float = 4.5,
      plddts: list | None = None) -> tuple:
    return find_interface(load_structure(data), dist_cutoff, plddts)


@find_interface.register
def _(data: pd.Series, dist_cutoff: float = 4.5) -> tuple:
    return find_interface(
        load_structure(data["best_model_path"]),  # type:ignore
        dist_cutoff,
        data["plddt"])
