from functools import singledispatch
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx

from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Residue import Residue

from af_tools.analysis.structure_tools import load_structure


def find_chain_interface(chainA: Chain,
                         chainB: Chain,
                         chainA_searcher: NeighborSearch | None = None,
                         chainB_searcher: NeighborSearch | None = None,
                         dist_cutoff: float = 4.5,
                         plddts: list[float] | None = None):

    if chainA_searcher is None:
        chainA_searcher = NeighborSearch(list(chainA.get_atoms()))
    if chainB_searcher is None:
        chainB_searcher = NeighborSearch(list(chainB.get_atoms()))

    A_int: set[int] = set()
    A_plddts: list[float] = []
    B_int: set[int] = set()
    B_plddts: list[float] = []

    for a_atom in chainA.get_atoms():
        for int_res in chainB_searcher.search(a_atom.get_coord(),
                                              dist_cutoff,
                                              level="R"):
            assert isinstance(int_res, Residue)
            B_int.add(int_res.id[1])
            if plddts is not None:
                for int_atom in int_res.get_atoms():
                    B_plddts.append(plddts[int_atom.serial_number - 1])

    for b_atom in chainB.get_atoms():
        for int_res in chainA_searcher.search(b_atom.get_coord(),
                                              dist_cutoff,
                                              level="R"):
            assert isinstance(int_res, Residue)
            A_int.add(int_res.id[1])

            if plddts is not None:
                for int_atom in int_res.get_atoms():
                    A_plddts.append(plddts[int_atom.serial_number - 1])

    if (len(A_int) + len(B_int)) == 0:
        return None, None, None, None
    return A_plddts, A_int, B_plddts, B_int


@singledispatch
def find_interface(data: Any,
                   dist_cutoff: float = 4.5,
                   plddts=None) -> tuple[float, nx.DiGraph]:
    raise NotImplementedError((f"Argument type {type(data)} for data is"
                               "not implemented for find_interface function."))


@find_interface.register
def _(data: Structure,
      dist_cutoff: float = 4.5,
      plddts: list | None = None) -> tuple[float, nx.DiGraph]:

    chains = list(data.get_chains())
    len_chains = len(chains)

    int_graph = nx.DiGraph()
    int_graph.add_nodes_from(range(len_chains))

    searchers = np.zeros(len(chains), dtype=object)

    for i, chain in enumerate(chains):
        searchers[i] = NeighborSearch(list(chain.get_atoms()))

    for i, chainA in enumerate(chains):
        for j, chainB in enumerate(chains):
            if i <= j:
                continue
            A_plddts, A_int, B_plddts, B_int = find_chain_interface(
                chainA=chainA,
                chainB=chainB,
                chainA_searcher=searchers[i],
                chainB_searcher=searchers[j],
                plddts=plddts,
                dist_cutoff=dist_cutoff)

            int_graph.add_edge(i, j, plddts=A_plddts, res=A_int)
            int_graph.add_edge(j, i, plddts=B_plddts, res=B_int)

    if int_graph.number_of_edges == 0:
        mean_plddt = 30.0
    else:
        mean_plddt = np.mean(
            np.concatenate(
                list(nx.get_edge_attributes(int_graph, "plddts").values())))
    assert isinstance(mean_plddt, float)
    return mean_plddt, int_graph


@find_interface.register
def _(data: str,
      dist_cutoff: float = 4.5,
      plddts: list | None = None) -> tuple[float, nx.DiGraph]:
    return find_interface(load_structure(data), dist_cutoff, plddts)


@find_interface.register
def _(data: pd.Series, dist_cutoff: float = 4.5) -> tuple[float, nx.DiGraph]:
    return find_interface(
        load_structure(data["best_model_path"]),  # type:ignore
        dist_cutoff,
        data["plddt"])
