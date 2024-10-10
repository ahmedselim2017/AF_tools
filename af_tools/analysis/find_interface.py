from functools import singledispatch
from typing import Any

from Bio.PDB.Atom import Atom
import numpy as np
import pandas as pd
import networkx as nx

from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Residue import Residue

from af_tools.analysis.structure_tools import load_structure


@singledispatch
def find_interface(
    data: Any,
    data_type: str | None = None,
    dist_cutoff: float = 4,
) -> tuple[float, nx.Graph]:
    raise NotImplementedError((f"Argument type {type(data)} for data is"
                               "not implemented for find_interface function."))


@find_interface.register
def _(data: str,
      data_type: str | None = None,
      dist_cutoff: float = 4) -> tuple[float, nx.Graph]:
    return find_interface(load_structure(data),
                          data_type=data_type,
                          dist_cutoff=dist_cutoff)


@find_interface.register
def _(data: pd.Series,
      data_type: str | None = None,
      dist_cutoff: float = 4) -> tuple[float, nx.Graph]:
    return find_interface(
        load_structure(data["best_model_path"]),  # type:ignore
        data_type=data["output_type"],
        dist_cutoff=dist_cutoff)


@find_interface.register
def _(
    data: Structure,
    data_type: str | None = None,
    dist_cutoff: float = 4,
) -> tuple[float, nx.Graph]:

    assert data_type

    chains = list(data.get_chains())

    int_graph = nx.Graph()

    searchers = np.zeros(len(chains), dtype=object)

    for i, chain in enumerate(chains):
        searchers[i] = NeighborSearch(list(chain.get_atoms()))

    for i, chainA in enumerate(chains):
        for j, chainB in enumerate(chains):
            if i <= j:
                continue

            for a_atom in chainA.get_atoms():
                for b_atom in searchers[j].search(a_atom.get_coord(),
                                                  dist_cutoff,
                                                  level="A"):
                    assert isinstance(b_atom, Atom)

                    a_res = a_atom.get_parent()
                    b_res = b_atom.get_parent()

                    assert isinstance(a_res, Residue)
                    assert isinstance(b_res, Residue)

                    # chainid-resnum-resname
                    v1 = f"{chainA.get_id()}-{a_res.get_id()[1]}-{a_res.get_resname().title()}"
                    v2 = f"{chainB.get_id()}-{b_res.get_id()[1]}-{b_res.get_resname().title()}"

                    int_graph.add_node(v1)
                    int_graph.add_node(v2)

                    if (v1, v2) not in int_graph.edges:
                        edge_plddts = []
                        for a_res_atom in a_res.get_atoms():
                            if "AF3" in data_type:
                                edge_plddts.append(a_res_atom.get_bfactor())
                            elif a_res_atom.name == "CA":
                                edge_plddts.append(a_res_atom.get_bfactor())
                                break

                        for b_res_atom in b_res.get_atoms():
                            if "AF3" in data_type:
                                edge_plddts.append(b_res_atom.get_bfactor())
                            elif b_res_atom.name == "CA":
                                edge_plddts.append(b_res_atom.get_bfactor())
                                break

                        int_graph.add_edge(v1, v2, count=1, plddts=edge_plddts)
                    else:
                        int_graph.edges[(v1, v2)]["count"] += 1

    if int_graph.number_of_edges() == 0:
        mean_plddt = 30.0
    else:
        mean_plddt = np.mean(  # type: ignore
            sum(nx.get_edge_attributes(int_graph, "plddts").values(), []))
    assert isinstance(mean_plddt, float)
    return mean_plddt, int_graph
