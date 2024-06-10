import pytest
import numpy as np

from af_tools.afparser import AFParser


def test_colabfold_pred_plddt() -> None:
    colabfold_output = AFParser("./tests/data/colabfold").get_output()
    fig = colabfold_output.plot_all_plddts()[0]

    # fig.show()
    # input()


def test_colabfold_multimer_pred_plddt() -> None:
    colabfold_output = AFParser("./tests/data/colabfold_multimer").get_output()
    fig = colabfold_output.plot_all_plddts()[0]

    # fig.show()
    # input()


def test_af3_pred_plddt() -> None:
    af3output = AFParser("./tests/data/af3").get_output()
    fig = af3output.plot_all_plddts()[0]

    # fig.show()
    # input()


def test_af3_multimer_pred_plddt() -> None:
    af3output = AFParser("./tests/data/af3_multimer").get_output()
    fig = af3output.plot_all_plddts()[0]

    # fig.show()
    # input()


def test_af3_pred_pae() -> None:
    af3output = AFParser("./tests/data/af3").get_output()
    fig = af3output.plot_all_plddts()[0]

    # fig.show()
    # input()


def test_af3_multimer_pred_pae() -> None:
    af3output = AFParser("./tests/data/af3_multimer").get_output()
    fig = af3output.plot_all_paes()[0]

    # fig.show()
    # input()


def test_colabfold_multimer_pred_pae() -> None:
    af2output = AFParser("./tests/data/colabfold_multimer").get_output()
    fig = af2output.plot_all_paes()[0]

    # fig.show()
    # input()


def test_colabfold_hist() -> None:
    af2output = AFParser("./tests/data/colabfold_recursive",
                         process_number=12).get_output()
    fig = af2output.plot_plddt_hist()

    # fig.show()
    # input()


def test_af3_hist() -> None:
    af3output = AFParser("./tests/data/af3_recursive",
                         process_number=12).get_output()
    fig = af3output.plot_plddt_hist()

    # fig.show()
    # input()


def test_colabfcolabfold_rmsd() -> None:
    colabfold_output = AFParser("./tests/data/colabfold_rmsd",
                                process_number=12).get_output()
    rmsds, plddts = colabfold_output.calculate_rmsds_plddts(rank_indeces=[0])
    hbscan = colabfold_output.get_rmsd_plddt_hbscan(rmsds=rmsds, plddts=plddts)
    fig = colabfold_output.plot_rmsd_plddt(rmsds=rmsds,
                                           plddts=plddts,
                                           hbscan=hbscan)

    clust_paths, clust_plddts = colabfold_output.get_rmsd_plddt_cluster_paths(
        rank_indeces=[0], hbscan=hbscan)

    # print(clust_paths[np.argmax(clust_plddts)])
    # fig.show()
    # input()
