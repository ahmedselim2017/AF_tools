import pytest

from af_tools.afparser import AFParser
from af_tools.afplotter import AFPlotter


def test_colabfold_recursive() -> None:
    colabfold_output = AFParser(
        "./tests/data/colabfold_recursive").get_output()

    assert len(colabfold_output.predictions) == 93


def test_colabfold_recursive_multiprocessing() -> None:
    colabfold_output = AFParser("./tests/data/colabfold_recursive",
                                process_number=12).get_output()

    assert len(colabfold_output.predictions) == 93


def test_af3_recursive() -> None:
    af3_output = AFParser("./tests/data/af3_recursive").get_output()

    assert len(af3_output.predictions) == 10


def test_af3_recursive_multiprocessing() -> None:
    af3_output = AFParser("./tests/data/af3_recursive",
                          process_number=12).get_output()

    assert len(af3_output.predictions) == 10


def test_colabfold_rmsd() -> None:
    output = AFParser("./tests/data/colabfold_rmsd",
                      process_number=12).get_output()
    plotter = AFPlotter()

    rmsds = output.calculate_rmsds(rank_index=0)
    fig = plotter.plot_rmsds(rmsds, [])

    fig.show()
    input()
