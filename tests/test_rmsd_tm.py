import pytest

from af_tools.afparser import AFParser
from af_tools.afplotter import AFPlotter
import af_tools.utils as utils


def test_calc_rmsd() -> None:
    afoutput = AFParser("./tests/data/colabfold_rmsd").get_output()

    rmsd = utils.calculate_rmsd(afoutput.predictions[0].models[0].model_path,
                                afoutput.predictions[1].models[0].model_path)
    assert rmsd == 1.1397067864395125


def test_calc_pairwise_rmsds() -> None:
    afoutput = AFParser("./tests/data/colabfold_rmsd",
                        process_number=12).get_output()
    plotter = AFPlotter()

    rmsds = afoutput.calculate_pairwise_rmsds(0)

    fig = plotter.plot_upper_trig(rmsds)
    fig.show()
    input()


def test_calc_tm() -> None:
    afoutput = AFParser("./tests/data/colabfold_rmsd").get_output()

    tm = utils.calculate_tm(afoutput.predictions[0].models[0].model_path,
                            afoutput.predictions[1].models[0].model_path)
    assert tm == 0.963


def test_calc_pairwise_tms() -> None:
    afoutput = AFParser("./tests/data/colabfold_rmsd",
                        process_number=12).get_output()

    plotter = AFPlotter()

    tms = afoutput.calculate_pairwise_tms(0)

    fig = plotter.plot_upper_trig(tms)
    fig.show()
    input()
