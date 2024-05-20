from af_tools.analyze import AFOutput
import pytest


def test_invalid_path() -> None:
    with pytest.raises(Exception) as e_info:
        AFOutput("./tests/data/wrongpath")


def test_af2_pred_detection() -> None:
    af2output = AFOutput("./tests/data/af2")
    assert af2output.predictions[0].num_ranks == 5
    assert af2output.predictions[0].af_version == 2
    assert af2output.predictions[0].models_relaxed != None
    assert af2output.predictions[0].models_unrelaxed != None


def test_af3_pred_detection() -> None:
    af3output = AFOutput("./tests/data/af3")
    assert af3output.predictions[0].num_ranks == 5
    assert af3output.predictions[0].af_version == 3
    assert af3output.predictions[0].models_relaxed == None
    assert af3output.predictions[0].models_unrelaxed != None


def test_af2_pred_plddt() -> None:
    af2output = AFOutput("./tests/data/af2")
    fig = af2output.plot_plddt_graphs(is_relaxed=True)[0]

    fig.show()
    # input()


def test_af3_pred_plddt() -> None:
    af3output = AFOutput("./tests/data/af3")
    fig = af3output.plot_plddt_graphs(is_relaxed=False)[0]

    fig.show()
    # input()
