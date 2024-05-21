from af_tools.analyze import AFOutput
import pytest


def test_invalid_path() -> None:
    with pytest.raises(Exception) as e_info:
        AFOutput("./tests/data/wrongpath")


def test_colabfold_pred_detection() -> None:
    colabfold_output = AFOutput("./tests/data/colabfold")
    assert colabfold_output.predictions[0].num_ranks == 5
    assert "alphafold2" in colabfold_output.predictions[0].af_version
    assert colabfold_output.predictions[0].models_relaxed != None
    assert colabfold_output.predictions[0].models_unrelaxed != None


def test_af3_pred_detection() -> None:
    af3output = AFOutput("./tests/data/af3")
    assert af3output.predictions[0].num_ranks == 5
    assert af3output.predictions[0].af_version == "alphafold3"
    assert af3output.predictions[0].models_relaxed == None
    assert af3output.predictions[0].models_unrelaxed != None


def test_colabfold_pred_plddt() -> None:
    colabfold_output = AFOutput("./tests/data/colabfold")
    fig = colabfold_output.plot_all_plddts()[0]

    # fig.show()
    # input()


def test_colabfold_multimer_pred_plddt() -> None:
    colabfold_output = AFOutput("./tests/data/colabfold_multimer")
    fig = colabfold_output.plot_all_plddts()[0]

    # fig.show()
    # input()


def test_af3_pred_plddt() -> None:
    af3output = AFOutput("./tests/data/af3")
    fig = af3output.plot_all_plddts()[0]

    # fig.show()
    # input()


def test_af3_multimer_pred_plddt() -> None:
    af3output = AFOutput("./tests/data/af3_multimer")
    fig = af3output.plot_all_plddts()[0]

    # fig.show()
    # input()


def test_af3_pred_pae() -> None:
    af3output = AFOutput("./tests/data/af3")
    fig = af3output.plot_pae(af3output.predictions[0])

    # fig.show()
    # input()


def test_af3_multimer_pred_pae() -> None:
    af3output = AFOutput("./tests/data/af3_multimer")
    fig = af3output.plot_pae(af3output.predictions[0])

    # fig.show()
    # input()


def test_colabfold_multimer_pred_pae() -> None:
    af2output = AFOutput("./tests/data/colabfold_multimer")
    fig = af2output.plot_pae(af2output.predictions[0])

    fig.show()
    input()
