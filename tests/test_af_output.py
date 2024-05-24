import pytest

from af_tools.afparser import AFParser


def test_af3_hist() -> None:
    af3output = AFParser("./tests/data/af3_recursive",
                         process_number=12).get_output()
    fig = af3output.plot_plddt_hist()

    fig.show()
    input()
    exit()


def test_invalid_path() -> None:
    with pytest.raises(Exception) as e_info:
        AFParser("./tests/data/wrongpath").get_output()


def test_colabfold_pred_detection() -> None:
    colabfold_output = AFParser("./tests/data/colabfold").get_output()
    assert colabfold_output.predictions[0].num_ranks == 5
    assert "alphafold2" in colabfold_output.predictions[0].af_version
    assert colabfold_output.predictions[0].models != None


def test_af3_pred_detection() -> None:
    af3output = AFParser("./tests/data/af3").get_output()
    assert af3output.predictions[0].num_ranks == 5
    assert af3output.predictions[0].af_version == "alphafold3"
    assert af3output.predictions[0].models != None


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
