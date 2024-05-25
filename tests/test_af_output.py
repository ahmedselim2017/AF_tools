import pytest

from af_tools.afparser import AFParser


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
