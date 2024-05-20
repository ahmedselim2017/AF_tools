from af_tools.analyze import AFOutput
import pytest


def test_invalid_path() -> None:
    with pytest.raises(Exception) as e_info:
        AFOutput("./data/wrongpath")


def test_af2_pred() -> None:
    af2output = AFOutput("./data/af2")
    assert af2output.predictions[0].num_ranks == 5
    assert af2output.predictions[0].af_version == 2
    assert af2output.predictions[0].models_relaxed != None
    assert af2output.predictions[0].models_unrelaxed != None


def test_af3_pred() -> None:
    af3output = AFOutput("./data/af3")
    assert af3output.predictions[0].num_ranks == 5
    assert af3output.predictions[0].af_version == 3
    assert af3output.predictions[0].models_relaxed == None
    assert af3output.predictions[0].models_unrelaxed != None
