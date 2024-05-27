import pytest

from af_tools.data_types import colabfold_msa


def test_colabfold_msa_sample() -> None:
    msa = colabfold_msa.ColabfoldMSA("./tests/data/colabfold/msa.a3m")

    for i in range(10):
        msa.sample_records(2**i, 5)
