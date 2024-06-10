import pytest

from af_tools.data_types import colabfold_msa
from pathlib import Path


def test_colabfold_msa_sample() -> None:
    msa = colabfold_msa.ColabfoldMSA("T1185s2/T1185s2.a3m")

    for i in range(10):
        msa.sample_records(2**i, 5, output_dir=Path("T1185s2/T1185s2.a3m"))
