from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

from numpy.typing import NDArray


@dataclass(frozen=True, kw_only=True, slots=True)
class AFModel:
    name: str
    model_path: Path
    json_path: Path
    rank: int
    mean_plddt: float
    ptm: float
    pae: NDArray
    iptm: float
    af_version: str

    def get_best_model_path(self) -> Path:
        return self.model_path


@dataclass(frozen=True, kw_only=True, slots=True)
class AFPrediction:
    name: str
    num_ranks: int
    af_version: str
    is_colabfold: bool
    best_mean_plddt: float
    models: Sequence[AFModel]


@dataclass(frozen=True, kw_only=True, slots=True)
class AF2Model(AFModel):
    residue_plddts: NDArray
    chain_ends: list[int]
    relaxed_pdb_path: Path | None

    def get_best_model_path(self) -> Path:
        if self.relaxed_pdb_path is not None:
            return self.relaxed_pdb_path
        else:
            return self.model_path


@dataclass(frozen=True, kw_only=True, slots=True)
class AF2Prediction(AFPrediction):
    models: Sequence[AF2Model]
    pass


@dataclass(frozen=True, kw_only=True, slots=True)
class AF3Model(AFModel):
    atom_plddts: NDArray
    atom_chain_ends: list[int]
    token_chain_ends: list[int]


@dataclass(frozen=True, kw_only=True, slots=True)
class AF3Prediction(AFPrediction):
    models: Sequence[AF3Model]
    pass
