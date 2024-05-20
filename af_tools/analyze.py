from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, kw_only=True, slots=True)
class PredictedModel:
    name: str
    pdb_path: Path
    json_path: Path
    rank: int
    res_length: int
    mean_pLDDT: float
    pTM: float
    pae: np.ndarray
    atom_pLDDT: np.ndarray | None
    residue_pLDDT: np.ndarray | None
    af_version: int
    is_relaxed: bool | None


@dataclass(frozen=True, kw_only=True, slots=True)
class Prediction:
    name: str
    num_ranks: int
    path: Path
    af_version: int
    models_relaxed: list[PredictedModel]
    models_unrelaxed: list[PredictedModel]


class AFOutput:

    def __init__(self, path: str):
        self.path: Path = self.check_path(path)

    def check_path(self, path) -> Path:
        p: Path = Path(path)
        if not p.is_dir():
            raise Exception(
                f"Alphafold output directory is not a valid directory: {p}")
        return p

    def get_predictions(self):
        for af2_pred_done in self.path.glob("*.done.txt"):
            name: str = af2_pred_done.name.split(".")[0]

            m_unrel_pdb_paths: list[Path] = sorted(
                self.path.glob(f"{name}_unrelaxed_rank_*pdb"))
            m_json_paths: list[Path] = sorted(
                self.path.glob(f"{name}_scores_rank_*pdb"))

            unrelaxed_models: list[PredictedModel] = []
            relaxed_models: list[PredictedModel] = []
            for i, (m_unrel_pdb_path, m_json_path) in enumerate(
                    zip(m_unrel_pdb_paths, m_json_paths)):

                with open(m_json_path, "r") as m_json_file:
                    m_json: dict = json.load(m_json_file)

                np_plddt: NDArray = np.asarray(m_json["plddt"])
                np_pae: NDArray = np.asarray(m_json["pae"])
                unrelaxed_models.append(
                    PredictedModel(name=name,
                                   pdb_path=m_unrel_pdb_path,
                                   json_path=m_json_path,
                                   rank=i + 1,
                                   res_length=len(m_json["plddt"]),
                                   mean_pLDDT=np.average(np_plddt, axis=0),
                                   pTM=m_json["ptm"],
                                   pae=np_pae,
                                   atom_pLDDT=None,
                                   residue_pLDDT=np_plddt,
                                   af_version=2,
                                   is_relaxed=False))

            m_rel_pdb_paths: list[Path] = sorted(
                self.path.glob(f"{name}_relaxed_rank_*pdb"))
            for i, m_rel_pdb_path in enumerate(m_rel_pdb_paths):

                m_json_path: Path = next(
                    self.path.glob(
                        f"{name}_scores_rank_{i+1:03d}_alphafold2*json"))
                with open(m_json_path, "r") as m_json_file:
                    m_json: dict = json.load(m_json_file)

                np_plddt = np.asarray(m_json["plddt"])
                np_pae = np.asarray(m_json["pae"])
                relaxed_models.append(
                    PredictedModel(name=name,
                                   pdb_path=m_rel_pdb_path,
                                   json_path=m_json_path,
                                   rank=i + 1,
                                   res_length=len(m_json["plddt"]),
                                   mean_pLDDT=np.average(np_plddt, axis=0),
                                   pTM=m_json["ptm"],
                                   pae=np_pae,
                                   atom_pLDDT=None,
                                   residue_pLDDT=np_plddt,
                                   af_version=2,
                                   is_relaxed=False))
            Prediction(
                name=name,
                num_ranks=len(unrelaxed_models),
                path=self.path,
                af_version=2,
                models_relaxed=relaxed_models,
                models_unrelaxed=unrelaxed_models,
            )

            return
