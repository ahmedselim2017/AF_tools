from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import orjson

from af_tools.data_types.afoutput import AFOutput


class AF2Output(AFOutput):

    def __init__(self,
                 path: Path,
                 pred_name: Path,
                 process_number: int = 1,
                 should_load: set[str] | None = None,
                 is_colabfold: bool = True):

        self.is_colabfold = is_colabfold
        self.pred_name = pred_name
        self.should_load = should_load if should_load else set(
            ["mean_plddt", "mult_conf", "ptm", "iptm"])
        super().__init__(path=path, process_number=process_number)

    def get_data(self) -> list[list[Any]]:
        if self.is_colabfold:
            return self.get_colab_data()
        else:
            raise NotImplementedError(
                "Non Colabfold AF2 prediction parsing is not yet implemented.")

    def get_colab_data(self) -> list[list[Any]]:
        data: list[list[Any]] = []
        with open(self.path / f"{self.pred_name}.a3m") as pred_msa_file:
            msa_header_info = pred_msa_file.readline().replace("#",
                                                               "").split("\t")
            msa_header_seq_lengths = [
                int(x) for x in msa_header_info[0].split(",")
            ]
            is_multimer = True if len(msa_header_seq_lengths) == 1 else False

        m_unrel_paths = self.path.glob(
            f"{self.pred_name}_unrelaxed_rank_*.pdb")
        for m_unrel_path in m_unrel_paths:

            m_rel_path: Path | None = self.path / m_unrel_path.name[::-1].replace(
                "_unrelaxed_rank_"[::-1], "_relaxed_rank_"[::-1], 1)[::-1]
            assert isinstance(m_rel_path, Path)
            m_rel_path = m_rel_path if m_rel_path.is_file() else None

            m_scores_path: Path = self.path / m_unrel_path.name[::-1].replace(
                "_unrelaxed_rank_"[::-1], "_scores_rank_"[::-1], 1)[::-1]

            if self.should_load is not None:
                with open(m_scores_path, "rb") as m_scores_file:
                    m_scores_data = orjson.loads(m_scores_file.read())
                    if "mean_plddt" in self.should_load:
                        plddt = np.asarray(m_scores_data["plddt"], dtype=float)
                        mean_plddt = np.mean(plddt)
                        plddt = plddt if "plddt" in self.should_load else None
                    else:
                        mean_plddt = None
                        plddt = np.asarray(
                            m_scores_data["plddt"], dtype=float
                        ) if "plddt" in self.should_load else None

                    pae = np.asarray(
                        m_scores_data["pae"],
                        dtype=float) if "pae" in self.should_load else None
                    ptm = m_scores_data[
                        "ptm"] if "ptm" in self.should_load else None
                    if is_multimer:
                        iptm = m_scores_data[
                            "iptm"] if "ptm" in self.should_load else None
                        mult_conf = 0.8 * m_scores_data[
                            "iptm"] + 0.2 * m_scores_data[
                                "ptm"] if "mult_conf" in self.should_load else None
                    else:
                        iptm = None
                        mult_conf = None
                data.append([
                    self.path, self.pred_name, "COLAB_AF2",
                    m_rel_path if m_rel_path is not None else m_unrel_path,
                    m_unrel_path, m_scores_path, is_multimer, plddt,
                    mean_plddt, pae, ptm, iptm, mult_conf
                ])
            else:
                data.append([
                    self.path, self.pred_name, "COLAB_AF2",
                    m_rel_path if m_rel_path is not None else m_unrel_path,
                    m_unrel_path, m_scores_path, is_multimer, None, None, None,
                    None, None, None
                ])

        return data
