from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import brotli
import orjson

from af_tools.data_types.afoutput import AFOutput


class AF2Output(AFOutput):

    def __init__(self,
                 path: Path,
                 pred_name: str,
                 should_load: list[str] | None = None,
                 is_colabfold: bool = True,
                 use_brotli: bool = False):

        self.is_colabfold = is_colabfold
        self.pred_name = pred_name
        super().__init__(path=path,
                         should_load=should_load,
                         use_brotli=use_brotli)

    def get_data(self) -> list[list[Any]]:
        if self.is_colabfold:
            return self.get_colab_data()
        else:
            print(self.path)
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
            msa_header_seq_cardinalities = [
                int(x) for x in msa_header_info[1].split(",")
            ]

            chain_lengths: list[int] = []
            for seq_len, seq_cardinality in zip(msa_header_seq_lengths,
                                                msa_header_seq_cardinalities):
                chain_lengths += [seq_len] * seq_cardinality

            chain_ends: list[int] = []
            for chain_len in chain_lengths:
                if chain_ends == []:
                    chain_ends.append(chain_len)
                else:
                    chain_ends.append(chain_len + chain_ends[-1])

            is_multimer = True if len(msa_header_seq_lengths) != 1 else False

        m_unrel_paths = self.path.glob(
            f"{self.pred_name}_unrelaxed_rank_*.pdb*")
        for m_unrel_path in m_unrel_paths:

            m_rel_path: Path | None = self.path / m_unrel_path.name[::-1].replace(
                "_unrelaxed_rank_"[::-1], "_relaxed_rank_"[::-1], 1)[::-1]
            assert isinstance(m_rel_path, Path)
            m_rel_path = m_rel_path if m_rel_path.is_file() else None

            m_scores_path: Path = self.path / m_unrel_path.name[::-1].replace(
                "_unrelaxed_rank_"[::-1], "_scores_rank_"[::-1], 1)[::-1]

            if not m_scores_path.is_file():
                print(self.path)
                raise ValueError(f"{m_scores_path.absolute()} does not exist.")

            m_scores_path = m_scores_path.with_suffix(".json")
            if (not m_scores_path.is_file() or self.use_brotli
                ) and m_scores_path.with_suffix(".json.br").is_file():
                m_scores_path = m_scores_path.with_suffix(".json.br")
            assert m_scores_path.is_file()

            plddt: None | NDArray = None
            mean_plddt: None | float = None
            pae: None | NDArray = None
            ptm: None | float = None
            iptm: None | float = None
            mult_conf: None | float = None

            if self.should_load is not None:
                with open(m_scores_path, "rb") as m_scores_file:
                    m_scores_data: dict | None = None
                    if m_scores_path.suffix == ".json":
                        m_scores_data = orjson.loads(m_scores_file.read())
                    elif m_scores_path.suffix == ".br":
                        m_scores_data = orjson.loads(
                            brotli.decompress(m_scores_file.read()))

                    assert m_scores_data
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
                self.path,
                self.pred_name,
                "AF2_COLAB",
                str(m_rel_path)
                if m_rel_path is not None else str(m_unrel_path),
                str(m_unrel_path),
                str(m_scores_path),
                None,  # summary JSON path
                chain_ends,  # chain_ends for plddt
                chain_ends,  # chain_ends for pae
                is_multimer,
                plddt,
                mean_plddt,
                pae,
                ptm,
                iptm,
                mult_conf,
                None,  # contact_probs
                None,  # atom_chain_ids
                None,  # token_chain_ids
                None  # token_res_ids
            ])
        return data
