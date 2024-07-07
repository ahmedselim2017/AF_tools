from dataclasses import dataclass
import multiprocessing
import operator
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
import orjson
from tqdm import tqdm
from natsort import natsorted

from af_tools import utils
from af_tools.data_types.afoutput import AFOutput
from af_tools.output_types import AF3Prediction, AF3Model


class AF3Output(AFOutput):

    def get_data(self) -> list[list[Any]]:
        data: list[list[Any]] = []
        self.should_load = []  # TODO

        m_paths = self.path.glob("*_model_*.cif")
        pred_name = None
        for m_path in m_paths:
            if pred_name is None:
                pred_name = "_model_".join(m_path.name.split("_model_")[:-1])

            m_summary_path = self.path / m_path.name[::-1].replace(
                "_model_"[::-1], "_summary_confidences_"[::-1], 1)[::-1]
            m_full_data_path = self.path / m_path.name[::-1].replace(
                "_model_"[::-1], "_full_data_"[::-1], 1)[::-1]

            plddt: None | NDArray = None
            mean_plddt: None | float = None
            pae: None | NDArray = None
            contact_probs: None | NDArray = None
            atom_chain_ids: None | list[str] = None
            token_chain_ids: None | list[str] = None
            token_res_ids: None | list[int] = None
            atom_chain_ends: None | list[int] = None
            token_chain_ends: None | list[int] = None

            if "mean_plddt" in self.should_load or "plddt" in self.should_load or "pae" in self.should_load or "contact_probs" in self.should_load:
                with open(m_full_data_path, "rb") as m_full_data_file:
                    m_full_data = orjson.loads(m_full_data_file.read())

                    if "mean_plddt" in self.should_load:
                        plddt = np.asarray(m_full_data["atom_plddts"],
                                           dtype=float)
                        mean_plddt = np.mean(plddt)
                        if "plddt" in self.should_load:
                            atom_chain_ids = m_full_data["atom_chain_ids"]
                        else:
                            plddt = None

                    if "pae" in self.should_load:
                        pae = np.asarray(m_full_data["pae"], dtype=float)
                        token_chain_ids = m_full_data["token_chain_ids"]
                        token_res_ids = m_full_data["token_res_ids"]

                    if "contact_probs" in self.should_load:
                        contact_probs = np.asarray(
                            m_full_data["contact_probs"], dtype=float)
                        token_chain_ids = m_full_data["token_chain_ids"]
                        token_res_ids = m_full_data["token_res_ids"]

                atom_chain_lengths: list[int] = []
                atom_id_old = ""
                chain_length = 0
                for atom_id in m_full_data["atom_chain_ids"]:
                    if atom_id != atom_id_old:
                        if atom_id_old != "":
                            atom_chain_lengths.append(chain_length)
                        chain_length = 1
                    else:
                        chain_length += 1
                    atom_id_old = atom_id

                atom_chain_lengths.append(chain_length)
                atom_chain_ends = []
                for chain_len in atom_chain_lengths:
                    if atom_chain_ends == []:
                        atom_chain_ends.append(chain_len)
                    else:
                        atom_chain_ends.append(chain_len + atom_chain_ends[-1])

                token_chain_lengths: list[int] = []
                token_id_old = ""
                chain_length = 0
                for token_id in m_full_data["token_chain_ids"]:
                    if token_id != token_id_old:
                        if token_id_old != "":
                            token_chain_lengths.append(chain_length)
                        chain_length = 1
                    else:
                        chain_length += 1
                    token_id_old = token_id
                token_chain_lengths.append(chain_length)

                token_chain_ends = []
                for chain_len in token_chain_lengths:
                    if token_chain_ends == []:
                        token_chain_ends.append(chain_len)
                    else:
                        token_chain_ends.append(chain_len +
                                                token_chain_ends[-1])

            ptm: None | float = None
            iptm: None | float = None
            mult_conf: None | float = None
            if self.should_load is not None:
                with open(m_summary_path, "rb") as m_summary_file:
                    m_summary_data = orjson.loads(m_summary_file.read())

                    ptm = m_summary_data[
                        "ptm"] if "ptm" in self.should_load else None
                    iptm = m_summary_data[
                        "iptm"] if "iptm" in self.should_load else None
                    mult_conf = None if iptm is None else m_summary_data[
                        "ranking_score"]

            data.append([
                self.path, pred_name, "AF3_SERVER", m_path, m_path,
                m_full_data_path, m_summary_path, atom_chain_ends,
                token_chain_ends, True if mult_conf is not None else None,
                plddt, mean_plddt, pae, ptm, iptm, mult_conf, contact_probs,
                atom_chain_ids, token_chain_ids, token_res_ids
            ])

        return data
