from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.autonotebook import tqdm

from af_tools.data_types.af2output import AF2Output
from af_tools.data_types.afsample2output import AFSample2Output
from af_tools.data_types.af3output import AF3Output


class AFParser():

    def __init__(self,
                 path: Path | str,
                 output_type: str,
                 process_number: int = 1,
                 sort_plddt: bool = True,
                 should_load: list[str] | None = None):
        self.path = self._check_path(path).absolute()
        self.output_type = output_type
        self.output_types = ["AF3", "COLAB_AF2"]
        self.process_number = process_number
        self.sort_plddt = sort_plddt
        self.should_load = should_load

    def _get_output_data(self,
                         path: None | Path = None,
                         output_type: None | str = None) -> list[list[Any]]:
        if path is None:
            path = self.path
        if output_type is None:
            output_type = self.output_type

        data: list[list[Any]] = []
        if output_type == "AF3":
            pbar = tqdm(path.rglob("*terms_of_use.md"))
            for o_terms_path in pbar:
                with open(o_terms_path, "r") as o_terms_file:
                    if "AlphaFold Server" not in o_terms_file.readline():
                        raise ValueError(
                            f"{path.parent} isn't an AF3 output directory")

                data.extend(
                    AF3Output(o_terms_path.parent,
                              should_load=self.should_load).get_data())

        elif output_type == "COLAB_AF2":
            pbar = tqdm(path.rglob("*.done.txt"))
            for donetxt_path in pbar:
                data.extend(
                    AF2Output(
                        donetxt_path.parent,
                        donetxt_path.with_suffix("").with_suffix("").name,
                        should_load=self.should_load).get_data())
        elif output_type == "AFSAMPLE2":
            data.extend(
                AFSample2Output(path, should_load=self.should_load).get_data())

        elif output_type == "MIXED":
            for o_type in self.output_types:
                data.extend(
                    self._get_output_data(path=path, output_type=o_type))
        return data

    def get_dataframe(self):
        data = self._get_output_data()
        df = pd.DataFrame(
            data,
            columns=[  # type: ignore
                "output_path", "pred_name", "output_type", "best_model_path",
                "unrel_model_path", "full_json_path", "summary_json_path",
                "plddt_chain_ends", "pae_chain_ends", "is_multimer", "plddt",
                "mean_plddt", "pae", "ptm", "iptm", "mult_conf",
                "contact_probs", "atom_chain_ids", "token_chain_ids",
                "token_res_ids"
            ])
        return df

    def _check_path(self, path: str | Path) -> Path:
        if isinstance(path, str):
            p = Path(path).absolute()
        else:
            p = path.absolute()
        if not p.is_dir():
            raise Exception(
                f"Alphafold output directory is not a valid directory: {p}")
        return p
