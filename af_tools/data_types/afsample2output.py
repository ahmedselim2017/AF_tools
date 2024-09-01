from typing import Any
from pathlib import Path
from af_tools.data_types.afoutput import AFOutput

from tqdm.autonotebook import tqdm
import pickle
import numpy as np


class AFSample2Output(AFOutput):

    def get_data(self) -> list[list[Any]]:
        data: list[list[Any]] = []

        pbar = tqdm(self.path.glob("unrelaxed*pdb"))
        for pdb_path in pbar:
            pickle_path = self.path / pdb_path.with_suffix(
                ".pkl").name.replace("unrelaxed", "result")
            with open(pickle_path, "rb") as pickle_file:
                pickle_data = pickle.load(pickle_file)

            iptm: float | None = None
            try:
                iptm = pickle_data["iptm"]
                if iptm is None:
                    is_multimer = False
                else:
                    is_multimer = True
            except KeyError as _:
                is_multimer = False

            relaxed_path: Path | None = None
            if Path(self.path /
                    pdb_path.name.replace("unrelaxed", "relaxed")).is_file():
                relaxed_path = self.path / pdb_path.name.replace(
                    "unrelaxed", "relaxed")

            data.append([
                self.path,
                self.path.name,
                "AFSAMPLE2",
                str(relaxed_path) if relaxed_path else str(pdb_path),
                str(pdb_path),
                str(pickle_path),
                None,  # summary JSON path,
                None,  # TODO: chain_ends,
                None,  # TODO: chain_ends
                is_multimer,
                pickle_data["plddt"],
                np.mean(pickle_data["plddt"]),
                pickle_data["predicted_aligned_error"],
                pickle_data["ptm"],
                iptm,
                0.2 * pickle_data["ptm"] + 0.8 * iptm if iptm else None,
                None,
                None,
                None,
                None
            ])

        return data
