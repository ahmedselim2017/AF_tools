from dataclasses import dataclass
import multiprocessing
import operator
from pathlib import Path
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
import orjson
from tqdm import tqdm
from natsort import natsorted

from af_tools import utils
from af_tools.data_types.afoutput import AFOutput
from af_tools.output_types import AF3Prediction, AF3Model


class AF3Output(AFOutput):

    def get_preds_from_af3_dir(self, af3dir: Path) -> list[AF3Prediction]:
        af_version = "alphafold3"
        full_data_paths = natsorted(af3dir.rglob("*_full_data*.json"))
        summary_data_paths = natsorted(
            af3dir.glob("*summary_confidences_*.json"))
        model_paths = natsorted(af3dir.glob("*.cif"))

        pred_name = full_data_paths[0].name.split("_full_data_")[0]
        models: list[AF3Model] = []
        for i, (full_data_path, summary_data_path, model_path) in enumerate(
                zip(full_data_paths, summary_data_paths, model_paths)):
            with open(full_data_path,
                      "rb") as full_data_file, open(summary_data_path,
                                                    "rb") as summary_data_file:
                full_data = orjson.loads(full_data_file.read())
                summary_data = orjson.loads(summary_data_file.read())

            atom_chain_lengths: list[int] = []
            atom_id_old = ""
            chain_length = 0
            for atom_id in full_data["atom_chain_ids"]:
                if atom_id != atom_id_old:
                    if atom_id_old != "":
                        atom_chain_lengths.append(chain_length)
                    chain_length = 1
                else:
                    chain_length += 1
                atom_id_old = atom_id
            atom_chain_lengths.append(chain_length)

            atom_chain_ends: list[int] = []
            for chain_len in atom_chain_lengths:
                if atom_chain_ends == []:
                    atom_chain_ends.append(chain_len)
                else:
                    atom_chain_ends.append(chain_len + atom_chain_ends[-1])

            token_chain_lengths: list[int] = []
            token_id_old = ""
            chain_length = 0
            for token_id in full_data["token_chain_ids"]:
                if token_id != token_id_old:
                    if token_id_old != "":
                        token_chain_lengths.append(chain_length)
                    chain_length = 1
                else:
                    chain_length += 1
                token_id_old = token_id
            token_chain_lengths.append(chain_length)

            token_chain_ends: list[int] = []
            for chain_len in token_chain_lengths:
                if token_chain_ends == []:
                    token_chain_ends.append(chain_len)
                else:
                    token_chain_ends.append(chain_len + token_chain_ends[-1])

            atom_plddts = np.asarray(full_data["atom_plddts"])
            pae = np.asarray(full_data["pae"])
            models.append(
                AF3Model(
                    name=pred_name,
                    model_path=model_path,
                    json_path=full_data_path,
                    rank=i + 1,
                    mean_plddt=np.mean(atom_plddts, axis=0),
                    ptm=summary_data["ptm"],
                    pae=pae,
                    af_version=af_version,
                    atom_plddts=atom_plddts,
                    atom_chain_ends=atom_chain_ends,
                    token_chain_ends=token_chain_ends,
                ))
        return [
            AF3Prediction(name=pred_name,
                          num_ranks=len(models),
                          af_version=af_version,
                          models=models,
                          best_mean_plddt=models[0].mean_plddt,
                          is_colabfold=False)
        ]

    def get_predictions(self) -> Sequence[AF3Prediction]:
        predictions: list[AF3Prediction] = []

        if self.search_recursively:
            outputs = [
                x.parent for x in list(self.path.rglob("terms_of_use.md"))
            ]
            if self.process_number > 1:
                with multiprocessing.Pool(
                        processes=self.process_number) as pool:
                    results = tqdm(pool.imap_unordered(
                        utils.worker_af3output_get_pred, outputs),
                                   total=len(outputs),
                                   desc="Loading AF3Outputs")

                    predictions = [j for i in results for j in i]
            else:
                pbar = tqdm(outputs)
                for output_dir in pbar:
                    predictions += self.get_preds_from_af3_dir(output_dir)

                    pbar.set_description(f"Reading {str(output_dir)}")
        else:
            predictions = self.get_preds_from_af3_dir(self.path)

        if self.sort_plddt:
            predictions = sorted(predictions,
                                 reverse=True,
                                 key=operator.attrgetter("best_mean_plddt"))
        return predictions
