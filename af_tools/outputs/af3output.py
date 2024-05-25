from dataclasses import dataclass
import multiprocessing
from pathlib import Path
from typing import Sequence

from numpy.typing import NDArray
import orjson

from af_tools.outputs.afoutput import AFOutput
from af_tools.output_types import AF3Prediction, AF3Model


class AF3Output(AFOutput):

    def get_predictions(self) -> Sequence[AF3Prediction]:
        if self.process_number > 1:
            outputs = [
                x.parent for x in list(self.path.rglob("terms_of_use.md"))
            ]
            with multiprocessing.Pool(processes=self.process_number) as pool:
                results = pool.map(self._worker_get_pred, outputs)

            return [j for i in results for j in i]  # flatten the results
        else:
            af_version = "alphafold3"
            full_data_paths = sorted(self.path.glob("*_full_data*.json"))
            summary_data_paths = sorted(
                self.path.glob("*summary_confidences_*.json"))
            model_paths = sorted(self.path.glob("*.cif"))

            pred_name = full_data_paths[0].name.split("_full_data_")[0]
            models: list[AF3Model] = []
            for i, (full_data_path, summary_data_path,
                    model_path) in enumerate(
                        zip(full_data_paths, summary_data_paths, model_paths)):
                with open(full_data_path, "rb") as full_data_file, open(
                        summary_data_path, "rb") as summary_data_file:
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
                        token_chain_ends.append(chain_len +
                                                token_chain_ends[-1])

                models.append(
                    AF3Model(
                        name=pred_name,
                        model_path=model_path,
                        json_path=full_data_path,
                        rank=i + 1,
                        mean_plddt=sum(full_data["atom_plddts"]) /
                        len(full_data["atom_plddts"]),
                        ptm=summary_data["ptm"],
                        pae=full_data["pae"],
                        af_version=af_version,
                        atom_plddts=full_data["atom_plddts"],
                        atom_chain_ends=atom_chain_ends,
                        token_chain_ends=token_chain_ends,
                    ))
            return [
                AF3Prediction(
                    name=pred_name,
                    num_ranks=len(models),
                    af_version=af_version,
                    models=models,
                    is_colabfold=False,
                )
            ]

    def _worker_get_pred(self, path: Path) -> Sequence[AF3Prediction]:
        af3output = AF3Output(path=path, process_number=1)
        return af3output.predictions
