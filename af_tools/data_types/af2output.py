import multiprocessing
from pathlib import Path
from collections.abc import Sequence
import operator

from natsort import natsorted
import numpy as np
import orjson
from tqdm import tqdm

from af_tools import utils
from af_tools.data_types.afoutput import AFOutput
from af_tools.output_types import AF2Prediction, AF2Model


class AF2Output(AFOutput):

    def __init__(self,
                 path: Path,
                 *args,
                 process_number: int = 1,
                 search_recursively: bool = False,
                 is_colabfold: bool = True,
                 sort_plddt: bool = False,
                 **kwargs):

        self.is_colabfold = is_colabfold
        super().__init__(path=path,
                         process_number=process_number,
                         search_recursively=search_recursively,
                         sort_plddt=sort_plddt)

    def get_predictions(self) -> Sequence[AF2Prediction]:
        if self.is_colabfold:
            return self.get_colabfold_predictions()
        return self.get_af2_predictions()

    def get_preds_from_colabfold_dir(
            self, colabfold_dir: Path) -> list[AF2Prediction]:
        predictions: list[AF2Prediction] = []
        with open(colabfold_dir / "config.json", "rb") as config_file:
            config_data = orjson.loads(config_file.read())

        af_version = config_data["model_type"]
        num_ranks = config_data["num_models"]

        # predictions: list[AF2Prediction] = []
        for pred_done_path in natsorted(list(
                colabfold_dir.glob("*.done.txt"))):
            pred_name = pred_done_path.name.split(".")[0]

            with open(colabfold_dir / f"{pred_name}.a3m", "r") as msa_file:
                msa_header_info = msa_file.readline().replace("#",
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

            model_unrel_paths = natsorted(
                colabfold_dir.glob(f"{pred_name}_unrelaxed_rank_*.pdb"))
            model_rel_paths = natsorted(
                colabfold_dir.glob(f"{pred_name}_relaxed_rank_*.pdb"))
            score_paths = natsorted(
                colabfold_dir.glob(f"{pred_name}_scores_rank_*.json"))

            models: list[AF2Model] = []
            for i, (model_unrel_path, score_path) in enumerate(
                    zip(model_unrel_paths, score_paths)):
                model_rel_path = None
                if i < config_data["num_relax"]:
                    try:
                        model_rel_path = model_rel_paths[i].absolute()
                    except IndexError as a:
                        print(model_rel_paths, i, pred_name)
                        print(a)
                        exit()

                with open(score_path, "rb") as score_file:
                    score_data = orjson.loads(score_file.read())
                pae = np.asarray(score_data["pae"])
                plddt = np.asarray(score_data["plddt"])
                ptm = float(score_data["ptm"])
                iptm = float(score_data["iptm"])

                models.append(
                    AF2Model(name=pred_name,
                             model_path=model_unrel_path.absolute(),
                             relaxed_pdb_path=model_rel_path,
                             json_path=score_path.absolute(),
                             rank=i + 1,
                             mean_plddt=np.mean(plddt, axis=0),
                             pae=pae,
                             af_version=af_version,
                             residue_plddts=plddt,
                             chain_ends=chain_ends,
                             ptm=ptm,
                             iptm=iptm))

            predictions.append(
                AF2Prediction(
                    name=pred_name,
                    num_ranks=num_ranks,
                    af_version=af_version,
                    models=models,
                    best_mean_plddt=models[0].mean_plddt,
                    is_colabfold=True,
                ))

        return predictions

    def get_colabfold_predictions(self) -> Sequence[AF2Prediction]:

        predictions: list[AF2Prediction] = []
        if self.search_recursively:
            outputs = natsorted(
                [x.parent for x in list(self.path.rglob("config.json"))])
            if self.process_number > 1:
                with multiprocessing.Pool(
                        processes=self.process_number) as pool:
                    results = tqdm(pool.map(utils.worker_af2output_get_pred,
                                            outputs),
                                   total=len(outputs),
                                   desc="Loading Colabfold predictions")
                    predictions = [j for i in results
                                   for j in i]  # flatten the results
            else:
                pbar = tqdm(outputs)
                for output_dir in pbar:
                    predictions += self.get_preds_from_colabfold_dir(
                        output_dir)

                    pbar.set_description(f"Reading {str(output_dir)}")
        else:
            predictions = self.get_preds_from_colabfold_dir(self.path)

        if self.sort_plddt:
            predictions = sorted(predictions,
                                 reverse=True,
                                 key=operator.attrgetter("best_mean_plddt"))

        return predictions

    def get_af2_predictions(self) -> Sequence[AF2Prediction]:
        return []
