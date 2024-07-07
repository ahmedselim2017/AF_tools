from pathlib import Path
from typing import Any

from af_tools.data_types.afoutput import AFOutput
from af_tools.data_types.af2output import AF2Output
from af_tools.data_types.af3output import AF3Output


class AFParser():

    def __init__(self,
                 path: Path | str,
                 output_type: str,
                 process_number: int = 1,
                 sort_plddt: bool = True):
        self.path = self.check_path(path).absolute()
        self.output_type = output_type
        self.output_types = ["AF3", "COLAB"]
        self.process_number = process_number
        self.sort_plddt = sort_plddt

    def get_output(self,
                   path: None | Path = None,
                   output_type: None | str = None) -> list[list[Any]]:
        if path is None:
            path = self.path
        if output_type is None:
            output_type = self.output_type

        data: list[list[Any]] = []
        if output_type == "AF3":
            for o_terms_path in path.rglob("terms_of_use.md"):
                with open(o_terms_path, "r") as o_terms_file:
                    if "AlphaFold Server" not in o_terms_file.readline():
                        raise ValueError(
                            f"{path.parent} isn't an AF3 output directory")

                data.extend(
                    AF3Output(
                        o_terms_path.parent,  # type: ignore
                        self.process_number).get_models())  # type: ignore

        elif output_type == "COLAB":
            for donetxt_path in path.rglob("*.done.txt"):
                data.extend(
                    AF2Output(
                        donetxt_path.parent,  # type: ignore
                        self.process_number).get_models())  # type: ignore
        elif output_type == "MIXED":
            for o_type in self.output_types:
                data.extend(self.get_output(path=path, output_type=o_type))
        return data

    def check_path(self, path: str | Path) -> Path:
        if isinstance(path, str):
            p = Path(path).absolute()
        else:
            p = path.absolute()
        if not p.is_dir():
            raise Exception(
                f"Alphafold output directory is not a valid directory: {p}")
        return p
