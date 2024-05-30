from pathlib import Path

from af_tools.data_types.afoutput import AFOutput
from af_tools.data_types.af2output import AF2Output
from af_tools.data_types.af3output import AF3Output


class AFParser():

    def __init__(self, path: Path | str, process_number: int = 1):
        self.path = self.check_path(path)
        self.process_number = process_number

    def get_output(self) -> AFOutput:
        if (self.path / "config.json").is_file():
            return AF2Output(path=self.path,
                             process_number=self.process_number,
                             search_recursively=False,
                             is_colabfold=True)
        elif any(True for _ in self.path.rglob("config.json")):
            return AF2Output(path=self.path,
                             process_number=self.process_number,
                             search_recursively=True,
                             is_colabfold=True)
        elif (self.path / "ranking_debug.json").is_file():
            return AF2Output(path=self.path,
                             process_number=self.process_number,
                             search_recursively=False,
                             is_colabfold=False)
        elif any(True for _ in self.path.glob("*summary_confidences_*.json")):
            return AF3Output(path=self.path,
                             process_number=self.process_number,
                             search_recursively=False)
        elif any(True for _ in self.path.rglob("*summary_confidences_*.json")):
            return AF3Output(path=self.path,
                             process_number=self.process_number,
                             search_recursively=True)
        else:
            raise Exception(
                f"Given output directory does not contain Alphafold 2 or Alphafold 3 outputs: {self.path}"
            )

    def check_path(self, path: str | Path) -> Path:
        if isinstance(path, str):
            p = Path(path)
        else:
            p = path
        if not p.is_dir():
            raise Exception(
                f"Alphafold output directory is not a valid directory: {p}")
        return p
