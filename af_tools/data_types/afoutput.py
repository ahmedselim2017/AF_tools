from collections.abc import Sequence
from pathlib import Path

from numpy.typing import NDArray


class AFOutput:

    def __init__(self,
                 path: Path,
                 ref_path: Path | None = None,
                 should_load: list[str] | None = None,
                 use_brotli: bool = False):
        self.path = self.check_path(path)
        self.use_brotli = use_brotli
        self.ref_path = ref_path
        self.should_load = should_load if should_load else set(
            ["mean_plddt", "mult_conf", "ptm", "iptm"])

        self.ref_rmsds: NDArray | None = None
        self.pairwise_rmsds: NDArray | None = None
        self.ref_tms: NDArray | None = None
        self.pairwise_tms: NDArray | None = None

    def check_path(self, path: str | Path) -> Path:
        if isinstance(path, str):
            p = Path(path)
        else:
            p = path
        if not p.is_dir():
            raise Exception(
                f"Alphafold output directory is not a valid directory: {p}")
        return p

    def get_predictions(self) -> Sequence:
        raise Exception("Can't get predictions")
