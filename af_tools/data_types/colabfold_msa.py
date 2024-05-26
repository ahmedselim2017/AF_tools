from pathlib import Path
import random

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

import math


class ColabfoldMSA:

    def __init__(self, path: Path | str):
        self.path = self._check_path(path)
        self.header, self.records = self._load_msa()

    def _check_path(self, path: str | Path) -> Path:
        if isinstance(path, str):
            p = Path(path)
        else:
            p = path
        if not p.is_file():
            raise Exception(f"Given MSA is not a valid file: {p}")
        return p

    def _load_msa(self) -> tuple[str, list[SeqRecord]]:
        records: list[SeqRecord] = []
        with open(self.path, "r") as msa_file:
            header = msa_file.readline()
            for record in SeqIO.parse(msa_file, "fasta"):
                records.append(record)

        return (header, records)

    def _get_samples(self, sample_lenght: int,
                     sample_count: int) -> list[list[SeqRecord]]:
        samples: list[list[SeqRecord]] = []
        sample: list[SeqRecord] = []
        for _ in range(sample_count):
            sample = [self.records[0]]
            sample += random.sample(self.records, sample_lenght)

            samples.append(sample)
        return samples

    def sample_records(self,
                       sample_lenght: int,
                       sample_count: int,
                       save_samples: bool = True,
                       output_path: Path | None = None,
                       make_subdir: bool = False) -> list[list[SeqRecord]]:
        samples = self._get_samples(sample_lenght=sample_lenght,
                                    sample_count=sample_count)
        if save_samples:
            if not output_path:
                if make_subdir:
                    output_path = self.path.parent / f"{self.path.stem}_MSAs" / f"length_{sample_lenght}-count_{sample_count}"
                else:
                    output_path = self.path.parent / f"{self.path.stem}_MSAs"
                output_path.mkdir(exist_ok=True, parents=True)

            for i, sample in enumerate(samples):
                name = f"length_{sample_lenght}-count_{sample_count}-{i:0{math.ceil(math.log10(sample_count))}}.a3m"
                with open(output_path / name, "w") as output_file:
                    output_file.write(self.header)
                    SeqIO.write(sample, output_file, "fasta-2line")

        return samples
