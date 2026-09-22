"""
===============================================================================
Title:      Fasta
Outline:    Fasta class to handle fasta files.
            It supports:
            - Read headers and sequences
            - Writing
Author:     Alejandro Sánchez Cano
Date:       22/09/2026
===============================================================================
"""

# Built-in modules
from pathlib import Path
from collections.abc import Sequence

class Fasta:

    def __init__(self, records: list[tuple] | None, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.records = records or []
        self.headers = [header for header, _ in self.records]
        self.sequences = [sequence for _, sequence in self.records]  

    def __parse(self, string: str):
        records = []
        header = None
        sequence_lines = []
        for line in string.splitlines():
            line = line.strip()
            if line.startswith('>'):
                if header:
                    records.append((header, ''.join(sequence_lines)))
                header = line[1:]  # Remove '>'
                sequence_lines = []
            else:
                sequence_lines.append(line)
        if header:
            records.append((header, ''.join(sequence_lines)))
        return records

    @classmethod
    def from_records(cls, records: list[tuple]):
        """
        Create a Fasta object from given records (list of tuples).

        Parameters
        ----------
        records : list of tuples
            List of (header, sequence) tuples.
        """
        return cls(records=records, path=None)

    @classmethod
    def from_file(cls, path: str | Path):
        """
        Create a Fasta object by reading a fasta file.
        
        Parameters
        ----------
        path : str or Path
            Path to the fasta file.
        """
        records = []
        with open(path, 'r') as fasta_file:
            string = fasta_file.read()
            records = cls.__parse(cls, string)
        return cls(records=records, path=path)

    @classmethod
    def from_string(cls, string: str):
        """
        Create a Fasta object from a string containing fasta formatted text.
        
        Parameters
        ----------
        string : str
            String containing fasta formatted text.
        """
        records = cls.__parse(cls, string)
        return cls(records=records, path=None)

    @classmethod
    def concatenate(cls, other: Sequence['Fasta']) -> 'Fasta':
        """
        Take multiple Fasta objects and combine them into one. It looses 
        whatever path it was saved in 'self.path'
        """
        records = [record for fasta in other for record in fasta.records]
        return cls(records=records, path=None)

    def __len__(self):
        """
        Return the number of records in the fasta file.
        """
        return len(self.records)

    def __repr__(self) -> str:
        """
        Return a string representation of the Fasta object.
        """
        return f"Fasta(records={len(self.records)}, path={self.path})"

    def write(
        self, 
        out_path: str | Path | None = None,
        mode: str = 'w'
    ) -> None:
        """
        Write the fasta records to a file.
        
        Parameters
        ----------
        out_path : str or Path, optional
            Path to the output fasta file. If None, uses the original path.
        mode: str, optional
            Writing mode from open() function because sometimes we do not want
            to override the content of an existing fasta file.
        """
        out_path = Path(out_path or self.path)
        with open(out_path, mode) as fasta_file:
            for header, sequence in self.records:
                fasta_file.write(f">{header}\n")
                fasta_file.write(f"{sequence}\n")

if __name__ == "__main__":
    # Example usage
    fasta = Fasta.from_file("output.fasta")
    print(f"Number of records: {len(fasta)}")
    print(fasta.headers)
    fasta = Fasta.from_text(">seq1\nACGT\n>seq2\nTGCAT")
    print(f"Number of records: {len(fasta)}")
    fasta.write("output.fasta")
    print(fasta.headers)