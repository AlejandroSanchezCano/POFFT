"""
===============================================================================
Title:      DomainArchitecture class
Outline:    Utility class for representing and manipulating domain 
            architectures as a list of lists of strings. Each inner list 
            represents a set of domains at a specific position in the
            architecture, and the outer list represents the sequential order
            of these sets.
            Example: [['PF00643.30'], ['PF00438.26', 'PF00170.27']]
Author:     Alejandro Sánchez Cano
Date:       17/09/2026
===============================================================================
"""

# Third-party modules
import pandas as pd

# Custom modules
from misc import paths

class DomainArchitecture:

    # Class variables
    _to_names = {}
    _to_accessions = {}

    def __init__(self, architecture: list[list[str]]):
        self.architecture = architecture

    def __repr__(self):
        return f"DomainArchitecture({self.architecture})"

    def __str__(self):
        return f"DomainArchitecture with {len(self.architecture)} domains"

    def __iter__(self):
        return iter(self.architecture)

    def __len__(self):
        return len(self.architecture)

    def __getitem__(self, index):
        return self.architecture[index]
    
    def __hash__(self):
        return hash(tuple(tuple(domain) for domain in self.architecture))

    def __eq__(self, other):
        if not isinstance(other, DomainArchitecture):
            return NotImplemented
        return self.architecture == other.architecture

    def prettify(self) -> str:
        '''
        Return a human-readable string representation of the domain architecture.

        Returns
        -------
        str
            A string representation of the domain architecture.
        '''
        return ' | '.join([
            ', '.join(domains)
            for domains in self.architecture
        ])

    def domains(self) -> set[str]:
        '''
        Unique domain names in the architecture.

        Returns
        -------
        set[str]
            A set of all domain names in the architecture.
        '''
        return set([
            domain
            for sublist in self.architecture 
            for domain in sublist
        ])

    def equivalent(self, other: 'DomainArchitecture') -> bool:
        '''
        Check if two DomainArchitecture instances are equivalent, meaning they
        contain at least one common domain at each corresponding position in 
        their architectures.
        For example, [['A'], ['B', 'C']] is equivalent to [['A'], ['C', 'D']]
        but not to [['A'], ['D', 'E']].

        Parameters
        ----------
        other : DomainArchitecture
            Another instance of DomainArchitecture to compare with.

        Returns
        -------
        bool
            True if the two architectures are equivalent, False otherwise.
        '''
        # Must be same length
        if len(self) != len(other):
            return False

        # Check each position
        for domains1, domains2 in zip(self.architecture, other.architecture):
            domains1_set = set(domains1)
            domains2_set = set(domains2)
            if domains1_set.isdisjoint(domains2_set):
                return False

        return True

    @classmethod
    def _load_pfam_mapping(cls) -> None:
        '''
        Load domtblout and extract Pfam accessions and names in a dictionary.
        '''
        # Already loaded
        if cls._to_names and cls._to_accessions:
            return
        
        # Load the domtblout file
        domtblout_path = paths.FAMILIES / 'hmmscan.domtblout'
        df = pd.read_csv(domtblout_path, sep='\t')

        # Create mapping
        for row in df.itertuples():
            cls._to_names[row.pfam_accession] = row.target_name
            cls._to_accessions[row.target_name] = row.pfam_accession

    def as_pfam_accessions(self) -> 'DomainArchitecture':
        '''
        Convert the architecture to Pfam accessions.

        Returns
        -------
        DomainArchitecture
            The architecture represented as Pfam accessions.
        '''
        # Already in Pfam accessions
        if all(domain.startswith("PF") for domain in self.domains()):
            return self.architecture
        
        # Load mapping
        self._load_pfam_mapping()

        # Convert to Pfam accessions
        architecture = [
            [self._to_accessions[domain] for domain in domains]
            for domains in self.architecture
        ]
        return DomainArchitecture(architecture)
    
    def as_pfam_names(self) -> 'DomainArchitecture':
        '''
        Convert the architecture to Pfam names.

        Returns
        -------
        DomainArchitecture
            The architecture represented as Pfam names.
        '''
        # Already in Pfam names
        if not any(domain.startswith("PF") for domain in self.domains()):
            return self.architecture

        # Load mapping if not already loaded
        self._load_pfam_mapping()

        # Convert to Pfam names
        architecture = [
            [self._to_names[domain] for domain in domains]
            for domains in self.architecture
        ]
        return DomainArchitecture(architecture)

if __name__ == "__main__":

    domain_architecture = [['PF00643.30'], ['PF00438.26', 'PF00170.27']]
    architecture = DomainArchitecture(domain_architecture)
    print(architecture.prettify())

    domain_architecture = [['A'], ['B', 'C'], ['D', 'E', 'F']]
    architecture2 = DomainArchitecture(domain_architecture)
    print(architecture)
    print({"architecture": architecture})

    print({architecture: "value", architecture2: "value"})

    
    print(architecture.domains())

    print('START')
    print(architecture.as_pfam_names())
    print('END')
    print(architecture.as_pfam_accessions())
    a = architecture.as_pfam_names()
    print(a.architecture)

    from tqdm import tqdm
    for idx in tqdm(range(100)):
        architecture.as_pfam_names()
        architecture.as_pfam_accessions()

    
    domain_architecture = [['PF00643.30'], ['PF00438.26', 'PF00170.27']]
    architecture = DomainArchitecture(domain_architecture)
    domain_architecture = [['PF00643.30'], ['sthelse', 'PF00170.27']]
    architecture2 = DomainArchitecture(domain_architecture)
    domain_architecture = [['PF00643.30'], ['sthelse', 'sthelse']]
    architecture3 = DomainArchitecture(domain_architecture)
    print(architecture.equivalent(architecture2))  # True
    print(architecture.equivalent(architecture3))  # False