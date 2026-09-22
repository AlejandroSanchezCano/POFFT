# Built-in modules
import re
import json

# Third-party modules
from tqdm import tqdm

# Custom modules
from misc import paths
from misc.logger import logger
from tools.fasta import Fasta
from entity.protein import Protein
from entity.collection import ProteinCollection
logger.info('Importing modules completed')

###############################################################################
#######                           LOAD DATA                             #######
###############################################################################

# Load fasta
fasta_file = paths.FAMILIES / 'sequences.fasta'
fasta = Fasta.from_file(fasta_file)

# Load clustering
json_file = paths.FAMILIES / 'accession2representative.json'
with open(json_file, 'r') as handle:
    accession2representative = json.load(handle)

###############################################################################
#######                     ALL PROTEINS COLLECTION                      #######
###############################################################################

# Create Protein objects
all_proteins = []
for header, seq in tqdm(fasta.records, desc='Creating Protein objects'):
    uniprot = header.split('|')[1]
    taxon = int(re.search(r'OX=([0-9]+)', header).group(1))
    family = accession2representative[uniprot]
    protein = Protein(
        seq=seq, 
        uniprot=uniprot, 
        taxon=taxon, 
        family=family
    )
    all_proteins.append(protein)
logger.info(f"Number of total proteins: {len(all_proteins)}")

# Create ProteinCollection
protein_path = paths.COLLECTIONS / 'all.prot'
collection = ProteinCollection(
    file_path=protein_path,
    items=all_proteins
)
collection.to_hdf5()