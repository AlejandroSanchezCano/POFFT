
# Third-party modules
from tqdm import tqdm

# Custom modules
from esm2 import ESM2
from misc import paths
from misc import config
from misc.logger import logger
from entity.collection import ProteinCollection
logger.info('Importing modules completed')

# Gather Protein objects
file_path = paths.COLLECTIONS / 'all.prot'
collection = ProteinCollection(
    file_path=file_path,
    datasets=["seq"]
)
proteins = [protein for protein in collection]


# Compute and store ESM2 embeddings
esm2 = ESM2(config.ESM2_MODEL)
for protein in tqdm(proteins, desc='Computing ESM2 embeddings'):
    data = [('Protein', protein.seq)]
    esm2.prepare_data(data)
    esm2.run_model()
    perresidue, _ = esm2.extract_representations()
    protein.esm2_embeddings[config.ESM2_MODEL] = perresidue

# Save updated proteins
collection = ProteinCollection(
    file_path=file_path,
    items=proteins
    )
collection.to_hdf5()