# Third-party modules
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

# Custom modules
from core import seed
from misc import paths
from misc import config
from tool.esm2 import ESM2
from core.split import Split
from core.epoch import Epoch
from misc.logger import logger
from core.tracker import Tracker
from core.loss import BinaryFocalLoss
from core.early_stop import EarlyStop
from model.bottleneck import Bottleneck
from core.performance import Performance
from core.inspector import ModelInspector
from core.sampler import LengthBatchSampler
from core.dataset import ProteinPairDataset
from model.mtl.frozen import MultiTaskFrozen
from entity.collection import ProteinPairCollection
from model.classification_head import ClassificationHead
logger.info('Importing modules completed')

# Set seed
seed.set_seed(config.SEED)

###############################################################################
#######                              LOAD                               #######
###############################################################################

# Load protein pairs
protein_path = paths.COLLECTIONS / 'families.prot'
file_path = paths.COLLECTIONS / 'families.pair'
collection = ProteinPairCollection(
    file_path=file_path,
    proteins=protein_path,
)

# Initialize ESM2
esm2 = ESM2(config.ESM2_MODEL)

# Tokenize sequences
tokenized = {
    protein.uniprot: esm2.tokenize(protein.seq)
    for protein in tqdm(collection.proteins, desc="Tokenizing proteins")
}

# Create dataset
dataset = ProteinPairDataset(
    pairs=collection.pairs,
    tokenized=tokenized
)
dataset.log()

# Split dataset
splitter = Split(dataset)
train_dataset, val_dataset, test_dataset = splitter.simple(
    train_size=config.TRAIN_FRACTION,
    val_size=config.VAL_FRACTION,
    test_size=config.TEST_FRACTION
)

# Dynamic padding
collator = DataCollatorWithPadding(
    tokenizer=esm2.tokenizer,
    padding=True,
    return_tensors='pt'
)

# Collate function
def collate_fn(batch: list[dict]) -> dict:
    
    # Access elements
    tokenized = [item['input'] for item in batch]
    labels = [item['label'] for item in batch]
    identifiers = [item['identifier'] for item in batch]
    tokenized1, tokenized2 = zip(*tokenized)

    # Convert to specific format for collator: dictionary with (seq_len) ids and masks
    batch1 = [
        {
            'input_ids': input_ids.squeeze(0), 
            'attention_mask': attention_mask.squeeze(0)
        }
        for input_ids, attention_mask in tokenized1
    ]
    batch2 = [
        {
            'input_ids': input_ids.squeeze(0), 
            'attention_mask': attention_mask.squeeze(0)
        }
        for input_ids, attention_mask in tokenized2
    ]

    # Collate batches
    collated1 = collator(batch1)
    collated2 = collator(batch2)

    return {
        'inputs': (
            (collated1['input_ids'], collated1['attention_mask']),
            (collated2['input_ids'], collated2['attention_mask'])
        ),
        'labels': torch.stack(labels),
        'identifiers': identifiers
    }

# Build samplers
train_sampler = LengthBatchSampler(
    data_source=train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True
)
val_sampler = LengthBatchSampler(
    data_source=val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False
)
test_sampler = LengthBatchSampler(
    data_source=test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False
)
logger.info(f'Train sampler: {len(train_sampler)} batches')
logger.info(f'Validation sampler: {len(val_sampler)} batches')
logger.info(f'Test sampler: {len(test_sampler)} batches')

# Build dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    collate_fn=collate_fn,
    num_workers=config.NUM_WORKERS,
    persistent_workers=True,
    worker_init_fn=seed.seed_worker
)
val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_sampler,
    collate_fn=collate_fn,
    num_workers=config.NUM_WORKERS,
    persistent_workers=True,
    worker_init_fn=seed.seed_worker
)
test_loader = DataLoader(
    test_dataset,
    batch_sampler=test_sampler,
    collate_fn=collate_fn,
    num_workers=config.NUM_WORKERS,
    persistent_workers=True,
    worker_init_fn=seed.seed_worker
)
logger.info(f'Train dataloader: {len(train_loader)} batches')
logger.info(f'Validation dataloader: {len(val_loader)} batches')
logger.info(f'Test dataloader: {len(test_loader)} batches')

###############################################################################
#######                           MODEL SETUP                           #######
###############################################################################

# Bottleneck layer
bottleneck = Bottleneck(
    input_dim=esm2.hidden_size * 2,  # Concatenated representations
    bottleneck_dim=config.BOTTLENECK_DIM,
    dropout=config.BOTTLENECK_DROPOUT,
)

# Classification heads
heads = nn.ModuleList([
    ClassificationHead(
        input_dim=config.BOTTLENECK_DIM,
        hidden_dims=config.CLASSIFICATION_HEAD_HIDDEN_DIMS,
        dropout=config.CLASSIFICATION_HEAD_DROPOUT,
    )
    for _ in range(len(config.FAMILIES))
])

# Model
model = MultiTaskFrozen(
    encoder=esm2.model, 
    bottleneck=bottleneck,
    heads=heads
)

# Inspector
inspector = ModelInspector(model)
logger.info(f'Model initialized: {model.__class__.__name__}')
logger.info(f'Total parameters: {inspector.num_parameters(trainable=False)}')
logger.info(f'Trainable parameters: {inspector.num_parameters(trainable=True)}')

# Optimizer
optimizer = torch.optim.AdamW(
    inspector.parameters(trainable=True),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY
)

# Early stopping
early_stop = EarlyStop(
    patience=config.PATIENCE,
    min_delta=config.MIN_DELTA,
)

# Loss function
loss_fn = BinaryFocalLoss(
    alpha=config.FOCAL_LOSS_ALPHA,
    gamma=config.FOCAL_LOSS_GAMMA,
)

# Epoch
epoch = Epoch(        
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
)

# Tracker
tracker = Tracker()