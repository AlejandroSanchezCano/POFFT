
# Built-in modules
import os

# Third-party modules
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

# Custom modules
from core import seed
from misc import paths
from misc import config
from learn import utils
from tool.esm2 import ESM2
from core.split import Split
from misc.logger import logger
from core.epoch import EpochMTL
from core.tracker import Tracker
from model.mtl.frozen import Frozen
from core.loss import BinaryFocalLoss
from core.early_stop import EarlyStop
from model.bottleneck import Bottleneck
from core.inspector import ModelInspector
from core.task_weight import RankWeighting
from core.loader_cycler import LoaderCycler
from core.sampler import LengthBatchSampler
from core.dataset import ProteinPairDataset
from entity.collection import ProteinPairCollection
from model.classification_head import ClassificationHead
logger.info('Importing modules completed')

# Set seed
seed.set_seed(config.SEED)

# Job array
REPLICATE = 0 #int(os.getenv('SLURM_ARRAY_TASK_ID'))
logger.info(f'Running replicate: {REPLICATE}')

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

# Load tokenized sequences
tokenized = torch.load(
    paths.COLLECTIONS / 'families.tokenized',
    weights_only=False
)

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

# ESM2
esm2 = ESM2(config.ESM2_MODEL)

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

# Iterate over families
train_loaders = []
val_loaders = []
test_loaders = []
for family in config.FAMILIES:

    # Log family
    logger.info(f'Family: {family}')
    logger.info('-' * 60)

    # Select pairs from the test, val, test datasets (actually subsets)
    train_pairs = utils.extract_pairs(train_dataset)
    val_pairs = utils.extract_pairs(val_dataset)
    test_pairs = utils.extract_pairs(test_dataset)

    # Re-create datasets for the current family
    family_train_dataset = ProteinPairDataset(
        pairs=train_pairs,
        tokenized=tokenized
    )
    family_val_dataset = ProteinPairDataset(
        pairs=val_pairs,
        tokenized=tokenized
    )
    family_test_dataset = ProteinPairDataset(
        pairs=test_pairs,
        tokenized=tokenized
    )
    logger.info(f'Logging {family} training dataset:')
    family_train_dataset.log()
    logger.info(f'Logging {family} validation dataset:')
    family_val_dataset.log()
    logger.info(f'Logging {family} test dataset:')
    family_test_dataset.log()

    # Build samplers
    train_sampler = LengthBatchSampler(
        data_source=family_train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    val_sampler = LengthBatchSampler(
        data_source=family_val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    test_sampler = LengthBatchSampler(
        data_source=family_test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    logger.info(f'Train sampler: {len(train_sampler)} batches')
    logger.info(f'Validation sampler: {len(val_sampler)} batches')
    logger.info(f'Test sampler: {len(test_sampler)} batches')

    # Build dataloaders
    train_loader = DataLoader(
        family_train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        persistent_workers=True,
        worker_init_fn=seed.seed_worker,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        family_val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        persistent_workers=True,
        worker_init_fn=seed.seed_worker,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        family_test_dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        persistent_workers=True,
        worker_init_fn=seed.seed_worker,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f'Train dataloader: {len(train_loader)} batches')
    logger.info(f'Validation dataloader: {len(val_loader)} batches')
    logger.info(f'Test dataloader: {len(test_loader)} batches')

    # Append loaders
    train_loaders.append(train_loader)
    val_loaders.append(val_loader)
    test_loaders.append(test_loader)

# Create cyclers
train_cycler = LoaderCycler(train_loaders)
val_cycler = LoaderCycler(val_loaders)
test_cycler = LoaderCycler(test_loaders)

# Reseed
seed.set_seed(REPLICATE)

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
    for _ in config.FAMILIES
])

# Model
model = Frozen(
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
    lr=config.HEAD_LEARNING_RATE,
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

# Task weighting strategy
weighting_strategy = RankWeighting(
    number_of_tasks=len(config.FAMILIES),
    learning_rate=config.WEIGHT_LEARNING_RATE,
    epsilon=config.WEIGHT_EPSILON
)

# Epoch
epoch = EpochMTL(        
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    weighting_strategy=weighting_strategy,
)

# Tracker
tracker = Tracker()

###############################################################################
#######                        TRAIN AND EVALUATE                       #######
###############################################################################

# Loop over epochs
for epoch_idx in tqdm(range(config.EPOCHS), desc='Epochs', unit='epoch'):

    # Train
    train = epoch.train(train_cycler)
    # Evaluate
    val = epoch.evaluate(val_cycler)


    # Update weights
    pass

#FIXME: iterating over batches needs to also return the task index
#FIXME: cycler is good for training, but for evaluation it is just a dataloader


# We need to keep
# - Weights for each task
# - Losses for each task
# - Computed thresholds (general, not per-task) (val MCC-based)