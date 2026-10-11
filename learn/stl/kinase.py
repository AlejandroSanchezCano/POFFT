"""
===============================================================================
Title:      Train
Outline:    Trains a deep learning model with the following characteristics:
            - Framework: single-task learning (STL)
            - Dataset: protein pairs from the family PK_Tyr_Ser-Thr, Pkinase 
                (kinases)
            - Model: ESM2 frozen encoder + classification head
            - Loss: binary focal loss
Author:     Alejandro Sánchez Cano
Date:       08/10/2026
Time:       
===============================================================================
"""

# Built-in modules
import os

# Third-party modules
import torch
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
from core.epoch import EpochSTL
from core.tracker import Tracker
from model.stl import SingleTask
from core.loss import BinaryFocalLoss
from core.early_stop import EarlyStop
from core.inspector import ModelInspector
from core.sampler import LengthBatchSampler
from core.dataset import ProteinPairDataset
from entity.collection import ProteinPairCollection
from model.classification_head import ClassificationHead
logger.info('Importing modules completed')

# Choose family
FAMILY_IDX = 0 # PK_Tyr_Ser-Thr, Pkinase = kinases

# Set seed
seed.set_seed(config.SEED)

# Job array
REPLICATE = int(os.getenv('SLURM_ARRAY_TASK_ID'))
logger.info(f'Running replicate: {REPLICATE}')

# Variables
FRAMEWORK = 'stl'
TUNE = 'frozen'
TASK_NAME = 'kinase'

###############################################################################
#######                           LOAD DATA                             #######
###############################################################################

# Load protein pairs
protein_path = paths.COLLECTIONS / 'families.prot'
file_path = paths.COLLECTIONS / 'families.pair'
collection = ProteinPairCollection(
    file_path=file_path,
    proteins=protein_path,
)

# Select proteins
family = config.FAMILIES[FAMILY_IDX]
collection.pairs = [
    pair
    for pair in collection.pairs
    if pair.p1.family in family or pair.p2.family in family
]
logger.info(f'Family: {family}')

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

###############################################################################
#######                       BUILD DATALOADERS                         #######
###############################################################################

# ESM2
esm2 = ESM2(config.ESM2_MODEL)

# Make collate function
collate_fn = utils.make_collate_fn(esm2.tokenizer)

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
    worker_init_fn=seed.seed_worker,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_sampler,
    collate_fn=collate_fn,
    num_workers=config.NUM_WORKERS,
    persistent_workers=True,
    worker_init_fn=seed.seed_worker,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_sampler=test_sampler,
    collate_fn=collate_fn,
    num_workers=config.NUM_WORKERS,
    persistent_workers=True,
    worker_init_fn=seed.seed_worker,
    pin_memory=True
)
logger.info(f'Train dataloader: {len(train_loader)} batches')
logger.info(f'Validation dataloader: {len(val_loader)} batches')
logger.info(f'Test dataloader: {len(test_loader)} batches')

# Reseed
seed.set_seed(REPLICATE)

###############################################################################
#######                           MODEL SETUP                           #######
###############################################################################

# Classification head
head = ClassificationHead(
    input_dim=esm2.hidden_size * 2,  # Concatenated representations
    hidden_dims=config.CLASSIFICATION_HEAD_HIDDEN_DIMS,
    dropout=config.CLASSIFICATION_HEAD_DROPOUT,
)

# Model
model = SingleTask(
    encoder=esm2.model, 
    head=head,
    tune=TUNE,
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

# Epoch
epoch = EpochSTL(        
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
)

# Tracker
tracker = Tracker()

###############################################################################
#######                        TRAIN AND EVALUATE                       #######
###############################################################################

# Construct output directory
main_dir = paths.MODELS
replicate = str(REPLICATE)
out_dir = main_dir / FRAMEWORK / TASK_NAME / TUNE / replicate
# example: path.MODELS/mtl/combined/frozen/0
out_dir.mkdir(parents=True, exist_ok=True)
logger.info(f'Output directory: {out_dir}')

# Save config
utils.save_config(config=config, out_dir=out_dir)

# Loop over epochs
for epoch_idx in tqdm(range(config.EPOCHS), desc='Epochs', unit='epoch'):

    # Epoch directory
    epoch_dir = out_dir / str(epoch_idx + 1)
    epoch_dir.mkdir(parents=True, exist_ok=True)

    # Train
    train = epoch.train(train_loader)

    # Evaluate
    val = epoch.evaluate(val_loader)

    # Track
    tracker.track(
        train=train,
        validation=val,
        model=model
    )

    # Logging
    logger.info(f'Epoch {epoch_idx+1}/{config.EPOCHS}')
    logger.info(f'Train loss = {train.loss:.4f}')
    logger.info(f'Validation loss = {val.loss:.4f}')
    logger.info(f'Train balanced accuracy = {tracker.performance["train"].balanced_accuracy:.4f}')
    logger.info(f'Validation balanced accuracy = {tracker.performance["val"].balanced_accuracy:.4f}')
    logger.info(f'Train AUPRC = {tracker.performance["train"].auprc:.4f}')
    logger.info(f'Validation AUPRC = {tracker.performance["val"].auprc:.4f}')

    # Save model
    torch.save(
        model.state_dict(),
        epoch_dir / 'model.pt'
    )

    # Save results
    torch.save(
        train,
        epoch_dir / 'train.pt'
    )
    torch.save(
        val,
        epoch_dir / 'val.pt'
    )

    # Early stopping
    if early_stop(loss=val.loss):
        logger.info('Early stopping triggered.')
        break

# Best epoch
logger.info(
    f"Epoch {tracker.best_epoch + 1} had the best " 
    f"validation loss: {tracker.best_loss:.4f}"
)

# Save best model
torch.save(
    tracker.best_model, 
    out_dir / 'model.pt'
)

# Evaluate on test set with best model
model.load_state_dict(tracker.best_model)
epoch.model = model
test = epoch.evaluate(test_loader)

# Save test results
torch.save(
    test,
    out_dir / 'test.pt'
)