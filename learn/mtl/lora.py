# Built-in modules
import os

# Third-party modules
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

# Custom modules
from core import seed
from misc import paths
from misc import config
from learn import utils
from core.hook import Hook
from tool.esm2 import ESM2
from peft import LoraConfig
from core.split import Split
from misc.logger import logger
from core.epoch import EpochMTL
from model.mtl import MultiTask
from core.tracker import Tracker
from core.result import ResultMTL
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
REPLICATE = int(os.getenv('SLURM_ARRAY_TASK_ID'))
logger.info(f'Running replicate: {REPLICATE}')

# Variables
FRAMEWORK = 'mtl'
TUNE = 'lora'
TASK_NAME = 'combined'

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

###############################################################################
#######                       BUILD DATALOADERS                         #######
###############################################################################

# ESM2
esm2 = ESM2(config.ESM2_MODEL)

# Make collate function
collate_fn = utils.make_collate_fn(esm2.tokenizer)

# Iterate over families
train_loaders = []
val_loaders = []
test_loaders = []
for family in config.FAMILIES:

    # Log family
    logger.info(f"Building dataloaders for family: {family}")
    logger.info('-' * 60)

    # Select pairs from the test, val, test datasets (actually subsets)
    train_pairs = utils.extract_pairs(train_dataset)
    val_pairs = utils.extract_pairs(val_dataset)
    test_pairs = utils.extract_pairs(test_dataset)

    # Select pairs for the current family
    family_train_pairs = [
        pair
        for pair in train_pairs
        if pair.p1.family == family or pair.p2.family == family
    ]
    family_val_pairs = [
        pair
        for pair in val_pairs
        if pair.p1.family == family or pair.p2.family == family
    ]
    family_test_pairs = [
        pair
        for pair in test_pairs
        if pair.p1.family == family or pair.p2.family == family
    ]

    # Re-create datasets for the current family
    family_train_dataset = ProteinPairDataset(
        pairs=family_train_pairs,
        tokenized=tokenized
    )
    family_val_dataset = ProteinPairDataset(
        pairs=family_val_pairs,
        tokenized=tokenized
    )
    family_test_dataset = ProteinPairDataset(
        pairs=family_test_pairs,
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
        shuffle=True,
        drop_last=True # Only training drops last batch
    )
    val_sampler = LengthBatchSampler(
        data_source=family_val_dataset,
        batch_size=config.BATCH_SIZE * len(config.FAMILIES), # Effective batch size
        shuffle=False,
        drop_last=False
    )
    test_sampler = LengthBatchSampler(
        data_source=family_test_dataset,
        batch_size=config.BATCH_SIZE * len(config.FAMILIES), # Effective batch size
        shuffle=False,
        drop_last=False
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

    )
    val_loader = DataLoader(
        family_val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        persistent_workers=True,
        worker_init_fn=seed.seed_worker,
        pin_memory=True,
    )
    test_loader = DataLoader(
        family_test_dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        persistent_workers=True,
        worker_init_fn=seed.seed_worker,
        pin_memory=True,
    )
    logger.info(f'Train dataloader: {len(train_loader)} batches')
    logger.info(f'Validation dataloader: {len(val_loader)} batches')
    logger.info(f'Test dataloader: {len(test_loader)} batches')
 
    # Append loaders
    train_loaders.append(train_loader)
    val_loaders.append(val_loader)
    test_loaders.append(test_loader)

# Create train cycler
cycler = LoaderCycler(train_loaders)

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
model = MultiTask(
    encoder=esm2.model, 
    bottleneck=bottleneck,
    heads=heads,
    tune=TUNE,
    peft_config=LoraConfig(**config.LORA_CONFIG)
)

# Inspector
inspector = ModelInspector(model)
logger.info(f'Model initialized: {model.__class__.__name__}')
logger.info(f'Total parameters: {inspector.num_parameters(trainable=False)}')
logger.info(f'Trainable parameters: {inspector.num_parameters(trainable=True)}')

# Optimizer
encoder_params = [
    parameter 
    for parameter in model.encoder.parameters()
    if parameter.requires_grad
]
head_params = [
    parameter 
    for head in model.heads
    for parameter in head.parameters()
]
optimizer = torch.optim.AdamW([
    {
        'params': encoder_params, 
        'lr': config.LORA_LEARNING_RATE,
        'weight_decay': config.WEIGHT_DECAY
    },
    {
        'params': head_params, 
        'lr': config.HEAD_LEARNING_RATE, 
        'weight_decay': config.WEIGHT_DECAY
    },
])

# Loss function
loss_fn = BinaryFocalLoss(
    alpha=config.FOCAL_LOSS_ALPHA,
    gamma=config.FOCAL_LOSS_GAMMA,
)

# Task weighting strategy
weighting = RankWeighting(
    number_of_tasks=len(config.FAMILIES),
    learning_rate=config.TASK_WEIGHTS_LEARNING_RATE,
    epsilon=config.TASK_WEIGHTS_EPSILON
)

# Epoch
epoch = EpochMTL(        
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    weighting=weighting,
)

# Hook
hook = Hook(
    module=model.bottleneck,
    hook_type='forward',
    auto_register=False,
    capture=True,
    detach=True,
    cpu=True,
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
    train_result_mtl = epoch.train(cycler)

    # Validation
    val_result_stls = []
    for idx, val_loader in enumerate(val_loaders):
        with hook:
            val_result = epoch.evaluate(val_loader, task_idx=idx)
            val_result_stls.append(val_result)

        # Extract hooked representations
        pooled = torch.cat([
            history['input'][0]
            for history in hook.history
        ])
        bottlenecked = torch.cat([
            history['output']
            for history in hook.history
        ])

        hooked = {
            'identifiers': val_result.identifiers,
            'pooled': pooled,
            'bottlenecked': bottlenecked,
            'family': config.FAMILIES[idx]
        }

        # Save hooked representations
        hook_dir = epoch_dir / 'hooks'
        hook_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            hooked,
            hook_dir / f'{idx}.pt'
        )
        
        # Clear hook history
        hook.clear()

        print(f'Hooked representations for task {idx}:')
        print(f'  Pooled: {pooled.shape}')
        print(f'  Bottlenecked: {bottlenecked.shape}')
    
    dfsdfds

    # Testing
    test_result_stls = [
        epoch.evaluate(test_loader, task_idx=idx)
        for idx, test_loader in enumerate(test_loaders)
    ]

    # Task weighting update
    weighting.update(val_result_stls)

    # Aggregate as ResultSTL for tracking and logging
    train_result = train_result_mtl.to_stl()
    val_result = ResultMTL.from_stls(val_result_stls).to_stl()

    # Track
    tracker.track(
        train=train_result,
        validation=val_result,
        model=model
    )

    # Logging
    logger.info(f'Epoch {epoch_idx+1}/{config.EPOCHS}')
    logger.info(f'Train loss = {train_result.loss:.4f}')
    logger.info(f'Validation loss = {val_result.loss:.4f}')
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
        train_result_mtl,
        epoch_dir / 'train.pt'
    )
    torch.save(
        val_result_stls,
        epoch_dir / 'val.pt'
    )
    # Save test results
    torch.save(
        test_result_stls,
        epoch_dir / 'test.pt'
    )

    # Early stopping
    if early_stop(loss=val_result.loss):
        logger.info('Early stopping triggered.')
        break

# Best epoch
logger.info(
    f"Epoch {tracker.best_epoch + 1} had the best " 
    f"validation loss: {tracker.best_loss:.4f}"
)