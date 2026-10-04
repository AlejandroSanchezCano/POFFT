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
from model.frozen import Frozen
from core.early_stop import EarlyStop
from core.performance import Performance
from core.inspector import ModelInspector
from core.dataset import ProteinPairDataset
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

# Collate function
def collate_fn(batch: list[dict]) -> dict:
    
    # Access elements
    tokenized = [item['input'] for item in batch]
    labels = [item['label'] for item in batch]
    identifiers = [item['identifier'] for item in batch]
    tokenized1, tokenized2 = zip(*tokenized)

    # Dynamic padding
    collator = DataCollatorWithPadding(
        tokenizer=esm2.tokenizer,
        padding=True,
        return_tensors='pt'
    )

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
        'inputs': (collated1, collated2),
        'labels': torch.stack(labels),
        'identifiers': identifiers
    }

# Build dataloaders
train_loader = DataLoader(
    train_dataset,
    collate_fn=collate_fn,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    persistent_workers=True,
    worker_init_fn=seed.seed_worker
)
val_loader = DataLoader(
    val_dataset,
    collate_fn=collate_fn,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    persistent_workers=True,
    worker_init_fn=seed.seed_worker
)
test_loader = DataLoader(
    test_dataset,
    collate_fn=collate_fn,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    persistent_workers=True,
    worker_init_fn=seed.seed_worker
)

###############################################################################
#######                           MODEL SETUP                           #######
###############################################################################

# Models
head = ClassificationHead(
    input_dim=esm2.hidden_size * 2,  # Concatenated representations
    hidden_dims=config.CLASSIFICATION_HEAD_HIDDEN_DIMS,
    dropout=config.CLASSIFICATION_HEAD_DROPOUT,
)
model = Frozen(encoder=esm2.model, head=head)
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
pos_weight = torch.tensor([config.NEGATIVE_TO_POSITIVE_RATIO], device='cuda')
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Epoch
epoch = Epoch(        
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
)

###############################################################################
#######                        TRAIN AND EVALUATE                       #######
###############################################################################

# Loop over repetitions
for rep in tqdm(range(config.REPETITIONS), desc='Repetitions', unit='rep'):
    
    # Loop over epochs
    for epoch_idx in tqdm(range(config.EPOCHS), desc='Epochs', unit='epoch'):

        # Train
        train = epoch.train(train_loader)
        # Evaluate
        val = epoch.evaluate(val_loader)
        test = epoch.evaluate(test_loader)
        # Performance
        train_perf = Performance(
            true_labels=train.labels,
            predicted_logits=train.logits,
        )
        test_perf = Performance(
            true_labels=test.labels,
            predicted_logits=test.logits
        )
        # Logging
        logger.info(f'Epoch {epoch_idx+1}/{config.EPOCHS}')
        logger.info(f'Train loss = {train.loss:.4f}')
        logger.info(f'Test loss = {test.loss:.4f}')
        logger.info(f'Train balanced accuracy = {train_perf.balanced_accuracy:.4f}')
        logger.info(f'Test balanced accuracy = {test_perf.balanced_accuracy:.4f}')
        # Early stopping
        if early_stop(loss=result.loss, model=model):
            logger.info('Early stopping triggered.')
            break

    # Save model
    out_dir = paths.MODELS / model.__class__.__name__.lower() / rep
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(early_stop.best_model, out_dir / 'model.pt')