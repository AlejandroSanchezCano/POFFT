
# intact
INTACT_VERSION = '2026-01-09'

# cluster
IEVALUE_THRESHOLD = 1e-5
OVERLAP_THRESHOLD = 0.75
FAMILIES = [
    'HSP70, MreB_Mbl',
    'PK_Tyr_Ser-Thr, Pkinase',
    'Arf, Roc, Ras',
    '7tm_1',
]
NEGATIVE_TO_POSITIVE_RATIO = 10

# learn
SEED = 42
ESM2_MODEL = '35M'
TRAIN_FRACTION = 0.5
VAL_FRACTION = 0.25
TEST_FRACTION = 0.25
BATCH_SIZE = 8
NUM_WORKERS = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0
PATIENCE = 4
MIN_DELTA = 0
FOCAL_LOSS_ALPHA = 0.9
FOCAL_LOSS_GAMMA = 2
REPETITIONS = 10
EPOCHS = 20

# models
CLASSIFICATION_HEAD_HIDDEN_DIMS = [64, 64]
CLASSIFICATION_HEAD_DROPOUT = 0.3
LORA_CONFIG = {
    'task_type': 'FEATURE_EXTRACTION',
    'r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.1,
    'bias': 'none',
    'target_modules': ['query', 'key', 'value'],
    'layers_to_transform': [4, 5, 6, 7, 8, 9, 10, 11], # last 8 layers
}
BOTTLENECK_DIM = 512
BOTTLENECK_DROPOUT = 0.3