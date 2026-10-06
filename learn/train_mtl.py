
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