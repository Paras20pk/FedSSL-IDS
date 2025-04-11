from tensorflow.keras import layers, models

def create_ssl_encoder(input_shape=(100,)):
    """Base encoder for SSL (replace with your architecture)"""
    return models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.LayerNormalization()
    ])

def add_classification_head(encoder, num_classes=2):
    """Add attack classification head"""
    return models.Sequential([
        encoder,
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])